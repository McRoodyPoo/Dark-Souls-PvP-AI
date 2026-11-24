"""The bot's decision making: port of AIDecisions.c, MindRoutines.c, AIMethods.c,
SubRoutines.c and the main loop body from Source.c.

Flow per tick (same as metal-crow's MainLogicLoop):
  1. instinct(): hard-coded reactions (dodge an incoming hit, take a backstab, ...)
  2. defense mind: backstab-prediction net + backstab-escape rules
  3. dodge(): start or continue a dodge "subroutine" (multi-tick input sequence)
  4. attack mind: approach / attack-now net
  5. attack(): start or continue an attack subroutine
  6. retire subroutines marked as exiting

Behavior changes from the C, each fixing a bug:
  * Single-threaded and deterministic (C raced two threads on AttackChoice).
  * Steering is camera-relative (no Cheat Engine camera lock).
  * Subroutine history stores (kind, id); the C stored dodge and attack ids in one
    array where e.g. StandardRoll == MoveUp == 1, so "I just rolled" fired after moving.
  * Omnistep exits after its stick window instead of before it ("kind of not working").
  * Dead angle records itself as DEAD_ANGLE, not GHOST_HIT.
  * Pivot backstab has an exit (the C version never ended).
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import geometry as G
from . import tables as T
from .character import Character, Sub, current_attack
from .gamepad import (CIRCLE, DDOWN, DLEFT, L1, L2, MIDDLE, R1, R2, R3, SQUARE, TRIANGLE,
                      XLEFT, XRIGHT, YBOTTOM)
from .history import History, is_weapon_shield, poise_damage_for_attack, poise_damage_known
from .threat import ThreatFeatures
from .hitmemory import HitMemory
from .steer import Steering


class Def(IntEnum):
    NONE = 0
    STANDARD_ROLL = 1
    BACKSTEP = 2
    OMNISTEP_BACKWARDS = 3
    COUNTER_STRAFE_LEFT = 4
    COUNTER_STRAFE_RIGHT = 5
    L1_ATTACK = 6
    REVERSE_ROLL_BS = 7
    TOGGLE_ESCAPE = 8
    PERFECT_BLOCK = 9
    PARRY = 10
    HOLD_BLOCK = 11  # hold L1 through their swing (no stamina for a roll)
    TURN_TO_FACE = 12  # unlock and turn to face them (incoming pivot backstab)
    SIDESTEP = 13      # run across their line of fire (projectiles, flame streams)
    FLEE_ROLLS = 14    # fire tempest: roll-spam straight away from them, toward open floor


class Atk(IntEnum):
    NONE = 0
    MOVE_UP = 1
    GHOST_HIT = 2
    DEAD_ANGLE = 3
    BACKSTAB = 4
    TWO_HAND = 5
    SWITCH_WEAPON = 6
    HEAL = 7
    PIVOT_BS = 8
    SPACE = 9        # neutral game: hold just outside their reach, strafing unpredictably
    PUNISH = 10      # one R1 facing them, into a whiff or a stuck animation
    KICK = 11        # stick flick toward them + R1: point-blank answer (user: kicks are safe)
    RACE = 12        # neutral: run a tight circle round them to their back, like the user
    POKE = 13        # plain R1 from their front/side during a race (user: 56% connected)
    WAIT_MIDDLE = 14  # they're outside the arena walls: wait in the middle, facing them
    LINE_UP = 15      # at their back but facing wrong: unlocked turn into them, then the kick
    FOOTSIE = 16      # neutral: locked on, step in to threaten / back out to bait (user)
    RUN_BS = 17       # they move forward: locked-on sprint at them + kick (run backstab, user)
    JUMP = 18         # jumping attack: stick at them + R2 (vs turtles, user 2026-11-24)
    ROLL_GHOST = 19   # roll past their shield, rolling attack with a ghost turn (vs turtles)
    TURTLE = 20       # they're turtling: close in and mix kicks / jumping attacks / roll ghosts
    CHARGE = 21       # they camp / cast / buff at range: block+sprint straight at them (user)
    OKI = 22          # they're down (our backstab, kickstun): stand at their back for the wake-up
    KEEP_OUT = 23     # fire surge going: stay out of its range
    KNIVES = 24       # corner camper: back off, throw poison knives until they leave the corner


ROLL_DEFS = {Def.STANDARD_ROLL, Def.BACKSTEP, Def.OMNISTEP_BACKWARDS, Def.REVERSE_ROLL_BS}

# everything that presses R1 (blocked for a moment after getting up)
R1_ATTACK_TYPES = {Atk.JUMP, Atk.ROLL_GHOST, Atk.TURTLE,
                   Atk.PUNISH, Atk.GHOST_HIT, Atk.KICK, Atk.BACKSTAB, Atk.DEAD_ANGLE, Atk.POKE,
                   Atk.RUN_BS}


class Priority(IntEnum):
    NEUTRAL = 0      # no override
    DELAY = 1        # enemy winding up: don't commit to an attack
    DODGE = 2        # must dodge now (defense id set)
    ATTACK = 3       # must attack now (attack id set)


INACTIVE, ACTIVE, EXITING = 0, 1, 2
FISTS = 900000
EQUIP_COOLDOWN_MS = 2000
TOGGLE_RACE_MS = 4000            # after a toggle escape, a race this soon stays one-handed
PIVOT_SIDE_M = 1.3        # how wide to go round the enemy
PIVOT_BEHIND_M = 0.8      # the spot behind them we aim for
BS_CANCEL_RANGE = 3.0   # 2x backstab range: close enough that a backstab is a real threat

# AIMethods.h timings (ms)
R3_TRIGGER = 50
CAMERA_ROTATE_AFTER_LOCKON = 180
GAME_REGISTER_ACTION = 170
REVERSE_ROLL_TOTAL_S = (R3_TRIGGER + CAMERA_ROTATE_AFTER_LOCKON + GAME_REGISTER_ACTION + 50) / 1000


@dataclass
class Config:
    player_weapon_range: float = 2.5   # original hardcoded values
    enemy_weapon_range: float = 6.0
    prefer_two_hand: bool = True       # metal-crow's build: two-hand when out of range
    # metal-crow's build kept a weapon in the second left slot. With nothing to swap to
    # (e.g. fists in both hands) this re-fires every 500 ms forever, so it's off by default.
    switch_off_fists: bool = False
    cardinal_rolls: bool = False       # original only rolled along one stick axis
    use_attack_net: bool = True
    use_defense_net: bool = True
    # Poise through hits that can't stagger us and punish with a (pivot) backstab instead
    # of rolling. The original rolled from everything.
    poise_tank: bool = True
    tank_margin: float = 5.0           # our poise must beat their poise damage by this much
    tank_min_hp_frac: float = 0.4      # below this share of max HP, roll anyway
    tank_range: float = 3.5            # pivot only when this close; farther away, just walk in
    tank_max_dmg_frac: float = 0.08    # don't tank attacks known to cost more than this share of max HP (0.15 let it eat 287s)
    roll_past: bool = True             # roll past their side toward their back, not away
    default_enemy_reach: float = 3.0   # until we've been hit and learned their weapon's reach
    spacing_margin: float = 0.8        # hold this far outside their reach in neutral
    # Havel + Great Club build (user, 2026-11-23): no R1s at all - only backstabs (kick input),
    # and a ghost strike instead of an R1 out of a roll. False = the earlier pokes/punish style.
    backstab_only: bool = True


@dataclass
class Instinct:
    priority: Priority = Priority.NEUTRAL
    defense: Def = Def.NONE
    attack: Atk = Atk.NONE


BACKSTAB_ANIM = 225              # animation type of a backstab (ours or theirs)

# Reach per attack (user 2026-11-24: "no idea what to do against some weapons ... stands there
# and gets hit by spears, magic"). Recordings: lunging attacks started 2.5-5.3 m out (R2_2H up
# to 6.9 m), past the flat 3 m guess for an unknown weapon; Fire Surge hit 15 times from ~4 m
# and was never dodged; nothing handled projectiles at all.
LUNGE_ANIM = {41, 53, 59, 60, 88, 89, 90, 103, 113, 114, 115, 116}   # rolling/jumping/backstep/R2
LUNGE_REACH_M = 5.0
LONG_WEAPON_BONUS_M = 1.0        # spears, halberds, scythes (weapon class 1000-1249)
PROJECTILE_ANIM = {65, 66, 67, 69, 70, 143, 144, 147, 148, 157, 158, 173, 174}  # bows, sorceries,
                                 # lightning spears, fireballs: windups and casts
FLAME_ANIM = {161, 162}                                             # fire whip (surge: SURGE_ANIMS)
FLAME_REACH_M = 6.0
SPELL_REACH = {145: 4.0, 146: 4.0, 155: 5.0, 156: 5.0, 163: 5.0, 164: 5.0, 167: 2.5, 168: 2.5}
PROJECTILE_MIN_M = 4.0           # closer than this the normal roll logic handles it
PROJECTILE_MAX_M = 30.0
AIM_CONE_DEG = 35.0              # they're aiming at us
SIDESTEP_MS = 900
# Projectile timing (live 2026-11-24 vs a sorcerer: the sidestep started at the cast and ended
# just as the spell was released ~1 s later; it tracked us and hit, 37 hits / 8.7k damage).
# Hit time = cast start + release + distance / speed; roll so the i-frames cover it. The
# sorcery release is learned from hits ("proj_release:<family>" in hit memory).
PROJ_FAMILY = {157: "sorcery", 158: "sorcery", 173: "fireball", 174: "fireball",
               143: "miracle", 144: "miracle", 147: "spear", 148: "spear",
               65: "bow", 66: "bow", 67: "bow", 69: "bow", 70: "bow"}
PROJ_RELEASE_S = {"sorcery": 1.0, "fireball": 0.9, "miracle": 0.8, "spear": 0.8, "bow": 0.5}
PROJ_SPEED_MPS = {"sorcery": 9.0, "fireball": 8.0, "miracle": 15.0, "spear": 20.0, "bow": 25.0}
PROJ_CAST_PHASE = {158, 174, 144, 148, 67, 70}   # first seen in the release phase: start was earlier
PROJ_ROLL_LEAD_S = 0.2           # start the roll this long before the hit
PROJ_WINDOW_S = 0.6              # ...and give up this long after it
REACH_CAP_M = 4.5                # a learned non-lunge reach never counts for more than this
LUNGE_REACH_CAP_M = 7.0          # (+ the long-weapon bonus): one misattributed hit can't make
                                 # the bot roll from across the arena
LINE_UP_M = 1.45                 # at their back within this... (1.3 until 2026-11-24)
LINE_UP_DEG = 55.0               # ...and this far off it: turn in and take the backstab
KICK_FACING_MAX = 45.0           # only kick-backstab when facing them this closely (2026-11-23:
                                 # kick inputs facing 53-61 deg off came out as club R1s; 2026-11-24
                                 # at 35 it passed up 9 lined-up chances facing 35-50 off)
# Stick centred this long before a kick / jump flick. The game runs at 30 fps: a flick only
# counts if a game frame saw the stick centred, then the next saw it pushed with the button.
# Live 2026-11-24: 3 of 6 jump attempts after an 8 ms centre came out as a plain R2 heavy
# (anim 115), both after >= 67 ms were real jumps (113). User: "Never press R2 unless it's a
# jump attack". Two full frames.
KICK_NEUTRAL_MS = 70
FLICK_GIVE_UP_MS = 1500          # still mid-animation this long: drop the jump, never a plain R2
OKI_AFTER_BS_MS = 3000           # after our backstab ends: they're getting up
OKI_AFTER_KICK_MS = 1200         # kick connected: they're stunned
# User 2026-11-24: "if you manage to land a kick and the kick wasn't blocked, race for a
# backstab". Landed = it took HP off them while our kick animation played; not blocked = their
# shield wasn't up (a blocked kick does no HP damage). Recordings: 3 of ~150 kicks landed, ~20 HP.
KICK_ANIMS = {52, 112}
SHIELD_ANIMS = {75, 76, 77, 80, 81}
KICK_HIT_WINDOW_MS = 800
KICK_RACE_MS = 2500              # race for their back this long after a clean kick
# User 2026-11-24: "if the opponent's back is facing you, go for the backstab even if it means
# taking damage". Back to us = we're within BACK_TURNED_DEG of straight behind them, this close.
# Then: no dodging (the old check rolled from swings aimed the other way), go for their back.
# Only exception: a hit we know would kill us.
BACK_TURNED_DEG = 60.0
BACK_CHASE_M = 5.0
# Kick backstab as ONE motion (user 2026-11-24): kick (aimed at them: unlocked, kicks aim
# freely) > unlock > block + sprint together from the first instant (max speed ASAP) > let go of
# block > circle to their back > backstab. Every kick that isn't a backstab chains into it.
KICK_UNLOCK_MS = 30
# Turtles (user 2026-11-24: "did nothing in a corner while bot king wiggled around"): standing
# still with the shield up, facing us, this long: mix kicks (mostly) and jumping attacks, and
# rolling ghost strikes. Recordings: player B sat 2-7 s shield up and the bot just circled.
TURTLE_MS = 1500
TURTLE_M = 4.5
TURTLE_SHIELD_FRAC = 0.5
TURTLE_MIX = (("kick", 0.65), ("jump", 0.25), ("roll", 0.10))
TURTLE_GAP_MS = (500, 1200)
TURTLE_REACH = {"kick": 1.1, "jump": 2.4, "roll": 2.8}
TURTLE_APPROACH_MS = 2000
# The "no-cast" loop (user 2026-11-24: "player G is just doing the no cast animation and bot king is
# just angling for a backstab. player G is literally exploiting him"): animation type 132 (anim id
# 6399), 40% of that fight (310 of 782 s). Not an attack, so the bot raced for their back while
# they turned with it, round and round. In it within reach: hit them directly instead (the turtle
# mix: kicks mostly, jumping attacks, rolling ghost strikes), and stop racing.
NO_CAST_ANIMS = {132}
# Same fight, later: mostly they just stood (anim 0) and turned with us - races at 0.9 m that
# never gained on their back, then "they're behind us: turn", race again, for minutes. A race
# that gains nothing in RACE_NO_PROGRESS_MS: they turn with us. No races against them for
# RACE_FUTILE_MS; hit them with the turtle mix-up instead.
RACE_FUTILE_MS = 6000
# User 2026-11-24: "player G is just standing there ... in the corner, buff, and use pyros while bot
# king stands there and eats it". Two causes: the caster's MLGS was learned to reach 5.95 m (its
# projectile R2), so neutral spacing hovered at 6-7 m - pyromancy range; and nothing went in on a
# standing opponent at range. Spacing now assumes at most SPACING_REACH_CAP of melee reach (the
# dodge logic still uses the real reach), and a camping / casting / buffing opponent gets charged.
SPACING_REACH_CAP = 3.5
CHARGE_MIN_M, CHARGE_MAX_M = 2.0, 15.0
CHARGE_STATIC_MS = 1200
CHARGE_HANDOFF_M = 1.8           # close enough: the race / line-up / kick takes it from here
CHARGE_MAX_MS = 4000
CAST_ANIMS = set(range(141, 175)) - {163, 164}   # spells, pyromancies, buffs (not fire tempest / surge)
# Fire tempest / fire storm (user 2026-11-24): roll-spam directly away from the caster, toward the
# most open floor, until clear of it.
FIRE_STORM_ANIMS = {163, 164}
FLEE_SAFE_M = 8.5
FLEE_ROLL_GAP_MS = 700
FLEE_TAP_MS = 90
FLEE_MAX_MS = 4000
# Fire surge (user 2026-11-24): keep out of its range; inside it, roll away.
SURGE_ANIMS = {135, 136, 137, 138, 139, 140}
SURGE_REACH_M = 5.5              # recordings: hit from up to 5.0 m (one cast started 6.15 m out)
SURGE_SAFE_M = 6.5
# Corner campers (user 2026-11-24): back up and throw poison throwing knives until they leave the
# corner by 0.2 m.
CORNER_M = 2.2                   # within this of an arena corner (a polygon vertex)
CORNER_STILL_MS = 1500
KNIFE_RANGE_M = 6.0              # back off to about this far first
KNIFE_LEFT_M = 0.2               # they moved this far from their spot: stop
KNIFE_START_M = 3.5              # only from at least this far: up close it's a fight, not camping
                                 # (live 2026-11-24 it kept starting at 0.8-1.8 m mid-race and quitting)
KNIFE_GAP_MS = 1300
# Locked on for every throw (user 2026-11-24: "HE NEEDS TO BE LOCKED ON WHEN THROWING KNIVES"):
# hold R3 like the other lock-ons, check the game's lock flag, press again if it didn't take,
# re-lock if it drops, and never throw unlocked.
KNIFE_LOCK_HOLD_MS = 50
KNIFE_LOCK_RETRY_MS = 400
KNIFE_LOCK_TRIES = 5
KNIFE_TAP_MS = 120
KNIFE_MAX = 20
KNIFE_MAX_MS = 40000
KNIFE_COOLDOWN_MS = 4000
KNIFE_EMPTY_BACKOFF_MS = 300_000
# Wake-up machine (user 2026-11-24: "he is here to cook you breakfast"): they're down - after our
# backstab, kickstunned or staggered - so stand at their back and kick-backstab the moment they're up.
OKI_SPOT_M = 0.75
OKI_ARRIVE_M = 0.35
OKI_MAX_MS = 4500
# "Kick their back the very frame they wake up" (user 2026-11-24): at the spot, facing them, stick
# centred and unlocked so the kick goes out with zero delay; and with a learned wake-up time
# ("wake_ms:bs" after our backstab, "wake_ms:stun" after a kickstun / stagger, in hit memory) press
# WAKE_LEAD_MS before it so the kick lands on their first vulnerable frame.
WAKE_LEAD_MS = 50
OKI_FACE_DEG = 20.0
# Kicks lead the target: aim where they'll be this long from now (user: "aiming his kicks
# perfectly at the player every time").
KICK_LEAD_S = 0.15
POST_ROLL_MS = 1200               # this soon after a roll an attack would be a roll-R1
PUNISH_RUN_M = 1.5               # whiff punish: run in this far past our reach, then R1
RACE_EXTRA_M = 0.5               # race from spacing range (hold + this) inward
RACE_RADIUS_M = 0.85             # circle this close (user: 0.6-1.3 m in the last 0.6 s)
RACE_STEP_DEG = 70.0             # aim this far round them each tick
RACE_MAX_MS = 2200
RACE_IF_STATIC_MS = 1200         # race round them only once they've stood still this long
FOOTSIE_IN_M = 0.2               # threaten: step to their reach + this
FOOTSIE_OUT_M = 1.4              # bait: back out to their reach + this
FOOTSIE_PHASE_MS = (450, 1000)
RUN_BS_MIN_M = 1.5               # run backstab when they come forward from this...
RUN_BS_MAX_M = 5.0               # ...to this far away
RUN_BS_CLOSING_MPS = 2.3         # their speed toward us that counts as "coming forward"
RUN_BS_KICK_M = 0.8              # kick at contact (2026-11-23: kicks at 1.3 m all hit air in their face)
RUN_BS_CONTACT_S = 0.12          # ...or when we'd be there within this, at the current closing speed
RUN_BS_BAIL_M = 2.0              # they stop coming while we're still this far out: back to footsies
RUN_BS_MAX_MS = 1500
SPRINT_MIN_RUN_M = 2.2           # only start a sprint with this much ground left (it can't stop short)
SPRINT_HOLD_MS = 400             # once B is held to sprint, hold it at least this long (a short tap rolls)
RACE_BLOCK_MS = 160              # hold block this long while setting off (reach max speed sooner)
SPRINT_MIN_MS = 450              # once sprinting, hold it at least this long (a short tap is a roll)
RACE_LOCK_DEG = 40.0             # this close to their back...
RACE_LOCK_M = 1.4                # ...and this near: lock on and take the backstab
RACE_NO_PROGRESS_MS = 800        # not gaining on their back this long: go the other way
# User 2026-11-23: backstab attempts use the kick input ("kick confirm": the kick inputs give
# a backstab when it's on, a kick when it isn't - an option select); no kicks in neutral from
# the front; poke with plain R1s from the front/side instead.
POKE_M = 2.0                     # user's pokes: within 2 m, from their front/side
POKE_MIN_OFF_BACK = 60.0         # (closer to their back than this is kick territory)
POKE_EVERY_MS = (1200, 2500)
# Incoming pivot (unlocked) backstab, from every recording to 2026-11-23: in the 0.5 s before
# a host backstab they moved sideways across our line of sight at a median 2.1 m/s (other
# close neutral: 0.3). Locked on we turn at lock-on speed and they run round us; unlocked,
# the stick turns us faster (user).
PIVOT_DETECT_M = 1.3
PIVOT_LATERAL_MPS = 2.0
TURN_COOLDOWN_MS = 800
TURN_MAX_MS = 450
TANK_KEEP_FRAC = 0.20            # backstab-only build: trade only if this much HP is left after
TANK_UNKNOWN_FRAC = 0.25         # (damage not learned yet: assume this share of max HP)
OUTSIDE_M = 0.7                  # target this far outside the walkable area = beyond the walls
CENTER_PULL_M = 2.0              # spacing within this of a wall drifts toward the middle
WHIFF_MARGIN_S = 0.25            # punish only this long after their attack's hit time
PROBE_EVERY_MS = (2000, 3500)    # neutral: step into their reach edge to bait a swing
PROBE_MS = 400
NO_STRAFE_AFTER_ROLL_MS = 800
STRAFE_MIN_M = 1.6               # no counter-strafing closer than this
THREAT_ON = 0.3                   # threat net: warned before 80% of held-out backstabs, false
                                  # alarm 2.4% of the time (the old net: 98% but on 29% of the time)
WAKEUP_NO_R1_MS = 1500           # after getting up: no R1 of any kind (user: "never R1 on wake up")
FOLLOW_M = 1.6                   # host neutral this close...
FOLLOW_MS = 400                  # ...this long while we back off = they're following: roll out
ESCAPE_ROLL_COOLDOWN_MS = 1500
JAB_M = 0.9                      # pinned closer than this (or against a wall): R1 them
VICTIM_ESCAPE_EVERY_MS = 600
# stamina (measured 2026-11-23 on this character: roll ~30, attack ~34, max 192; blocking
# halves regen). User: "remember: stamina".
ROLL_STAMINA = 30
ATTACK_STAMINA = 34
POKE_STAMINA = ATTACK_STAMINA + ROLL_STAMINA     # an attack that still leaves a roll
WAIT_OUT_M = 2.0                 # knocked down with them this close...
WAIT_OUT_CHANCE = 0.4            # ...sometimes don't wake up early (mixes up their timing)
# Dials the per-opponent styles (adapt.py) may move; these are the hand-tuned defaults.
DEFAULT_STYLE = {"race_static_ms": RACE_IF_STATIC_MS, "run_bs_closing": RUN_BS_CLOSING_MPS,
                 "footsie_in_m": FOOTSIE_IN_M, "footsie_out_m": FOOTSIE_OUT_M,
                 "wait_out_chance": WAIT_OUT_CHANCE}
CLOSE_M = 1.2                    # host this close in neutral = backstab setup: get out
DISENGAGE_COOLDOWN_MS = 1500


class Brain:
    def __init__(self, steering: Steering, defense_net=None, attack_net=None,
                 config: Config | None = None, rng: random.Random | None = None,
                 hits: HitMemory | None = None):
        self.steer = steering
        self.hits = hits if hits is not None else HitMemory(path=None)
        self._last_hp = None
        self.defense_net = defense_net
        self.attack_net = attack_net
        # Backstab threat net trained on the bot's own fights (tools/train_threat.py); when
        # loaded it replaces metal-crow's defense net (held-out AUC 0.86 vs 0.59, 2026-11-24).
        self.threat_net = None
        self.threat = ThreatFeatures()
        self.style = dict(DEFAULT_STYLE)     # set per exchange by adapt.Adapter
        self._e_anim_last = 0
        self._atk_start_d = None
        self._proj = None                    # a projectile cast at us: {"t0", "fam", "rolled"}
        self._kick_anim = False
        self._kick_watch = None              # (time our kick started, their hp then, shield up)
        self._kick_race_until = -1e9
        self._back_turned = False
        self._back_chase_until = -1e9
        self._chain_until = -1e9             # kick chain: block + sprint at once while racing
        self._shield_hist: deque = deque()   # (t, shield up) of the opponent, for turtle checks
        self._turtle_next = -1e9
        self._race_futile_until = -1e9
        self._turtle_move = None
        self._turtle_go_t = None
        self._kick_unlock = False
        self._e_prev = None                  # (t, x, y) last tick, for their velocity
        self.items = None                    # escape.DugoutEscape: quick-item position (knives)
        self._knife_next = -1e9
        self._kn = {}
        self._edown = None                   # (kind, start ms) while they're down
        self._stick_centred_prev = False     # unknown at start: wait the full centring
        self._stick_centred_since = -1e9
        self._flick_lead = KICK_NEUTRAL_MS
        self._flee_set, self._flee_safe = FIRE_STORM_ANIMS, FLEE_SAFE_M
        self._e_vel = (0.0, 0.0)
        self._side = 1.0
        self.cfg = config or Config()
        self.rng = rng or random.Random()
        self.history = History()
        self.dodge_state, self.dodge_type, self.t_def = INACTIVE, Def.NONE, 0.0
        self.attack_state, self.attack_type, self.t_atk = INACTIVE, Atk.NONE, 0.0
        self._pivot_angle = None
        self._pivot_behind_t = None
        self.arena = None            # ArenaMap: walls of the arena (arena_map.py), or None
        self._roll_heading = None    # world heading for a disengage roll, else None
        self._pending_heading = None
        self._disengage_t = -1e9
        self._jab_now = False
        self._race_side = 0.0
        self._footsie_in = True
        self._footsie_flip_at = None
        self._sprint_since = None        # B held for a sprint since (no roll on early release)
        self._sprint_req = False
        self._static_since = None
        self._run_kick_t = None
        self._race_sprint_t = None
        self._race_locking = False
        self._next_poke_at = -1e9
        self._last_roll_t = -1e9
        self._toggle_escape_t = -1e9
        self._e_track = []               # recent (t, x, y) of the enemy, for their lateral speed
        self._turn_t = -1e9
        self._turn_relock = False
        self._race_best = None           # (smallest angle off their back so far, when)
        self._was_down = False
        self._followed_since = None
        self._wait_out = None            # this knockdown: stay down (True) or wake early
        self._escape_roll_t = -1e9
        self._woke_t = -1e9
        self._roll_from = None
        self._last_damage_t = -1e9
        self._punish_r1_t = None
        self._victim_escape_t = -1e9
        self._next_probe_at = None    # first probe a couple of seconds into spacing
        self._probe_until = -1e9
        self._said_wall = 0.0
        self._last_equip_change = -1e9
        self._strafe_locked: bool | None = None
        self._cs_prev = None         # (enemy bearing, time) for the counter-strafe
        self._cs_side = 0            # counter-strafe side: +1 / -1, 0 = not chosen yet
        self._dodged_swing = None    # (anim id, anim type) of the attack we already dodged
        self._toggle_t = None         # last lock-on toggle during a pivot
        self._bs_attempt_t = -1e9      # when we last pressed R1 for a backstab
        self._bs_method = "kick"
        self._oki_until = -1e9         # they're getting up / stunned: stance-cancel is safe
        self._kick_t = -1e9            # when our last kick went out
        self._was_backstabbing = False
        self._strafe_dir = 1.0
        self._strafe_flip_at = 0.0
        self._said_swing = None
        self._max_hp = 0
        self.now = 0.0
        self.last_instinct = Instinct()
        self.log: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _say(self, msg: str) -> None:
        self.log.append(msg)

    def _last_self(self):
        return self.history.last_subroutines_self[0]

    def _finish(self, kind: str, sub_id) -> None:
        if kind == "def":
            self.dodge_state = EXITING
        else:
            self.attack_state = EXITING
        self.history.push_subroutine_self((kind, int(sub_id)))

    def _toward(self, p, e, offset_deg: float = 0.0):
        """Stick toward the enemy, rotated by offset_deg (same sign as the C's `angle - offset`)."""
        h = G.heading_to(p, e) - math.radians(offset_deg)
        return self.steer.stick_for_heading(h, self.cam_yaw)

    @staticmethod
    def _they_face_us(p, e, cone_deg: float = 70.0) -> bool:
        """We're in their front cone (they can circle us; we can't backstab them)."""
        fx, fy = G.facing_vector(e)
        rx, ry = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        n = math.hypot(rx, ry) or 1e-6
        return (rx * fx + ry * fy) / n > math.cos(math.radians(cone_deg))

    def _open_heading(self, p, e) -> float:
        """Away from them, bent toward whichever side has the most open floor."""
        away = G.heading_to(p, e) + math.pi
        walls = self._walls(p)
        if walls is None:
            return away
        best, best_score = away, -1e9
        for off in (0, 30, -30, 60, -60, 90, -90):
            h = away + math.radians(off)
            score = min(walls.free_run(p.loc_x, p.loc_y, h, 3.0), 3.0) - 0.004 * abs(off)
            if score > best_score:
                best, best_score = h, score
        return best

    def _walls(self, p):
        """The arena map, if we're in that arena (hosts can summon us anywhere)."""
        m = self.arena
        return m if m is not None and m.here(p) else None

    def _set_stick(self, pad, xy) -> None:
        pad.lx, pad.ly = xy

    def in_dodge(self) -> bool:
        return self.dodge_state != INACTIVE

    def in_attack(self) -> bool:
        return self.attack_state != INACTIVE

    def in_any(self) -> bool:
        return self.in_dodge() or self.in_attack()

    def _override_low_priority(self) -> None:
        if self.dodge_type in (Def.COUNTER_STRAFE_LEFT, Def.COUNTER_STRAFE_RIGHT):
            self.dodge_state, self.dodge_type = INACTIVE, Def.NONE
            self._strafe_locked = None
        if self.attack_type in (Atk.MOVE_UP, Atk.SWITCH_WEAPON, Atk.PIVOT_BS, Atk.SPACE, Atk.RACE,
                                Atk.WAIT_MIDDLE, Atk.FOOTSIE):
            # PIVOT_BS too: live 2026-11-22 the bot tanked one hit, started pivoting, and the
            # pivot blocked the roll from the follow-up swing (poise 9.5 left) -> staggered.
            self.attack_state, self.attack_type = INACTIVE, Atk.NONE
            self._pivot_angle = self._pivot_behind_t = None

    def _poke_ok(self, p, e, atk_choice) -> bool:
        """Plain R1 from their front/side while racing (the user: 55 pokes, 56% connected,
        backstabbed after only 7 - it breaks up their circling)."""
        if self.cfg.backstab_only:
            return False                   # no R1s on this build
        racing = self.in_attack() and self.attack_type == Atk.RACE and self.now - self.t_atk >= 250
        if not racing or e.sub != Sub.NEUTRAL or self.now < self._next_poke_at:
            return False
        dist = G.distance(p, e)
        if dist > min(POKE_M, p.weapon_range) or p.stamina < POKE_STAMINA or self.waking_up():
            return False
        fx, fy = G.facing_vector(e)
        rx, ry = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        off_back = math.degrees(math.atan2(abs(rx * fy - ry * fx), -(rx * fx + ry * fy)))
        if off_back < POKE_MIN_OFF_BACK:
            return False                   # near their back: that's the backstab's job
        self._next_poke_at = self.now + self.rng.uniform(*POKE_EVERY_MS)
        self._say(f"poke ({dist:.1f} m, {off_back:.0f} deg off their back)")
        return True

    def _enemy_reach(self, e) -> float:
        """How far their CURRENT attack can reach us from (neutral: their weapon's reach)."""
        base = self.hits.reach(e.r_weapon)
        r = (base + 0.3) if base else self.cfg.default_enemy_reach
        at = e.animation_type
        if at in SPELL_REACH:
            return SPELL_REACH[at]
        if at in FLAME_ANIM:
            return FLAME_REACH_M
        if at in SURGE_ANIMS:
            return SURGE_REACH_M
        long = 1000 <= e.r_weapon // 1000 < 1250
        if long and not base:
            r += LONG_WEAPON_BONUS_M
        if at in LUNGE_ANIM:
            r = max(r, LUNGE_REACH_M + (LONG_WEAPON_BONUS_M if long else 0.0))
        start = self.hits.reach_start(e.r_weapon, at, e.anim_id) if at else None
        if start:
            cap = (LUNGE_REACH_CAP_M if at in LUNGE_ANIM else REACH_CAP_M) + (LONG_WEAPON_BONUS_M if long else 0.0)
            r = max(r, min(start, cap) + 0.3)
        return r

    def _proj_hit_in(self, dist: float) -> float | None:
        """Seconds until the projectile cast at us should land (negative: it should have)."""
        if self._proj is None:
            return None
        fam = self._proj["fam"]
        release = PROJ_RELEASE_S[fam]
        if fam == "sorcery":
            release = self.hits.value("proj_release:sorcery", release)
        hit_at = self._proj["t0"] / 1000 + release + dist / PROJ_SPEED_MPS[fam]
        return hit_at - self.now / 1000

    def _ranged_dodge(self, p, e, dist: float) -> bool:
        """A projectile aimed at us (until it's time to roll), or a flame stream we're in."""
        at = e.animation_type
        if self.is_down(p) or p.in_backstab or e.in_backstab:
            return False
        if at in PROJECTILE_ANIM:
            left = self._proj_hit_in(dist)
            # get out of line while they wind up, but stop in time for the roll
            return (PROJECTILE_MIN_M < dist < PROJECTILE_MAX_M and self._they_face_us(p, e, AIM_CONE_DEG)
                    and (left is None or left > PROJ_ROLL_LEAD_S + 0.15))
        if at in FLAME_ANIM:
            return dist < FLAME_REACH_M and self._they_face_us(p, e, 45.0)
        return False

    def _back_to_us(self, p, e, dist: float) -> bool:
        """Their back is turned to us (we're behind them, close enough to go for it)."""
        if (dist > BACK_CHASE_M or e.in_backstab or p.in_backstab or self.is_down(p) or e.hp <= 1
                or p.animation_type == BACKSTAB_ANIM or e.animation_type == BACKSTAB_ANIM
                or e.animation_type in FIRE_STORM_ANIMS or e.animation_type in SURGE_ANIMS):
            return False                       # (not during a backstab, ours or theirs)
        fx, fy = G.facing_vector(e)
        rx, ry = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        behind = -(rx * fx + ry * fy)
        return behind > 0 and math.degrees(math.atan2(abs(rx * fy - ry * fx), behind)) <= BACK_TURNED_DEG

    def _would_kill(self, p, e) -> bool:
        """Their current attack is known to hit harder than the HP we have left."""
        hurt = self.hits.get(e.r_weapon, e.animation_type) if e.animation_type else None
        return bool(hurt) and hurt >= p.hp - 1

    def _st(self, key: str):
        return self.style.get(key, DEFAULT_STYLE[key])

    def is_down(self, p) -> bool:
        """Getting up from their backstab / a knockdown / a stagger."""
        return (p.in_backstab and not self._we_backstabbed()) or p.sub == Sub.POISE_BROKEN

    def waking_up(self) -> bool:
        return self.now - self._woke_t < WAKEUP_NO_R1_MS or self._was_down

    def _only_low_priority(self) -> bool:
        """Nothing running that _override_low_priority wouldn't cancel."""
        dodge_ok = not self.in_dodge() or self.dodge_type in (Def.COUNTER_STRAFE_LEFT,
                                                               Def.COUNTER_STRAFE_RIGHT)
        atk_ok = not self.in_attack() or self.attack_type in (Atk.MOVE_UP, Atk.SWITCH_WEAPON,
                                                              Atk.PIVOT_BS, Atk.SPACE, Atk.RACE,
                                                              Atk.WAIT_MIDDLE, Atk.FOOTSIE)
        return dodge_ok and atk_ok

    def _safely_exit(self) -> None:
        if self.dodge_state == EXITING:
            self.dodge_state, self.dodge_type = INACTIVE, Def.NONE
            self._strafe_locked = None
        if self.attack_state == EXITING:
            self.attack_state, self.attack_type = INACTIVE, Atk.NONE

    # --------------------------------------------------------------- main tick
    def observe(self, p: Character, e: Character, now_ms: float) -> None:
        """Keep rolling memories current while the human is playing."""
        self.history.push_distance(G.legacy_distance(p, e), now_ms)

    def reset_routines(self) -> None:
        """Drop any half-finished input sequence (e.g. when control changes hands)."""
        self.dodge_state, self.dodge_type = INACTIVE, Def.NONE
        self.attack_state, self.attack_type = INACTIVE, Atk.NONE
        self._pivot_angle = self._pivot_behind_t = None

    def tick(self, p: Character, e: Character, cam_yaw: float, now_ms: float, pad) -> None:
        self.now, self.cam_yaw = now_ms, cam_yaw
        self.threat.push(now_ms, p.loc_x, p.loc_y, p.rotation_raw, e.loc_x, e.loc_y, e.rotation_raw)
        # their reach: learned per weapon and per attack from where their hits landed (the C
        # used a flat 6 m, so the bot rolled from swings that could never reach it)
        p.weapon_range = self.cfg.player_weapon_range
        e.weapon_range = self._enemy_reach(e)
        if e.animation_type != self._e_anim_last:
            if e.animation_type in T.ATTACK_ANIM or e.animation_type in LUNGE_ANIM:
                self._atk_start_d = G.distance(p, e)      # where this attack started
            fam = PROJ_FAMILY.get(e.animation_type)
            if fam and PROJ_FAMILY.get(self._e_anim_last) != fam and self._they_face_us(p, e, AIM_CONE_DEG):
                # a new cast aimed at us: when will it land?
                late = 0.5 if e.animation_type in PROJ_CAST_PHASE else 0.0
                self._proj = {"t0": now_ms - late * 1000, "fam": fam, "rolled": False}
            self._e_anim_last = e.animation_type
        self.history.push_distance(G.legacy_distance(p, e), now_ms)
        self._max_hp = max(self._max_hp, p.hp)
        if self._last_hp is not None and 0 < p.hp < self._last_hp and e.animation_type in T.ATTACK_ANIM:
            self.hits.record(e.r_weapon, e.animation_type, self._last_hp - p.hp)
            if self._proj is not None and self._proj["fam"] == "sorcery":
                # learn when sorceries really land (release = time to hit - travel)
                obs = (now_ms - self._proj["t0"]) / 1000 - G.distance(p, e) / PROJ_SPEED_MPS["sorcery"]
                if 0.4 < obs < 2.0:
                    self.hits.record_value("proj_release:sorcery", obs)
                self._proj = None
            if self._atk_start_d is not None and e.animation_type not in PROJECTILE_ANIM:
                self.hits.record_reach_start(e.r_weapon, e.animation_type, self._atk_start_d,
                                             e.anim_id)
            self._last_damage_t = now_ms
            aid, key = current_attack(e.raw)
            if key and aid not in T.DODGE_TIMINGS:
                self.hits.record_hit_time(aid, e.raw.get(key, 0.0))
            if G.distance(p, e) < 6:
                self.hits.record_reach(e.r_weapon, G.distance(p, e))
            self._say(f"took {self._last_hp - p.hp} from anim {e.animation_type} "
                      f"(remembered: {self.hits.get(e.r_weapon, e.animation_type):.0f})")
        self._last_hp = p.hp
        self.hits.maybe_save()
        self._e_track = [x for x in self._e_track if now_ms - x[0] <= 150] + [(now_ms, e.loc_x, e.loc_y)]
        vx, vy = self._enemy_velocity()
        if math.hypot(vx, vy) < 0.8 and e.sub == Sub.NEUTRAL:
            self._static_since = self._static_since or now_ms
        else:
            self._static_since = None
        self._sprint_req = False
        if self._e_prev is not None and 0 < now_ms - self._e_prev[0] < 200:
            dt = (now_ms - self._e_prev[0]) / 1000.0
            vx, vy = (e.loc_x - self._e_prev[1]) / dt, (e.loc_y - self._e_prev[2]) / dt
            if math.hypot(vx, vy) < 12.0:                 # not a teleport / target switch
                self._e_vel = (0.6 * self._e_vel[0] + 0.4 * vx, 0.6 * self._e_vel[1] + 0.4 * vy)
        self._e_prev = (now_ms, e.loc_x, e.loc_y)
        # how long they stay down, to kick on the frame they're up
        e_down = (e.in_backstab or e.sub == Sub.POISE_BROKEN) and e.hp > 1
        if e_down and self._edown is None:
            kind = "bs" if (p.animation_type == BACKSTAB_ANIM or self._was_backstabbing) else "stun"
            self._edown = (kind, now_ms)
        elif not e_down and self._edown is not None:
            kind, t0 = self._edown
            if 200 < now_ms - t0 < 6000:
                self.hits.record_value(f"wake_ms:{kind}", now_ms - t0, keep=0.7)
            self._edown = None
        self._shield_hist.append((now_ms, e.animation_type in SHIELD_ANIMS))
        while self._shield_hist and now_ms - self._shield_hist[0][0] > TURTLE_MS:
            self._shield_hist.popleft()
        # oki: our backstab just ended (they're on the floor), or our kick connected
        if p.animation_type == BACKSTAB_ANIM:
            self._was_backstabbing = True
        elif self._was_backstabbing:
            self._was_backstabbing = False
            self._oki_until = now_ms + OKI_AFTER_BS_MS
            self._say("backstab landed: oki (next attempt is a stance cancel)")
        # getting up from their backstab / a knockdown / a stagger: R1 there gets punished
        # (recordings to 2026-11-23: R1 within 1.5 s of 15 of 38 wake-ups, hit after 9 of them)
        down = self.is_down(p)
        if self._was_down and not down:
            self._woke_t = now_ms
        self._was_down = down
        if down and self.in_attack() and self.attack_type in R1_ATTACK_TYPES:
            self.attack_state, self.attack_type = INACTIVE, Atk.NONE   # no queued R1 for the way up
        # a clean kick (took HP, their shield wasn't up): race for their back
        kicking = p.animation_type in KICK_ANIMS
        if kicking and not self._kick_anim:
            self._kick_watch = (now_ms, e.hp, e.animation_type in SHIELD_ANIMS)
            # kick > unlock > block+sprint > circle > backstab: one motion (user, 2026-11-24)
            self._kick_race_until = now_ms + KICK_RACE_MS
            self._chain_until = now_ms + KICK_RACE_MS
            self._override_low_priority()
            self._say("kicked: unlock, block + sprint, round to their back")
        self._kick_anim = kicking
        if self._kick_watch is not None:
            t_k, hp_k, shield = self._kick_watch
            shield = shield or e.animation_type in SHIELD_ANIMS
            self._kick_watch = (t_k, hp_k, shield)
            if 0 < hp_k - e.hp < 300 and e.hp > 1:
                self._kick_watch = None
                if shield:
                    self._say("kick landed on their shield: no race")
                else:
                    self._kick_race_until = now_ms + KICK_RACE_MS
                    self._override_low_priority()
                    self._say(f"kick landed clean ({hp_k - e.hp} dmg): racing for their back")
            elif now_ms - t_k > KICK_HIT_WINDOW_MS:
                self._kick_watch = None
        if (now_ms - self._kick_t < 1000 and e.sub == Sub.POISE_BROKEN
                and self._oki_until < now_ms):
            self._kick_t = -1e9
            if self.cfg.backstab_only:
                # missed the backstab but the kick landed: straight back in for another one,
                # kick input again (user, 2026-11-23); the instinct pivots / backstabs from here
                self._say("kick landed (no backstab): going for the back again")
            else:
                self._oki_until = now_ms + OKI_AFTER_KICK_MS
                self._say("kick connected: oki")
        pad.reset()
        # forget the swing we dodged once it's over (or a different one started)
        if self._dodged_swing and (e.sub >= Sub.ACTIVE_HURTBOX_OVER
                                   or (e.anim_id, e.animation_type) != self._dodged_swing):
            self._dodged_swing = None

        inst = self.instinct(p, e)
        self.last_instinct = inst
        atk_choice = self.attack_mind(p, e)    # attack mind first; defense mind may override
        def_choice, bs_cancel = self.defense_mind(p, e)
        if self._back_turned:
            def_choice, bs_cancel = Def.NONE, False     # their back's to us: no backing off
        if bs_cancel and not self.waking_up():
            atk_choice = Atk.GHOST_HIT
        if self._jab_now:
            self._jab_now = False
            atk_choice = Atk.PUNISH
        if self._poke_ok(p, e, atk_choice):
            atk_choice = Atk.POKE
        low_for = lambda a: p.stamina < (ATTACK_STAMINA if a == Atk.BACKSTAB else POKE_STAMINA)
        if inst.priority == Priority.ATTACK and inst.attack in R1_ATTACK_TYPES and low_for(inst.attack):
            inst = Instinct()              # not enough stamina to swing and still roll
        if atk_choice in R1_ATTACK_TYPES and low_for(atk_choice):
            atk_choice = Atk.SPACE
        if self.waking_up():
            # never R1 on wake-up (user, 2026-11-23): drop every R1 choice before attack()
            # sees it (converting it there would restart SPACE every tick)
            if inst.priority == Priority.ATTACK and inst.attack in R1_ATTACK_TYPES:
                inst = Instinct()
            if atk_choice in R1_ATTACK_TYPES:
                atk_choice = Atk.SPACE
        elif self.now - self._disengage_t < 50 and atk_choice == Atk.NONE:
            atk_choice = Atk.SPACE
        if inst.priority == Priority.ATTACK and def_choice in (Def.COUNTER_STRAFE_LEFT,
                                                               Def.COUNTER_STRAFE_RIGHT):
            def_choice = Def.NONE   # we're the one in backstab position: don't strafe away

        if inst.priority == Priority.DODGE or self.in_dodge() or def_choice != Def.NONE:
            self.dodge(p, e, inst, def_choice, pad)
        # The C only entered attack() when a *mind* picked an attack, so instinct
        # decisions (take this backstab, two-hand) silently did nothing unless the
        # net happened to fire too. Instinct attacks count here.
        wants_attack = atk_choice != Atk.NONE or inst.priority == Priority.ATTACK
        if self.in_attack() or (wants_attack and def_choice == Def.NONE):
            self.attack(p, e, inst, atk_choice, pad)
        # sprint guard: 2026-11-23 ~3.5 rolls/min came from sprints cut short (B released early
        # = a roll). Once sprinting, keep B held SPRINT_HOLD_MS unless a roll is wanted anyway.
        if self._sprint_req:
            self._sprint_since = self._sprint_since or now_ms
        elif self._sprint_since is not None:
            if now_ms - self._sprint_since < SPRINT_HOLD_MS and not pad.buttons & R1:
                pad.buttons |= CIRCLE
            else:
                self._sprint_since = None
        it = self.items
        if (it is not None and it.n_items > 0 and it.cycles % it.n_items != 0 and not it.active
                and not (self.in_attack() and self.attack_type == Atk.KNIVES)):
            it.restore_soapstone(now_ms, pad)       # a cut-short knife routine: back to the soapstone
        if self._was_down:
            # down (their backstab, kickstun, stagger): hold block, we come up facing them
            # faster (user, 2026-11-23)
            pad.buttons |= L1
        else:
            self._wait_out = None
        centred = abs(pad.lx - MIDDLE) < 2000 and abs(pad.ly - MIDDLE) < 2000
        if centred and not self._stick_centred_prev:
            self._stick_centred_since = self.now
        self._stick_centred_prev = centred
        self._safely_exit()

    # -------------------------------------------------------------- instinct
    def instinct(self, p: Character, e: Character) -> Instinct:
        d = Instinct()
        dist = G.distance(p, e)
        atk_kind = T.ATTACK_ANIM.get(e.animation_type, 0)
        self._back_turned = self._back_to_us(p, e, dist) and not self._would_kill(p, e)
        if self._back_turned:
            self._back_chase_until = self.now + 300

        # Lowest priority first; later rules overwrite earlier ones.
        # Equipment toggles get a cooldown: if the game didn't register the press (or the
        # build can't do it), the C re-pressed every 40-500 ms, toggling it on and off.
        cooled = self.now - self._last_equip_change > EQUIP_COOLDOWN_MS
        # Two-handed by default (user, 2026-11-23), whenever out of their reach - except after a
        # toggle escape while racing for the back: the toggle got us to max speed faster (like
        # block does), so stay one-handed for that race.
        toggled_racing = (self.now - self._toggle_escape_t < TOGGLE_RACE_MS
                          and self.in_attack() and self.attack_type == Atk.RACE)
        if (cooled and self.cfg.prefer_two_hand and not p.two_handing and not toggled_racing
                and dist > e.weapon_range + 0.5 and e.sub not in (Sub.WINDUP, Sub.WINDUP_CLOSING,
                                                                 Sub.ACTIVE_DURING_HURTBOX)):
            d = Instinct(Priority.ATTACK, attack=Atk.TWO_HAND)
        if cooled and self.cfg.switch_off_fists and p.l_weapon == FISTS and dist > e.weapon_range * 1.75:
            d = Instinct(Priority.ATTACK, attack=Atk.SWITCH_WEAPON)

        if dist <= e.weapon_range and not p.in_backstab and not e.in_backstab and not self._back_turned:
            about_to_hit = (
                (atk_kind == 3 and e.sub <= Sub.ACTIVE_DURING_HURTBOX) or
                (atk_kind in (2, 4) and Sub.WINDUP_CLOSING <= e.sub <= Sub.ACTIVE_DURING_HURTBOX)
            )
            swing = (e.anim_id, e.animation_type)
            # Tank only when we can do something with it (pivot / punish range). Live
            # 2026-11-23 it "tanked" a 287-damage hit from 4.1 m four times: stood there and
            # took it, with nothing to follow up with.
            tank = about_to_hit and dist <= self.cfg.tank_range and self._can_tank(p, e)
            walls = self._walls(p)
            back_to_wall = walls is not None and not walls.behind_reachable(e)
            if tank and back_to_wall and (dist > p.weapon_range or self.waking_up() or self.cfg.backstab_only):
                tank = 0.0     # can't pivot behind them and can't (or mustn't) hit them: dodge
            if about_to_hit and swing != self._said_swing:
                self._said_swing = swing
                if tank:
                    self._say(f"incoming hit: anim type {e.animation_type}, {tank:.1f} poise dmg "
                              f"vs our {p.poise:.0f}: tanking it and going for the backstab")
                else:
                    self._say(f"incoming hit: anim type {e.animation_type}, sub {e.sub.name}, "
                              f"{e.dodge_time_remaining:.2f}s left")
            if tank:
                if self.attack_type != Atk.PIVOT_BS:   # don't cancel the pivot we're tanking for
                    self._override_low_priority()
                if back_to_wall:
                    # their back is against the wall: no getting behind them, trade instead
                    d = Instinct(Priority.ATTACK, attack=Atk.PUNISH)
                else:
                    d = Instinct(Priority.ATTACK, attack=Atk.PIVOT_BS)
                about_to_hit = False
            if about_to_hit and swing == self._dodged_swing and not self.in_dodge():
                # already dodged this swing: don't cancel what we're doing every tick for a
                # dodge that won't start (it restarted the kick-chain race every tick)
                about_to_hit = False
            if about_to_hit:
                self._override_low_priority()
                d = Instinct(Priority.DODGE)
                if p.sub == Sub.POISE_BROKEN and 0.2 < e.dodge_time_remaining < 0.3:
                    d.defense = Def.TOGGLE_ESCAPE
                    return d
                if p.sub != Sub.POISE_BROKEN:
                    last = self._last_self()
                    if (dist <= 3 and REVERSE_ROLL_TOTAL_S < e.dodge_time_remaining and
                            (last != ("def", Def.REVERSE_ROLL_BS) or
                             REVERSE_ROLL_TOTAL_S + 0.3 > e.dodge_time_remaining)):
                        d.defense = Def.REVERSE_ROLL_BS
                        walls = self._walls(p)
                        if walls is not None and walls.free_run(
                                p.loc_x, p.loc_y, G.heading_to(p, e) + math.pi) < 1.5:
                            d.defense = Def.STANDARD_ROLL      # wall behind: roll sideways
                    elif (0 < e.dodge_time_remaining < 0.15 and
                          (p.two_handing or is_weapon_shield(p.l_weapon)) and p.sub == Sub.NEUTRAL):
                        d.defense = Def.PERFECT_BLOCK
                    else:
                        d.defense = Def.STANDARD_ROLL
                if self._last_self() == ("def", Def.TOGGLE_ESCAPE):
                    d.defense = Def.STANDARD_ROLL   # they're comboing: get out
            elif atk_kind == 1 or (atk_kind in (2, 4) and e.sub == Sub.WINDUP):
                d = Instinct(Priority.DELAY)

        # Fire tempest: get out, rolling, straight away from them toward open floor (user).
        # Fire surge inside its range (aimed at us): same, until out of its range.
        flee = None
        if e.animation_type in FIRE_STORM_ANIMS and dist < FLEE_SAFE_M:
            flee = (FIRE_STORM_ANIMS, FLEE_SAFE_M)
        elif (e.animation_type in SURGE_ANIMS and dist < SURGE_REACH_M
              and self._they_face_us(p, e, 60.0)):
            flee = (SURGE_ANIMS, SURGE_SAFE_M)
        if flee and not self.is_down(p) and not p.in_backstab:
            self._flee_set, self._flee_safe = flee
            if not (self.in_dodge() and self.dodge_type == Def.FLEE_ROLLS):
                self._override_low_priority()
                if self.in_dodge():
                    self.dodge_state, self.dodge_type = INACTIVE, Def.NONE
                self._dodged_swing = None
            return Instinct(Priority.DODGE, defense=Def.FLEE_ROLLS)

        # Projectile on its way: roll so the i-frames cover the moment it lands.
        left = self._proj_hit_in(dist)
        if left is not None and left < -PROJ_WINDOW_S:
            self._proj = None                      # it missed (or hit someone else)
        elif (left is not None and left <= PROJ_ROLL_LEAD_S and not self._proj["rolled"]
              and d.priority < Priority.DODGE and dist < PROJECTILE_MAX_M
              and not (self.is_down(p) or p.in_backstab or e.in_backstab)):
            if self.in_dodge() and self.dodge_type == Def.SIDESTEP:
                self.dodge_state, self.dodge_type = INACTIVE, Def.NONE   # stop running: roll
            self._proj["rolled"] = True
            self._dodged_swing = None              # the sidestep for this cast doesn't count
            self._override_low_priority()
            self._say(f"{self._proj['fam']} due in {max(left, 0):.2f}s from {dist:.1f} m: rolling")
            d = Instinct(Priority.DODGE, defense=Def.STANDARD_ROLL)

        # Spells, arrows and flame streams: run across their line of fire (projectiles miss a
        # target moving sideways; a flame stream only covers a narrow cone in front of them).
        if d.priority < Priority.DODGE and self._ranged_dodge(p, e, dist):
            if not (self.in_dodge() and self.dodge_type == Def.SIDESTEP):
                self._override_low_priority()
            d = Instinct(Priority.DODGE, defense=Def.SIDESTEP)

        # Live 2026-11-23 the backstab landed, then the bot "attempted" 21 more during its own
        # backstab animation (it reads as neutral and perfectly lined up) and buffered an R1.
        bs = G.backstab_state(p, e, dist) if p.animation_type != BACKSTAB_ANIM else 0
        if bs:
            # In the tight window: plain backstab (R1 facing them). The port used GHOST_HIT
            # here, which turns away right after R1: live, that threw away a backstab.
            # Behind but not in position: pivot into it (was MOVE_UP, straight at them).
            committed = e.sub in (Sub.LOCK_IN, Sub.ACTIVE_HURTBOX_OVER, Sub.POISE_BROKEN)
            if self._bs_ready(p, e, dist):
                want = Atk.BACKSTAB        # input is a kick ("kick confirm" option select)
            elif e.in_backstab:
                # they're lying down: the wake-up routine holds us still at their back, stick
                # centred, so the kick goes out the frame they're up (line-up walked into them)
                want = None
            elif self._at_their_back(p, e, dist) and self._backstabbable(e):
                # 2026-11-23: 180 of 223 backstab chances went unused, nearly all with us at
                # their back facing the wrong way. Turn into them (unlocked: the camera can't
                # spoil a lock-on we don't need), then the kick fires.
                want = Atk.LINE_UP
            elif committed and dist <= 2.0 and not (
                    self._walls(p) is not None and not self._walls(p).behind_reachable(e)):
                # only chase their back while they're stuck in an animation; in neutral a
                # human just turns or backs into a wall and punishes the chase (user, live)
                want = Atk.PIVOT_BS
            else:
                want = None
            if want in R1_ATTACK_TYPES and self.waking_up():
                want = None        # never R1 on wake-up (and don't cancel spacing for it)
            if want == Atk.BACKSTAB and p.stamina < ATTACK_STAMINA:
                want = None
            if want is not None:
                if not (self.in_attack() and self.attack_type == want):
                    self._override_low_priority()
                d = Instinct(Priority.ATTACK, attack=want)
        # their back is turned to us: go for it, whatever they're doing (user, 2026-11-24)
        if self._back_turned and d.priority < Priority.ATTACK and not self.waking_up():
            want = Atk.LINE_UP if self._at_their_back(p, e, dist) else Atk.RACE
            if self._bs_ready(p, e, dist) and p.stamina >= ATTACK_STAMINA:
                want = Atk.BACKSTAB
            if not (self.in_attack() and self.attack_type in (want, Atk.BACKSTAB, Atk.LINE_UP, Atk.PIVOT_BS)):
                self._override_low_priority()
            d = Instinct(Priority.ATTACK, attack=want)
        return d

    def _winning_race(self, p, e) -> bool:
        """Closer to their back than they are to ours: commit, don't turn away.
        (2026-11-23: race <-> pivot-turn was the bot's most common back-and-forth up close -
        "gets in their face and wiggles when he should commit", user.)"""
        def off_back(a, b):
            fx, fy = G.facing_vector(b)
            rx, ry = a.loc_x - b.loc_x, a.loc_y - b.loc_y
            return math.degrees(math.atan2(abs(rx * fy - ry * fx), -(rx * fx + ry * fy)))
        return off_back(p, e) <= off_back(e, p)

    def _enemy_velocity(self):
        tr = self._e_track
        if len(tr) < 2 or tr[-1][0] - tr[0][0] < 60:
            return 0.0, 0.0
        dt = (tr[-1][0] - tr[0][0]) / 1000.0
        return (tr[-1][1] - tr[0][1]) / dt, (tr[-1][2] - tr[0][2]) / dt

    def _enemy_closing_speed(self, p, e) -> float:
        """Their speed toward us (m/s, + = coming at us)."""
        vx, vy = self._enemy_velocity()
        ux, uy = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        n = math.hypot(ux, uy) or 1e-6
        return (vx * ux + vy * uy) / n

    def _enemy_static_ms(self) -> float:
        return self.now - self._static_since if self._static_since is not None else 0.0

    def _sprint(self, pad) -> None:
        """Hold B to sprint; the tick keeps it held SPRINT_HOLD_MS so it can't come out as a roll."""
        self._sprint_req = True
        pad.buttons |= CIRCLE

    def _enemy_lateral_speed(self, p, e) -> float:
        """Their speed across our line of sight (m/s), from the last ~0.1 s."""
        tr = self._e_track
        if len(tr) < 2 or tr[-1][0] - tr[0][0] < 60:
            return 0.0
        dt = (tr[-1][0] - tr[0][0]) / 1000.0
        vx, vy = (tr[-1][1] - tr[0][1]) / dt, (tr[-1][2] - tr[0][2]) / dt
        ux, uy = e.loc_x - p.loc_x, e.loc_y - p.loc_y
        n = math.hypot(ux, uy) or 1e-6
        return abs(vx * uy - vy * ux) / n

    def _into_their_back(self, p, e):
        """Stick for the last step of a backstab setup, unlocked: wide of their back line ->
        go to the spot on it (~0.75 m behind them); on the line -> turn into them."""
        fx, fy = G.facing_vector(e)
        rx, ry = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        behind = -(rx * fx + ry * fy)
        pos = math.degrees(math.atan2(abs(rx * fy - ry * fx), max(behind, 1e-6)))
        bx, by = e.loc_x - fx * 0.75, e.loc_y - fy * 0.75
        if pos > 40.0 and math.hypot(bx - p.loc_x, by - p.loc_y) > 0.2:
            return self.steer.stick_for_heading(math.atan2(bx - p.loc_x, by - p.loc_y), self.cam_yaw,
                                                magnitude=0.6)
        mag = 0.3 if G.distance(p, e) < 0.8 else 0.5
        return self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=mag)

    @staticmethod
    def _at_their_back(p, e, dist: float) -> bool:
        if dist > LINE_UP_M:
            return False
        fx, fy = G.facing_vector(e)
        rx, ry = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        behind = -(rx * fx + ry * fy)
        return behind > 0 and math.degrees(math.atan2(abs(rx * fy - ry * fx), behind)) <= LINE_UP_DEG

    def _bs_ready(self, p, e, dist: float) -> bool:
        """Press the backstab now? (Shared by the instinct and the routines that hand over to it,
        or they restart every tick waiting for each other.)"""
        geo = (G.can_backstab_now(p, e, dist) and self._facing_off_deg(p, e) <= KICK_FACING_MAX
               and (p.sub in (Sub.NEUTRAL, Sub.RECOVER) or not self.cfg.backstab_only))
        return geo and ((self._backstabbable(e) and not e.in_backstab) or self._wake_due())

    def _wake_in(self) -> float | None:
        """ms until they should be up again (learned per kind), or None if we don't know."""
        if self._edown is None:
            return None
        kind, t0 = self._edown
        learned = self.hits.table.get(f"wake_ms:{kind}")
        return None if learned is None else t0 + learned - self.now

    def _wake_due(self) -> bool:
        """Lying down after our backstab, about to get up: press now so the kick lands on the
        first frame they're backstabbable."""
        left = self._wake_in()
        return left is not None and -300 < left <= WAKE_LEAD_MS

    @staticmethod
    def _facing_off_deg(p, e) -> float:
        """How far our facing is from the direction to them."""
        h = G.heading_to(p, e)
        return math.degrees(abs((G.facing_heading(p) - h + math.pi) % (2 * math.pi) - math.pi))

    def _we_backstabbed(self) -> bool:
        return self.now - self._bs_attempt_t < 3500

    @staticmethod
    def _backstabbable(e) -> bool:
        """Live 2026-11-22: R1 from dead behind at 0.99 and 1.16 m during their windup gave
        plain R1s. Wait for their recovery (or neutral) instead of pressing mid-swing."""
        return e.sub not in (Sub.WINDUP, Sub.WINDUP_CLOSING, Sub.WINDUP_GHOST_HIT,
                             Sub.ACTIVE_DURING_HURTBOX)

    def _can_tank(self, p, e) -> float:
        """Poise damage of the incoming attack if we can take it without staggering, else 0."""
        if not self.cfg.poise_tank or p.sub == Sub.POISE_BROKEN or p.in_backstab:
            return 0.0
        if "poise" not in p.raw or self._max_hp <= 0:
            return 0.0
        dmg = poise_damage_known(e.r_weapon, e.animation_type)
        if dmg is None:                     # attack not in the table: don't gamble
            return 0.0
        hurt = self.hits.get(e.r_weapon, e.animation_type)
        if self.cfg.backstab_only:
            # Havel/club (user, 2026-11-23): "rolls when he has enough poise to trade with a
            # backstab - this should not be". A ~490 Zweihander hit for a 1150-1450 backstab is a
            # good trade; just don't trade into a hit we might not survive.
            need = (hurt if hurt is not None else TANK_UNKNOWN_FRAC * self._max_hp)
            if p.hp - need < TANK_KEEP_FRAC * self._max_hp:
                return 0.0
        else:
            if p.hp < self.cfg.tank_min_hp_frac * self._max_hp:
                return 0.0
            if hurt is not None and hurt > self.cfg.tank_max_dmg_frac * self._max_hp:
                return 0.0                  # it doesn't stagger us, but it hurts too much
        if p.poise > dmg + self.cfg.tank_margin:
            return max(dmg, 0.1)
        return 0.0

    # --------------------------------------------------------------- minds
    def defense_features(self, p, e) -> np.ndarray:
        h = self.history.distances
        return np.array([G.legacy_distance(p, e), h[0], h[1], h[2], h[3],
                         G.legacy_angle_delta_from_front(p, e), e.velocity,
                         G.legacy_rotation_difference(p, e)])

    def backstab_threat(self, p, e) -> float:
        """-1..1: how likely a backstab on us is in the next second."""
        if self.threat_net is not None:
            x = self.threat.features(p.animation_type != 0, e.animation_type != 0)
            return self.threat_net(x) if x is not None else -1.0
        return self.defense_net(self.defense_features(p, e))

    def defense_mind(self, p, e) -> tuple[Def, bool]:
        choice = Def.NONE
        dist = G.distance(p, e)
        if self.cfg.use_defense_net and (self.threat_net is not None or self.defense_net is not None):
            out = self.backstab_threat(p, e)
            # not at point blank: every recording to 2026-11-23, host neutral and facing us
            # within 1.3 m -> counter-strafe got backstabbed 24% of the time (19 times),
            # backing off 8%.
            if (out > (THREAT_ON if self.threat_net is not None else 0.5) and STRAFE_MIN_M <= dist < 5 and e.sub == Sub.NEUTRAL
                    and self.now - self._disengage_t > NO_STRAFE_AFTER_ROLL_MS):
                walls = self._walls(p)
                if walls is not None and walls.back_covered(p):
                    # wall right behind us: nowhere for them to stand for a backstab
                    if self.now - self._said_wall > 5000:
                        self._said_wall = self.now
                        self._say("back to the wall: no need to counter-strafe")
                else:
                    choice = Def.COUNTER_STRAFE_LEFT
                    self._last_bs_pred = float(out)
        # Live 2026-11-23: the host hugged the bot (0.3-0.6 m) against the wall; it held its
        # ground strafing, turned to follow them along the wall, and showed them its back.
        # A human that close in neutral is setting up a backstab: roll out to open floor.
        pivoting = self.in_attack() and self.attack_type == Atk.PIVOT_BS
        facing_us = (self._they_face_us(p, e, 35.0) and self.now - self.t_atk > 600) if pivoting \
            else self._they_face_us(p, e)          # mid-pivot: only if they turned back to us
        if (False and dist < CLOSE_M and e.sub == Sub.NEUTRAL and facing_us
                and p.stamina >= ROLL_STAMINA and p.sub >= Sub.LOCK_IN and self._only_low_priority()
                and self.now - self._disengage_t > DISENGAGE_COOLDOWN_MS):
            # Recordings to 2026-11-23: at point blank a roll got backstabbed 33%, a kick 1 of
            # 2, a counter-strafe 24%, backing off (SPACE, locked on) 8%. So back off; if we
            # can't (they're right on us, or a wall's behind us), jab them with an R1.
            walls = self._walls(p)
            stuck = walls is not None and walls.free_run(
                p.loc_x, p.loc_y, G.heading_to(p, e) + math.pi) < 1.0
            self._disengage_t = self.now
            if (dist < JAB_M or stuck) and not self.waking_up():
                self._override_low_priority()
                self._jab_now = True
                self._say(f"pinned ({dist:.1f} m{', wall behind' if stuck else ''}): jabbing")
            elif not (self.in_attack() and self.attack_type == Atk.SPACE):
                self._override_low_priority()      # drop strafing / pivoting: back off
                self._say(f"too close ({dist:.1f} m) in neutral: backing off")
        if p.in_backstab and not self._we_backstabbed():
            # waking up from THEIR backstab: avoid the chain. (in_backstab is also set while
            # we do the backstabbing; live the bot mashed escapes through its own backstab
            # and rolled away from a knocked-down enemy.)
            # User 2026-11-23: a quick wake-up every time lets them prepare (parry / backstab
            # the get-up). With them close enough to try, sometimes stay down instead.
            if self._wait_out is None:
                self._wait_out = dist <= WAIT_OUT_M and self.rng.random() < self._st("wait_out_chance")
                if self._wait_out:
                    self._say(f"down with them {dist:.1f} m away: waiting it out (mix-up)")
            if (not self._wait_out
                    and self.now - self._victim_escape_t > VICTIM_ESCAPE_EVERY_MS):
                self._victim_escape_t = self.now   # (was every 0.13-0.4 s: 20 presses per backstab)
                choice = self.rng.choice([Def.OMNISTEP_BACKWARDS, Def.REVERSE_ROLL_BS])
        # Enemy close behind us: hit them to cancel their backstab. The C had no
        # distance check, so it swung at the air whenever they were anywhere behind.
        # Recordings to 2026-11-23, host neutral and on us: ghost hit (back turned in the
        # windup) -> backstabbed 4 of 7, walking backward 6 of 15 (they follow, and with lag
        # the backstab lands from 1.2-1.9 m on our screen), reverse roll 0 of 12.
        pivoting = self.in_attack() and self.attack_type == Atk.PIVOT_BS   # running round them
        behind_us = (dist <= BS_CANCEL_RANGE and bool(G.backstab_state(e, p, dist))
                     and e.sub == Sub.NEUTRAL and self._they_face_us(p, e) and not pivoting
                     and not p.in_backstab and not e.in_backstab)
        followed = (dist < FOLLOW_M and e.sub == Sub.NEUTRAL and e.animation_type != BACKSTAB_ANIM
                    and self._they_face_us(p, e)            # them on us, not us behind them
                    and not pivoting and not p.in_backstab and not self.in_dodge())
        if not followed:
            self._followed_since = None
        elif self._followed_since is None:
            self._followed_since = self.now
        # retired 2026-11-23 for racing (user): being close is the plan now
        stuck_on_us = False and followed and self.now - self._followed_since >= FOLLOW_MS
        if ((behind_us or stuck_on_us) and p.stamina >= ROLL_STAMINA and p.sub >= Sub.LOCK_IN
                and self._only_low_priority()
                and self.now - self._escape_roll_t > ESCAPE_ROLL_COOLDOWN_MS):
            self._override_low_priority()
            self._escape_roll_t = self.now
            self._followed_since = None
            # user 2026-11-23: roll less - turn to face them instead of rolling out
            choice = Def.TURN_TO_FACE if behind_us else Def.REVERSE_ROLL_BS
            self._say("they're behind us: turn to face them" if behind_us
                      else f"they keep following ({dist:.1f} m): reverse roll out")
        elif (choice == Def.NONE and dist < PIVOT_DETECT_M and e.sub == Sub.NEUTRAL and not pivoting
              and not self._winning_race(p, e)
              and not p.in_backstab and p.sub >= Sub.LOCK_IN and self._only_low_priority()
              and self.now - self._turn_t > TURN_COOLDOWN_MS
              and self._enemy_lateral_speed(p, e) > PIVOT_LATERAL_MPS):
            self._override_low_priority()
            self._turn_t = self.now
            choice = Def.TURN_TO_FACE
            self._say(f"incoming pivot ({dist:.1f} m, {self._enemy_lateral_speed(p, e):.1f} m/s across): "
                      "unlock and turn")
        return choice, False

    def attack_features(self, p, e) -> np.ndarray:
        h = self.history
        return np.array(
            list(h.distances) +
            [h.enemy_stamina_estimate(e), e.poise, poise_damage_for_attack(p.r_weapon, 46),
             p.poise, poise_damage_for_attack(e.r_weapon, 46)] +
            list(h.ai_hp) + [p.stamina] + list(h.last_anim_types_enemy) + [p.bleed_status],
            dtype=np.float64)

    def attack_mind(self, p, e) -> Atk:
        choice = Atk.NONE
        dist = G.distance(p, e)
        hold = self._spacing_reach(e) + self.cfg.spacing_margin
        walls = self._walls(p)
        if walls is not None and walls.wall_distance(e.loc_x, e.loc_y) < -OUTSIDE_M:
            # Live 2026-11-23 the host waited outside the walls (16.8 m away) and the bot pushed
            # into the wall toward them for 23 s. Wait in the middle instead.
            return Atk.WAIT_MIDDLE
        if e.animation_type in SURGE_ANIMS and dist >= SURGE_REACH_M - 0.5:
            choice = Atk.KEEP_OUT               # fire surge: stay out of its range (user)
        elif self._they_are_down(p, e) and dist <= 8.0:
            choice = Atk.OKI                    # they're down: be at their back for the wake-up
        elif self._corner_camping(p, e, dist):
            choice = Atk.KNIVES                 # corner camper: poison knives till they come out
        elif self.now < self._kick_race_until and dist <= hold + RACE_EXTRA_M + 2.0:
            choice = Atk.RACE                   # after a kick: get round to their back now
        elif self._camping(e, dist, hold) and not self.waking_up():
            choice = Atk.CHARGE                 # they sit / cast / buff at range: go get them
        elif dist > hold + 1.5:
            choice = Atk.MOVE_UP                # far: close in to spacing range
        elif ((self._turtling(p, e, dist) or self._no_cast(e, dist))
              and self.now >= self._turtle_next and not self.waking_up() and p.stamina >= ATTACK_STAMINA):
            choice = Atk.TURTLE                 # turtling / no-cast loop: kicks / jumping attacks / roll ghosts
        elif e.sub == Sub.NEUTRAL:
            # User 2026-11-23 (6-3 with this character vs the host the bot went 3-26 against):
            # don't back off in neutral, race round them for the back.
            # User 2026-11-23: pivots shouldn't be the only plan. From neutral, threaten with
            # forward movement and bait theirs (footsies); punish their forward movement with a
            # run backstab; race round for the back only when they sit still.
            closing = self._enemy_closing_speed(p, e)
            if (RUN_BS_MIN_M <= dist <= RUN_BS_MAX_M and closing > self._st("run_bs_closing")
                    and p.stamina >= ROLL_STAMINA + ATTACK_STAMINA and not self.waking_up()):
                choice = Atk.RUN_BS
            elif (dist <= hold + RACE_EXTRA_M and self._enemy_static_ms() > self._st("race_static_ms")
                  and not self._shield_mostly_up()     # a still shield gets the turtle mix-up
                  and not self._no_cast(e, dist)):     # so does the no-cast loop
                choice = Atk.RACE
            else:
                choice = Atk.FOOTSIE
        bleed_ok = "bleed_status" not in p.raw or p.bleed_status > 40
        whiffed = Sub.LOCK_IN <= e.sub < Sub.NEUTRAL      # their attack is over, they're recovering
        if e.animation_type in T.DODGE_ANIM:
            whiffed = False         # a roll isn't a whiff: punishing rolls hit 30% (15/50) and
                                    # they roll behind us; attacks' recoveries 51% (41/80)
        if self.now - self._last_damage_t < 700:
            whiffed = False         # it hit us: not a whiff, and they're probably chaining
        if e.sub == Sub.ACTIVE_HURTBOX_OVER and e.dodge_time_remaining > -WHIFF_MARGIN_S:
            whiffed = False         # only just "over" by the table: lunges hit late (live 2026-11-23)
        # Attack only into a whiff / stuck animation. Live 2026-11-22 the attack net fired a
        # GHOST_HIT + DEAD_ANGLE at a neutral host 2.4 m away: ~1.2 s committed, and the ghost
        # hit turns our back to them during the windup -> they walked round and backstabbed.
        if choice == Atk.NONE and not whiffed:
            # they're mid-attack but it isn't about to hit us (instinct handles that): keep
            # moving. Live 2026-11-23 the bot stood still out of every roll ("like a dingus")
            # until their whole string was over.
            choice = Atk.SPACE
        if dist <= p.weapon_range + PUNISH_RUN_M and p.stamina >= POKE_STAMINA and bleed_ok and whiffed:
            if not self.cfg.backstab_only:
                choice = Atk.PUNISH
            elif self.now - self._last_roll_t < POST_ROLL_MS and dist <= p.weapon_range:
                choice = Atk.GHOST_HIT     # "if you're going to roll R1, do a ghost strike" (user)
        return choice

    # ---------------------------------------------------------------- dodge
    def dodge(self, p, e, inst: Instinct, net_choice: Def, pad) -> None:
        # One dodge per swing. The roll routine ends as soon as the roll locks in, and the
        # same incoming hit then started another one every tick (live 2026-11-22: ~30
        # starts per swing), mashing circle and queueing extra rolls.
        swing = (e.anim_id, e.animation_type)
        repeat = inst.priority == Priority.DODGE and swing == self._dodged_swing
        choice = inst.defense if inst.priority == Priority.DODGE else net_choice
        if choice in ROLL_DEFS and p.stamina < ROLL_STAMINA:
            # can't afford the roll: block it (escape rolls just don't happen)
            choice = Def.HOLD_BLOCK if inst.priority == Priority.DODGE else Def.NONE
        if not self.in_any() and p.sub >= Sub.LOCK_IN and not repeat and choice != Def.NONE:
            self.dodge_type = choice
            self.dodge_state = ACTIVE
            self.t_def = self.now
            self._roll_heading = (self._pending_heading if choice == Def.STANDARD_ROLL
                                  and inst.priority != Priority.DODGE else None)
            if choice in ROLL_DEFS:
                self._last_roll_t = self.now
            if choice == Def.TOGGLE_ESCAPE:
                self._toggle_escape_t = self.now
            if inst.priority == Priority.DODGE:
                self._dodged_swing = swing
            why = ""
            if self.dodge_type in (Def.COUNTER_STRAFE_LEFT, Def.COUNTER_STRAFE_RIGHT):
                why = f" (backstab predicted {getattr(self, '_last_bs_pred', 0.0):.2f})"
            self._say(f"start dodge {self.dodge_type.name}{why}")
        self._pending_heading = None
        if not self.in_dodge():
            return
        t = self.now - self.t_def
        fn = {
            Def.STANDARD_ROLL: self._roll, Def.BACKSTEP: self._backstep,
            Def.OMNISTEP_BACKWARDS: self._omnistep, Def.COUNTER_STRAFE_LEFT: self._strafe_left,
            Def.COUNTER_STRAFE_RIGHT: self._strafe_right, Def.L1_ATTACK: self._l1,
            Def.REVERSE_ROLL_BS: self._reverse_roll, Def.TOGGLE_ESCAPE: self._toggle_escape,
            Def.PERFECT_BLOCK: self._perfect_block, Def.PARRY: self._parry,
            Def.HOLD_BLOCK: self._hold_block, Def.TURN_TO_FACE: self._turn_to_face,
            Def.SIDESTEP: self._sidestep, Def.FLEE_ROLLS: self._flee_rolls,
        }.get(self.dodge_type)
        if fn is None:   # e.g. staggered: detected the hit but nothing to do
            self.dodge_state = INACTIVE
            return
        fn(p, e, t, pad)

    def _roll(self, p, e, t, pad):
        # Done once THIS roll has started. Live 2026-11-23 a second roll (their follow-up
        # came while we were still in the first) quit on the spot because we were "locked
        # in" and never pressed circle: the game buffers it near the end, so press anyway.
        if t == 0.0:
            self._roll_from = p.anim_id if p.sub == Sub.LOCK_IN else None
        new_roll = p.sub == Sub.LOCK_IN and p.anim_id != self._roll_from
        if (t >= 60 and new_roll) or p.sub == Sub.POISE_BROKEN or t > 900:
            return self._finish("def", Def.STANDARD_ROLL)
        # One press. The C hammered circle 4 times (the old driver could drop inputs); on a reliable
        # pad each extra press queued another roll.
        if t < 60:
            pad.buttons |= CIRCLE
        if 0 <= t < 300 and self._roll_heading is not None:
            self._set_stick(pad, self.steer.stick_for_heading(self._roll_heading, self.cam_yaw))
            return
        if 0 <= t < 300:
            offset = 90.0
            last = self._last_self()
            if G.backstab_state(p, e) == 1:
                offset = 0.0                       # roll toward their back
            elif last in (("def", Def.STANDARD_ROLL), ("def", Def.TOGGLE_ESCAPE)):
                offset = 120.0                     # second roll: get away from r1 spam
            elif self.cfg.roll_past and G.distance(p, e) <= 3.0:
                # roll past their side to a point beside and behind them: the i-frames carry
                # us through the swing and we come out at their back for the backstab
                fx, fy = G.facing_vector(e)
                rx, ry = fy, -fx
                side = 1.0 if (p.loc_x - e.loc_x) * rx + (p.loc_y - e.loc_y) * ry >= 0 else -1.0
                walls = self._walls(p)
                for sd in (side, -side):
                    tx, ty = e.loc_x + rx * sd * 0.9 - fx * 1.0, e.loc_y + ry * sd * 0.9 - fy * 1.0
                    if walls is None or walls.contains(tx, ty):
                        self._set_stick(pad, self.steer.stick_for_heading(
                            math.atan2(tx - p.loc_x, ty - p.loc_y), self.cam_yaw))
                        return
                # their back is to the wall on both sides: plain roll below
            walls = self._walls(p)
            if walls is not None and offset:
                # roll whichever way has more room (the C always rolled the same side)
                h = G.heading_to(p, e)
                room = lambda off: walls.free_run(p.loc_x, p.loc_y, h - math.radians(off))
                if room(offset) < 2.0 and room(-offset) > room(offset):
                    offset = -offset
            lx, ly = self._toward(p, e, offset)
            if self.cfg.cardinal_rolls:
                if abs(lx - MIDDLE) > abs(ly - MIDDLE):
                    pad.lx = lx
                else:
                    pad.ly = ly
            else:
                pad.lx, pad.ly = lx, ly

    def _backstep(self, p, e, t, pad):
        if t < 40:
            pad.buttons |= CIRCLE
        else:
            self._finish("def", Def.BACKSTEP)

    def _omnistep(self, p, e, t, pad):
        if t < 40:
            pad.buttons |= CIRCLE
        elif 80 < t < 120:
            self._set_stick(pad, self._toward(p, e))
        elif t >= 120:
            self._finish("def", Def.OMNISTEP_BACKWARDS)

    def _strafe(self, p, e, t, pad, left: bool):
        sub_id = Def.COUNTER_STRAFE_LEFT if left else Def.COUNTER_STRAFE_RIGHT
        dist = G.distance(p, e)
        # The C pressed R3 unconditionally, which UNLOCKS when already locked on, then
        # bailed out after 60 ms because it wasn't locked: the bot just flickered lock-on.
        if t < 30:
            if t == 0.0 or self._strafe_locked is None:
                self._strafe_locked = not p.locked_on   # True: we lock now, unlock after
                self._cs_side, self._cs_prev = 0, None
            if self._strafe_locked:
                pad.buttons |= R3
        elif t < 60:
            pass
        elif t < 800 or G.backstab_state(p, e, dist) == 1 or dist < 1.3:
            # Live 2026-11-23: 15 of the 24 backstabs on the bot landed during this strafe,
            # which (ported from the C) always went LEFT, so players just circled the other
            # way. Now: strafe toward the side they're circling to (where they'll be in
            # 0.3 s), which keeps them in front, and back off a little while close, since
            # circling at range is much harder. Re-decided every tick, with hysteresis.
            self._set_stick(pad, self._toward(p, e, self._strafe_offset(p, e, dist, left)))
        elif t < 830:
            if self._strafe_locked:
                pad.buttons |= R3                  # unlock only if we locked
        elif t < 860:
            pass
        else:
            self._strafe_locked = None
            return self._finish("def", sub_id)

    def _strafe_offset(self, p, e, dist: float, left: bool) -> float:
        bearing = G.heading_to(p, e)
        now = self.now
        rate = 0.0
        if self._cs_prev is not None and 0 < now - self._cs_prev[1] < 250:
            d = (bearing - self._cs_prev[0] + math.pi) % (2 * math.pi) - math.pi
            rate = d / ((now - self._cs_prev[1]) / 1000.0)        # rad/s, around us
        self._cs_prev = (bearing, now)
        rel = (bearing - G.facing_heading(p) + math.pi) % (2 * math.pi) - math.pi
        future = rel + rate * 0.3
        if self._cs_side == 0:
            self._cs_side = (1 if future > 0 else -1) if abs(future) > 0.05 else (1 if left else -1)
        elif self._cs_side * future < -math.radians(10):             # clearly the other side now
            self._cs_side = -self._cs_side
        mag = 110.0 if dist < 2.5 else 90.0                          # >90: sideways and away
        walls = self._walls(p)
        if walls is not None and walls.free_run(
                p.loc_x, p.loc_y, bearing + math.radians(self._cs_side * mag)) < 0.8:
            mag = 75.0                                               # wall that way: slide along it
        # _toward moves along heading_to - offset; moving that way swings their bearing
        # the other way, so to pull a target on our + side back in front, offset is -.
        return -self._cs_side * mag

    def _strafe_left(self, p, e, t, pad):
        self._strafe(p, e, t, pad, True)

    def _strafe_right(self, p, e, t, pad):
        self._strafe(p, e, t, pad, False)

    def _l1(self, p, e, t, pad):
        if t < 30:
            self._set_stick(pad, self._toward(p, e))
            pad.buttons |= L1
        else:
            self._finish("def", Def.L1_ATTACK)

    def _reverse_roll(self, p, e, t, pad):
        start = R3_TRIGGER + CAMERA_ROTATE_AFTER_LOCKON
        if t < R3_TRIGGER and not p.locked_on:
            pad.buttons |= R3
        if start < t < start + GAME_REGISTER_ACTION:
            pad.ly = YBOTTOM                       # back, relative to the lock-on camera
            pad.buttons |= R3 | CIRCLE
        if t > start + GAME_REGISTER_ACTION:
            self._finish("def", Def.REVERSE_ROLL_BS)

    def _toggle_escape(self, p, e, t, pad):
        if t < 30:
            pad.hat = DLEFT
        if t > 60:
            self._finish("def", Def.TOGGLE_ESCAPE)

    def _perfect_block(self, p, e, t, pad):
        if t < 30:
            pad.buttons |= L1
        if t > 60:
            self._finish("def", Def.PERFECT_BLOCK)

    def _turn_to_face(self, p, e, t, pad):
        """Unlock, swing round to face them with the stick, lock back on."""
        if t == 0.0:
            self._turn_relock = bool(p.locked_on)
        if t < 30:
            if p.locked_on:
                pad.buttons |= R3                  # unlock: free turning is faster
            return
        facing_ok = self._facing_off_deg(p, e) < 20.0
        if (facing_ok and t > 120) or t > TURN_MAX_MS:
            if self._turn_relock and not p.locked_on and t < TURN_MAX_MS + 60:
                pad.buttons |= R3                  # back on
                return
            return self._finish("def", Def.TURN_TO_FACE)
        self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=0.5))

    def _sidestep(self, p, e, t, pad):
        """Sprint at right angles to their line of fire, to the side with more open floor."""
        h = G.heading_to(p, e)
        if t == 0.0:
            walls = self._walls(p)
            room = (lambda off: walls.free_run(p.loc_x, p.loc_y, h + off, 3.0)) if walls else (lambda off: 3.0)
            self._side = 1.0 if room(math.pi / 2) >= room(-math.pi / 2) else -1.0
            self._say(f"{T.Anim(e.animation_type).name if e.animation_type in T.Anim._value2member_map_ else e.animation_type}"
                      f" aimed at us from {G.distance(p, e):.1f} m: sidestepping")
        if t >= SIDESTEP_MS:
            return self._finish("def", Def.SIDESTEP)
        self._set_stick(pad, self.steer.stick_for_heading(h + self._side * math.pi / 2, self.cam_yaw))
        if t > 60:
            self._sprint(pad)                     # the sprint guard keeps it a sprint, not a roll

    def _flee_rolls(self, p, e, t, pad):
        """Roll after roll straight away from them, bent toward the most open floor, until we're
        clear of the fire tempest (or it's over); run when out of stamina."""
        dist = G.distance(p, e)
        if t == 0.0:
            self._flee_next = 0.0
            self._say(f"{'fire surge' if self._flee_set is SURGE_ANIMS else 'fire tempest'} at {dist:.1f} m: rolling away")
        storm = e.animation_type in self._flee_set
        if t > FLEE_MAX_MS or (dist >= self._flee_safe and t > FLEE_ROLL_GAP_MS) or (not storm and t > 1500):
            return self._finish("def", Def.FLEE_ROLLS)
        self._set_stick(pad, self.steer.stick_for_heading(self._open_heading(p, e), self.cam_yaw))
        if t >= self._flee_next and p.stamina >= ROLL_STAMINA:
            self._flee_tap_until = t + FLEE_TAP_MS
            self._flee_next = t + FLEE_ROLL_GAP_MS
            self._last_roll_t = self.now
        if t < getattr(self, "_flee_tap_until", -1.0):
            pad.buttons |= CIRCLE                  # short tap = roll

    def _hold_block(self, p, e, t, pad):
        """Shield up until their swing is over."""
        pad.buttons |= L1
        self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=0.3))
        if (t > 150 and e.sub >= Sub.ACTIVE_HURTBOX_OVER) or t > 1500:
            self._finish("def", Def.HOLD_BLOCK)

    def _parry(self, p, e, t, pad):
        if t < 30:
            pad.buttons |= L2
        if t > 60:
            self._finish("def", Def.PARRY)

    # --------------------------------------------------------------- attack
    def attack(self, p, e, inst: Instinct, net_choice: Atk, pad) -> None:
        start_bs = inst.attack == Atk.BACKSTAB and self.attack_type != Atk.BACKSTAB
        # a whiff punish (or the point-blank kick) cuts into spacing/strafing right away
        urgent = (inst.priority not in (Priority.ATTACK, Priority.DODGE)
                  and net_choice in (Atk.PUNISH, Atk.KICK, Atk.POKE, Atk.GHOST_HIT, Atk.RUN_BS)
                  and self.attack_type != net_choice
                  and self.in_any() and self._only_low_priority())
        if urgent:
            self._override_low_priority()
        if (not self.in_any() or start_bs) and p.sub >= Sub.RECOVER:
            if inst.priority == Priority.ATTACK:
                self.attack_type = inst.attack
            elif inst.priority == Priority.DELAY:
                # waiting for the dodge timing: keep moving (it stood still, live 2026-11-23)
                self.attack_type = Atk.MOVE_UP if net_choice == Atk.MOVE_UP else Atk.SPACE
            elif inst.priority == Priority.DODGE and net_choice in (Atk.SPACE, Atk.MOVE_UP):
                # a hit is coming: dodging is the job. (Live 2026-11-23 the dodge path cancelled
                # SPACE every tick and this restarted it every tick: ~60 restarts per swing.)
                self.attack_type = Atk.NONE
            else:
                self.attack_type = net_choice
            if self.attack_type != Atk.NONE:
                self.attack_state = ACTIVE
                self.t_atk = self.now
                if self.attack_type in (Atk.TWO_HAND, Atk.SWITCH_WEAPON):
                    self._last_equip_change = self.now
                self._say(f"start attack {self.attack_type.name}")
        if not self.in_attack():
            return
        fn = {
            Atk.MOVE_UP: self._move_up, Atk.GHOST_HIT: self._ghost_hit,
            Atk.DEAD_ANGLE: self._dead_angle, Atk.BACKSTAB: self._backstab,
            Atk.TWO_HAND: self._two_hand, Atk.SWITCH_WEAPON: self._switch_weapon,
            Atk.HEAL: self._heal, Atk.PIVOT_BS: self._pivot_bs, Atk.SPACE: self._space,
            Atk.PUNISH: self._punish, Atk.KICK: self._kick, Atk.RACE: self._race, Atk.POKE: self._poke,
            Atk.WAIT_MIDDLE: self._wait_middle, Atk.LINE_UP: self._line_up,
            Atk.FOOTSIE: self._footsie, Atk.RUN_BS: self._run_bs,
            Atk.JUMP: self._jump, Atk.ROLL_GHOST: self._roll_ghost, Atk.TURTLE: self._turtle,
            Atk.CHARGE: self._charge, Atk.OKI: self._oki,
            Atk.KEEP_OUT: self._keep_out, Atk.KNIVES: self._knives,
        }.get(self.attack_type)
        if fn is None:
            self.attack_state = INACTIVE
            return
        fn(p, e, self.now - self.t_atk, pad)

    def _ghost_hit(self, p, e, t, pad):
        if p.locked_on and t < 50:
            pad.buttons |= R3
        if 10 < t < 50:
            pad.buttons |= R1                      # start the swing
        if p.sub == Sub.WINDUP_GHOST_HIT:
            self._set_stick(pad, self._toward(p, e))           # swing back into them
        elif t > 50:
            self._set_stick(pad, self._toward(p, e, 180.0))    # face away during windup
        if t > 500 and p.sub > Sub.WINDUP_GHOST_HIT:
            self._finish("atk", Atk.GHOST_HIT)

    def _dead_angle(self, p, e, t, pad):
        if p.locked_on and t < 50:
            pad.buttons |= R3
        if p.animation_type in T.DODGE_ANIM:
            self._set_stick(pad, self._toward(p, e))  # leave the roll first so we don't kick
            self.t_atk = self.now
        elif t < 50:
            pad.buttons |= R1
        else:
            self._set_stick(pad, self._toward(p, e, 60.0))
        if t > 500:
            self._finish("atk", Atk.DEAD_ANGLE)

    def _backstab(self, p, e, t, pad):
        """Backstab input, as one of two option selects (user, 2026-11-23):
          * KICK (stick flicked toward them + R1): a kick counts as a backstab input. Miss the
            angle and it still kicks them (kickstun / knockdown -> R1 or another attempt);
            blocked, it's back to neutral. Much safer than a whiffed R1. Used in neutral.
          * STANCE CANCEL (Y, then R1 - in that order): backstabs if the angle is right,
            otherwise just swaps 1H/2H stance. Safe only when they can't race us to the
            backstab, so used on oki (they're getting up / stunned).
        The stick goes neutral first so the kick's forward is a fresh input: walking forward
        into R1 gave plain R1s (live 2026-11-22)."""
        if t == 0.0:
            # Havel build: always the kick input (user). 2026-11-23: 30 stance cancels on oki
            # missed and swapped the club's grip - a slow animation that got backstabbed.
            self._bs_method = ("stance" if self.now < self._oki_until and not self.cfg.backstab_only
                               else "kick")
            self._bs_attempt_t = self.now
            if self._bs_method == "kick":
                self._kick_t = self.now
            fx, fy = G.facing_vector(e)
            relx, rely = p.loc_x - e.loc_x, p.loc_y - e.loc_y
            behind, side = -(relx * fx + rely * fy), relx * fy - rely * fx
            self._say(f"backstab attempt ({'stance cancel, oki' if self._bs_method == 'stance' else 'kick'}): "
                      f"{G.distance(p, e):.2f} m, "
                      f"{math.degrees(math.atan2(abs(side), behind)):.0f} deg off their back, "
                      f"facing {G.angle_diff_deg(e.rotation_deg, p.rotation_deg):.0f} deg, "
                      f"their state {e.sub.name} (anim {e.animation_type})")
        if self._bs_method == "stance":
            if t < 80:
                pad.buttons |= TRIANGLE            # Y first...
            if 30 <= t < 110:
                pad.buttons |= R1                  # ...then R1 (R1 first would be an R1)
        else:
            # kick input aimed straight at them, unlocked (user 2026-11-24: kicks aim freely;
            # 2026-11-23's club R1s came from flicking toward them while locked on)
            if self._aimed_flick(p, e, t, pad, R1):
                self._finish("atk", Atk.BACKSTAB)
                return
        if self._bs_method == "stance" and t > 220:
            self._finish("atk", Atk.BACKSTAB)

    def _aimed_flick(self, p, e, t, pad, button) -> bool:
        """Unlock (kicks aim freely unlocked), stick centred, then flicked straight AT them
        with the button: R1 = kick, R2 = jumping attack (user, 2026-11-24). True when done."""
        if t == 0.0:
            self._kick_unlock = bool(p.locked_on)
            # stick already centred long enough (ready stance): the flick can go out sooner
            centred_for = (self.now - self._stick_centred_since) if self._stick_centred_prev else 0.0
            self._flick_lead = (KICK_NEUTRAL_MS + KICK_UNLOCK_MS + 30 if self._kick_unlock
                                else max(0.0, KICK_NEUTRAL_MS - centred_for))
        if self._kick_unlock and t < KICK_UNLOCK_MS:
            pad.buttons |= R3
        if button == R2 and self._flick_lead <= t < self._flick_lead + 8 and p.animation_type != 0:
            # still in an animation: an R2 now would buffer into a plain heavy. Keep the stick
            # centred and wait (or give up), never press R2 without the flick.
            if t > FLICK_GIVE_UP_MS:
                return True
            self._flick_lead = t + KICK_NEUTRAL_MS
            return False
        lead = self._flick_lead
        if lead <= t < lead + 100:
            # aim where they'll be when the kick connects, not where they were
            tx, ty = e.loc_x + self._e_vel[0] * KICK_LEAD_S, e.loc_y + self._e_vel[1] * KICK_LEAD_S
            self._set_stick(pad, self.steer.stick_for_heading(math.atan2(tx - p.loc_x, ty - p.loc_y),
                                                              self.cam_yaw))
            if t < lead + 60:
                pad.buttons |= button
        return t > lead + 180

    def _kick(self, p, e, t, pad):
        """Kick, aimed at them."""
        if t == 0.0:
            self._kick_t = self.now
        if self._aimed_flick(p, e, t, pad, R1):
            self._finish("atk", Atk.KICK)

    def _jump(self, p, e, t, pad):
        """Jumping attack, aimed at them."""
        if self._aimed_flick(p, e, t, pad, R2):
            self._finish("atk", Atk.JUMP)

    def _roll_ghost_inputs(self, p, e, t, pad) -> bool:
        """Roll past the side of their shield, rolling attack out of it, ghost-turned into them."""
        if t == 0.0:
            self._kick_unlock = bool(p.locked_on)
            self._side = self.rng.choice([-1.0, 1.0])
        if self._kick_unlock and t < KICK_UNLOCK_MS:
            pad.buttons |= R3
        side = self._toward(p, e, 40.0 * self._side)
        if 50 <= t < 140:
            self._set_stick(pad, side)
            pad.buttons |= CIRCLE                  # tap: roll toward their flank
        elif 140 <= t < 560:
            self._set_stick(pad, side)
        elif 560 <= t < 620:
            pad.buttons |= R1                      # rolling attack
            self._set_stick(pad, self._toward(p, e, 180.0))
        elif 620 <= t < 800:
            self._set_stick(pad, self._toward(p, e, 180.0))   # ghost: wind up facing away...
        elif 800 <= t < 1100:
            self._set_stick(pad, self._toward(p, e))          # ...and swing back into them
        return t > 1100

    def _roll_ghost(self, p, e, t, pad):
        if self._roll_ghost_inputs(p, e, t, pad):
            self._finish("atk", Atk.ROLL_GHOST)

    @staticmethod
    def _spacing_reach(e) -> float:
        """Their reach for spacing purposes: learned reach, capped at a melee R1's (a long-range
        R2 or projectile hit once made the bot hover at 6-7 m from a pyromancer)."""
        return min(e.weapon_range, SPACING_REACH_CAP)

    def _camping(self, e, dist: float, hold: float) -> bool:
        """Standing still (not behind a shield), or casting / buffing, out at range."""
        if not (max(CHARGE_MIN_M, hold + 0.3) <= dist <= CHARGE_MAX_M) or e.hp <= 1:
            return False
        casting = e.animation_type in CAST_ANIMS
        still = (e.sub == Sub.NEUTRAL and self._enemy_static_ms() >= CHARGE_STATIC_MS
                 and not self._shield_mostly_up())
        return casting or still

    def _they_are_down(self, p, e) -> bool:
        """On the floor or stunned: our backstab just ended, they're in the backstab / kickstun,
        or staggered. (Not while WE are down or mid-backstab.)"""
        if self.is_down(p) or p.animation_type == BACKSTAB_ANIM or e.hp <= 1:
            return False
        return e.in_backstab or e.sub == Sub.POISE_BROKEN or self.now < self._oki_until

    def _charge(self, p, e, t, pad):
        """Unlocked, block + sprint together from the first instant, straight at them; close
        enough and the race / line-up / kick takes over."""
        dist = G.distance(p, e)
        if t == 0.0:
            self._say(f"they're camping at {dist:.1f} m ({'casting' if e.animation_type in CAST_ANIMS else 'standing still'}): charging")
        if dist <= CHARGE_HANDOFF_M or t > CHARGE_MAX_MS or e.hp <= 1:
            return self._finish("atk", Atk.CHARGE)
        if p.locked_on and t < 40:
            pad.buttons |= R3
        self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw))
        if t < RACE_BLOCK_MS:
            pad.buttons |= L1
        if p.stamina >= ROLL_STAMINA:
            self._sprint(pad)

    def _oki(self, p, e, t, pad):
        """Wake-up pressure: get to the spot just behind them and wait, facing them; the instinct
        kick-backstabs the moment they're backstabbable again."""
        if t == 0.0:
            self._say("they're down: getting behind them for the wake-up")
        if not self._they_are_down(p, e) or t > OKI_MAX_MS:
            return self._finish("atk", Atk.OKI)
        if p.locked_on and t < 40:
            pad.buttons |= R3                       # unlocked: free movement to the spot
        fx, fy = G.facing_vector(e)
        bx, by = e.loc_x - fx * OKI_SPOT_M, e.loc_y - fy * OKI_SPOT_M
        to_spot = math.hypot(bx - p.loc_x, by - p.loc_y)
        if to_spot > OKI_ARRIVE_M:
            self._set_stick(pad, self.steer.stick_for_heading(math.atan2(bx - p.loc_x, by - p.loc_y),
                                                              self.cam_yaw))
            if to_spot > 2.5 and p.stamina >= ROLL_STAMINA:
                self._sprint(pad)
        elif self._facing_off_deg(p, e) > OKI_FACE_DEG:
            # there: turn to face them without walking into them
            self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=0.2))
        # else: in position and facing them - stick centred, unlocked: the kick goes out instantly

    def _corner_camping(self, p, e, dist: float) -> bool:
        """Standing still in an arena corner, and we have poison knives to throw."""
        it, walls = self.items, self._walls(p)
        if it is None or it.knife_at is None or walls is None or e.hp <= 1 or dist > 15.0:
            return False
        if (self.now < self._knife_next or self.waking_up() or getattr(it, "active", False)
                or dist < KNIFE_START_M):
            return False
        near_corner = min(math.hypot(e.loc_x - vx, e.loc_y - vy) for vx, vy in walls.polygon) <= CORNER_M
        return near_corner and self._enemy_static_ms() >= CORNER_STILL_MS

    def _keep_out(self, p, e, t, pad):
        """Fire surge: stay just out of its range, facing them; back off if we're inside it."""
        dist = G.distance(p, e)
        if e.animation_type not in SURGE_ANIMS or t > 6000:
            return self._finish("atk", Atk.KEEP_OUT)
        if dist < SURGE_SAFE_M:
            self._set_stick(pad, self.steer.stick_for_heading(self._open_heading(p, e), self.cam_yaw))
        else:
            self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=0.15))

    def _knives(self, p, e, t, pad):
        """Corner camper: back off, switch to the poison knives, lock on, throw until they leave
        their spot by KNIFE_LEFT_M (or come for us), then switch back to the soapstone."""
        it, dist, k = self.items, G.distance(p, e), self._kn
        if t == 0.0:
            k.clear()
            k.update(phase="back", spot=(e.loc_x, e.loc_y), throws=0, next=0.0, tap_until=-1.0,
                     thrown_t=None, anim=False, empty=0, lock_t=None, lock_tries=0)
            self._say(f"they're camping a corner ({dist:.1f} m): backing off for poison knives")
        n = max(1, it.n_items)
        if k["phase"] in ("back", "select", "throw"):
            left = math.hypot(e.loc_x - k["spot"][0], e.loc_y - k["spot"][1]) >= KNIFE_LEFT_M
            why = ("they left the corner" if left else "too long" if t > KNIFE_MAX_MS
                   else "out of throws" if k["throws"] >= KNIFE_MAX else None)
            if why:
                self._say(f"knives: {why}: switching back to the soapstone")
                k["phase"] = "restore"
        if self.now < it.cycle_until:
            pad.hat = DDOWN                         # a d-pad press in progress
            return
        if k["phase"] == "back":
            if dist < KNIFE_RANGE_M and t < 3000:
                self._set_stick(pad, self.steer.stick_for_heading(self._open_heading(p, e), self.cam_yaw))
                return
            k["phase"] = "select"
        if k["phase"] == "select":
            if self.now < it.wait_until:
                return
            if it.cycles % n != it.knife_at % n:
                it._cycle(self.now, pad)
                return
            k["phase"], k["next"] = "throw", self.now + 300
            self._say("knives: poison knives selected")
        if k["phase"] == "throw":
            if not p.locked_on:
                k["tap_until"] = -1.0               # never throw unlocked (a tap in progress stops)
                if k["lock_t"] is None or self.now - k["lock_t"] >= KNIFE_LOCK_RETRY_MS:
                    if k["lock_tries"] >= KNIFE_LOCK_TRIES:
                        self._say("knives: can't lock on to them: switching back to the soapstone")
                        k["phase"] = "restore"
                        return
                    k["lock_t"], k["lock_tries"] = self.now, k["lock_tries"] + 1
                    self._say("knives: locking on")
                if self.now - k["lock_t"] < KNIFE_LOCK_HOLD_MS:
                    pad.buttons |= R3               # lock on: the knife flies at them
                k["next"] = max(k["next"], self.now + 100)   # let the camera settle first
                return
            k["lock_t"], k["lock_tries"] = None, 0
            if self.now < k["tap_until"]:
                pad.buttons |= SQUARE
                return
            if k["thrown_t"] is not None:
                if p.animation_type != 0:
                    k["anim"] = True
                if self.now - k["thrown_t"] > 900:
                    if not k["anim"]:
                        k["empty"] += 1
                        if k["empty"] >= 2:
                            self._say("knives: no throw animation twice (out of knives?)")
                            self._knife_next = self.now + KNIFE_EMPTY_BACKOFF_MS
                            k["phase"] = "restore"
                            return
                    k["thrown_t"] = None
            if self.now >= k["next"] and p.animation_type == 0 and k["thrown_t"] is None:
                k["tap_until"], k["next"] = self.now + KNIFE_TAP_MS, self.now + KNIFE_GAP_MS
                k["throws"] += 1
                self._say(f"knives: throw {k['throws']} ({dist:.1f} m)")
                k["thrown_t"], k["anim"] = self.now, False
                pad.buttons |= SQUARE
            return
        if k["phase"] == "restore":
            if self.now < it.wait_until:
                return
            if it.cycles % n != 0:
                it._cycle(self.now, pad)
                return
            self._knife_next = max(self._knife_next, self.now + KNIFE_COOLDOWN_MS)
            return self._finish("atk", Atk.KNIVES)

    def _no_cast(self, e, dist) -> bool:
        """They're in the no-cast pose, or just beat a race by turning with us, within mix-up
        range: don't circle them, hit them."""
        return dist <= TURTLE_M and (e.animation_type in NO_CAST_ANIMS or self.now < self._race_futile_until)

    def _shield_mostly_up(self) -> bool:
        up = sum(1 for _, sh in self._shield_hist if sh)
        return bool(self._shield_hist) and up >= TURTLE_SHIELD_FRAC * len(self._shield_hist)

    def _turtling(self, p, e, dist: float) -> bool:
        """Standing still, shield up, facing us, close (user, 2026-11-24)."""
        if dist > TURTLE_M or e.sub != Sub.NEUTRAL or self._enemy_static_ms() < TURTLE_MS:
            return False
        return self._they_face_us(p, e, 60.0) and self._shield_mostly_up()

    def _turtle(self, p, e, t, pad):
        """Break a turtle: pick kick (mostly) / jumping attack / rolling ghost strike, close to
        its reach, do it. A kick then chains into the block+sprint race to their back."""
        dist = G.distance(p, e)
        if t == 0.0:
            r, acc = self.rng.random(), 0.0
            for move, w in TURTLE_MIX:
                acc += w
                if r <= acc:
                    break
            if move == "roll" and not 1.5 <= dist <= 3.5:
                move = "kick"
            self._turtle_move, self._turtle_go_t = move, None
            why = ("no-cast loop" if e.animation_type in NO_CAST_ANIMS
                   else "they turn with us" if self.now < self._race_futile_until else "shield up")
            self._say(f"they're turtling ({dist:.1f} m, {why}): {move}")
        move = self._turtle_move
        if self._turtle_go_t is None:
            if dist > TURTLE_REACH[move] + 0.15 and t < TURTLE_APPROACH_MS:
                self._set_stick(pad, self._toward(p, e))      # walk into reach
                return
            self._turtle_go_t = self.now
        t2 = self.now - self._turtle_go_t
        if move == "kick":
            if t2 == 0.0:
                self._kick_t = self.now
            done = self._aimed_flick(p, e, t2, pad, R1)
        elif move == "jump":
            done = self._aimed_flick(p, e, t2, pad, R2)
        else:
            done = self._roll_ghost_inputs(p, e, t2, pad)
        if done:
            self._turtle_next = self.now + self.rng.uniform(*TURTLE_GAP_MS)
            self._finish("atk", Atk.TURTLE)

    def _punish(self, p, e, t, pad):
        """Run in if they're just out of reach, then a plain R1 facing them (no ghost-hit
        turn: that shows them our back). Gives up if they recover before we get there."""
        dist = G.distance(p, e)
        if t == 0.0:
            self._punish_r1_t = None
        if self._punish_r1_t is None:
            if dist <= p.weapon_range:
                self._punish_r1_t = self.now
            elif t > 700 or e.sub in (Sub.NEUTRAL, Sub.WINDUP, Sub.WINDUP_CLOSING):
                return self._finish("atk", Atk.PUNISH)        # too late: they've recovered
            else:
                self._set_stick(pad, self._toward(p, e))      # close in at a run
                return
        r1_t = self.now - self._punish_r1_t
        if r1_t >= 120:                    # stick centred around the press: R1, not a kick
            self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw, magnitude=0.5))
        if r1_t < 60:
            pad.buttons |= R1
        if r1_t > 450 and p.sub > Sub.WINDUP_GHOST_HIT:
            self._finish("atk", Atk.PUNISH)
        elif r1_t > 1200:
            self._finish("atk", Atk.PUNISH)

    def _footsie(self, p, e, t, pad):
        """Neutral (user, 2026-11-23): locked on, step into their reach to threaten, back out
        to bait their forward movement; random phase lengths so it can't be read. Re-decided
        every 300 ms like SPACE (low priority: dodges, punishes, run backstabs cut in)."""
        if t >= 300:
            return self._finish("atk", Atk.FOOTSIE)
        if not p.locked_on and t < 30:
            pad.buttons |= R3
        if self._footsie_flip_at is None or self.now >= self._footsie_flip_at:
            self._footsie_in = not self._footsie_in if self._footsie_flip_at is not None else True
            self._footsie_flip_at = self.now + self.rng.uniform(*FOOTSIE_PHASE_MS)
            self._strafe_dir = self.rng.choice([-1.0, 1.0])
        dist = G.distance(p, e)
        want = self._spacing_reach(e) + (self._st("footsie_in_m") if self._footsie_in else self._st("footsie_out_m"))
        radial = max(-1.0, min(1.0, (dist - want) / 0.8))           # + = in, - = out
        h = G.heading_to(p, e)
        walls = self._walls(p)
        if walls is not None and radial < 0 and walls.free_run(p.loc_x, p.loc_y, h + math.pi) < 1.0:
            radial = 0.0                                           # wall behind: don't back into it
        ang = math.atan2(self._strafe_dir * 0.35 * (1.0 - abs(radial)) + 0.15 * self._strafe_dir, radial)
        heading = h + ang
        if walls is not None:
            wd = walls.wall_distance(p.loc_x, p.loc_y)
            if wd < CENTER_PULL_M:                 # stay off the walls (as SPACE does)
                cx, cy = walls.center
                w = (CENTER_PULL_M - wd) / CENTER_PULL_M
                c = math.atan2(cx - p.loc_x, cy - p.loc_y)
                heading = math.atan2(math.sin(heading) + 1.5 * w * math.sin(c),
                                     math.cos(heading) + 1.5 * w * math.cos(c))
        self._set_stick(pad, self.steer.stick_for_heading(heading, self.cam_yaw,
                                                          magnitude=0.9 if abs(radial) > 0.3 else 0.5))

    def _run_bs(self, p, e, t, pad):
        """Run backstab (user, 2026-11-23): they come forward - sprint straight at them locked
        on and throw the kick-backstab on contact. Latency decides it: on their screen we're
        further through them than here, so the kick goes early."""
        dist = G.distance(p, e)
        if t == 0.0:
            self._run_kick_t = None
            self._say(f"run backstab ({dist:.1f} m, they're coming in at "
                      f"{self._enemy_closing_speed(p, e):.1f} m/s)")
        if t > RUN_BS_MAX_MS or (self._run_kick_t is None and (dist > RUN_BS_MAX_M + 1.0 or e.sub != Sub.NEUTRAL)):
            return self._finish("atk", Atk.RUN_BS)
        if not p.locked_on and t < 30:
            pad.buttons |= R3                      # locked on: we run straight at them
        if self._run_kick_t is None and t > 300 and dist > RUN_BS_BAIL_M and self._enemy_closing_speed(p, e) < 0.8:
            self._say("run backstab: they stopped coming, back to footsies")
            return self._finish("atk", Atk.RUN_BS)
        vx, vy = self._enemy_velocity()
        closing = max(0.1, self._enemy_closing_speed(p, e) + 5.0)    # them + our sprint (~5 m/s)
        contact_soon = dist / closing < RUN_BS_CONTACT_S
        if self._run_kick_t is None and (dist <= RUN_BS_KICK_M or contact_soon) and t > 60:
            self._run_kick_t = self.now
            self._bs_attempt_t = self.now
            self._kick_t = self.now
            self._say(f"run backstab: kick at {dist:.1f} m")
        if self._run_kick_t is None:
            self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw))
            if t > 40:
                self._sprint(pad)
            return
        k = self.now - self._run_kick_t
        if self._sprint_since is not None and self.now - self._sprint_since < SPRINT_HOLD_MS:
            pad.buttons |= CIRCLE                  # let the sprint finish first (a tap is a roll)
            self._set_stick(pad, self.steer.stick_for_heading(G.heading_to(p, e), self.cam_yaw))
            self._run_kick_t = self.now
            return
        if k < KICK_NEUTRAL_MS:
            return                                 # stick centred: the flick is a fresh input
        if k < KICK_NEUTRAL_MS + 100:
            self._set_stick(pad, self.steer.stick_for_heading(G.facing_heading(p), self.cam_yaw))
            if k < KICK_NEUTRAL_MS + 60:
                pad.buttons |= R1
            return
        if k > 400:
            self._finish("atk", Atk.RUN_BS)

    def _line_up(self, p, e, t, pad):
        """Unlocked, turn into them (moving toward them turns us to face them) until the kick
        is lined up; the instinct then takes the backstab."""
        dist = G.distance(p, e)
        if t > 500 or not self._at_their_back(p, e, dist) or self._bs_ready(p, e, dist):
            return self._finish("atk", Atk.LINE_UP)
        if p.locked_on and t < 30:
            pad.buttons |= R3
            return
        self._set_stick(pad, self._into_their_back(p, e))

    def _wait_middle(self, p, e, t, pad):
        """Walk to the middle of the arena and stand there, locked on if we can."""
        walls = self._walls(p)
        if walls is None or t > 300:
            return self._finish("atk", Atk.WAIT_MIDDLE)
        cx, cy = walls.center
        if math.hypot(cx - p.loc_x, cy - p.loc_y) > 1.0:
            self._set_stick(pad, self.steer.stick_for_heading(
                math.atan2(cx - p.loc_x, cy - p.loc_y), self.cam_yaw, magnitude=0.8))
        elif self.now - self._said_wall > 5000:
            self._said_wall = self.now
            self._say("they're outside the walls: waiting in the middle")

    def _poke(self, p, e, t, pad):
        """One plain R1 (stick centred, so it's not a kick), locked on so it goes at them."""
        if t < 60:
            pad.buttons |= R1
        if (t > 450 and p.sub > Sub.WINDUP_GHOST_HIT) or t > 1200:
            self._finish("atk", Atk.POKE)

    def _race(self, p, e, t, pad):
        """Race for their back (user, 2026-11-23): UNLOCKED from the start, block for a moment to
        get to max speed faster, then sprint round the side we're on to the spot behind them.
        Close to their back: lock on (snaps us to face them) for the kick-backstab, which the
        instinct fires once we're lined up. Locked-on strafing is for defence (SPACE)."""
        dist = G.distance(p, e)
        hold = self._spacing_reach(e) + self.cfg.spacing_margin
        kick_race = self.now < self._kick_race_until or self.now < self._back_chase_until
        if (t > RACE_MAX_MS or dist > hold + RACE_EXTRA_M + (3.0 if kick_race else 1.0)
                or (e.sub != Sub.NEUTRAL and not kick_race)
                or (self._no_cast(e, dist) and not kick_race and not self._back_turned)):
            return self._finish("atk", Atk.RACE)          # (no-cast loop: hit them instead)
        fx, fy = G.facing_vector(e)
        rx, ry = fy, -fx                           # their right
        relx, rely = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        theta = math.atan2(relx * rx + rely * ry, relx * fx + rely * fy)   # 0 front, +-pi back
        off_back = 180.0 - abs(math.degrees(theta))
        if t == 0.0 or self._race_side == 0.0:
            self._race_side = 1.0 if theta >= 0 else -1.0
            self._race_best = (off_back, self.now)
            self._race_sprint_t = None
            self._race_locking = False
            self._say(f"racing for their back ({dist:.1f} m, {off_back:.0f} deg off it)")
        if off_back < self._race_best[0] - 10:
            self._race_best = (off_back, self.now)
        elif self.now - self._race_best[1] > RACE_NO_PROGRESS_MS:
            if not kick_race and not self._back_turned:
                # they turn with us (a still opponent beats a race every time): hit them instead
                self._race_futile_until = self.now + RACE_FUTILE_MS
                self._say(f"they turn with us ({off_back:.0f} deg off their back): no racing, hitting them")
                return self._finish("atk", Atk.RACE)
            self._race_side = -self._race_side     # (stunned / back turned) other way round
            self._race_best = (off_back, self.now)
        arriving = off_back < RACE_LOCK_DEG and dist < RACE_LOCK_M
        sprint_held = self._race_sprint_t is not None and self.now - self._race_sprint_t < SPRINT_MIN_MS
        if arriving and not sprint_held:
            # at their back: sprint off and turn into them, unlocked. (Locking on here failed
            # whenever the camera faced us - they were off screen, so R3 locked nothing.)
            self._race_sprint_t = None
            self._set_stick(pad, self._into_their_back(p, e))
            return
        self._race_locking = False
        if p.locked_on and t < 40:
            pad.buttons |= R3                      # unlocked from the start
        step = math.radians(min(RACE_STEP_DEG, max(0.0, off_back)))
        tgt = theta + self._race_side * step
        tx = e.loc_x + RACE_RADIUS_M * (math.cos(tgt) * fx + math.sin(tgt) * rx)
        ty = e.loc_y + RACE_RADIUS_M * (math.cos(tgt) * fy + math.sin(tgt) * ry)
        walls = self._walls(p)
        if walls is not None and not walls.contains(tx, ty):
            self._race_side = -self._race_side     # wall that way: round the other side
            tgt = theta + self._race_side * step
            tx = e.loc_x + RACE_RADIUS_M * (math.cos(tgt) * fx + math.sin(tgt) * rx)
            ty = e.loc_y + RACE_RADIUS_M * (math.cos(tgt) * fy + math.sin(tgt) * ry)
        self._set_stick(pad, self.steer.stick_for_heading(math.atan2(tx - p.loc_x, ty - p.loc_y), self.cam_yaw))
        chain = self.now < self._chain_until
        if chain:
            # after a kick: block AND sprint together from the first instant, then let go of
            # block and keep sprinting round to their back (user, 2026-11-24)
            if t < RACE_BLOCK_MS:
                pad.buttons |= L1
            if p.stamina >= ROLL_STAMINA:
                if self._race_sprint_t is None:
                    self._race_sprint_t = self.now
                self._sprint(pad)
        elif t < RACE_BLOCK_MS:
            pad.buttons |= L1                      # block while starting off: max speed sooner (user)
        elif p.stamina >= ROLL_STAMINA and (self._race_sprint_t is not None or dist > SPRINT_MIN_RUN_M):
            if self._race_sprint_t is None:
                self._race_sprint_t = self.now
            self._sprint(pad)                      # sprint (held, so no roll)

    def _space(self, p, e, t, pad):
        """Hold just outside their reach and strafe, switching direction at random intervals
        so a human can't read it. Re-decided every 300 ms."""
        if t >= 300:
            return self._finish("atk", Atk.SPACE)
        if not p.locked_on and t < 30:
            pad.buttons |= R3                      # lock on: strafe while facing them
        if self.now >= self._strafe_flip_at:
            self._strafe_dir = self.rng.choice([-1.0, 1.0])
            self._strafe_flip_at = self.now + self.rng.uniform(700, 2200)
        dist = G.distance(p, e)
        hold = self._spacing_reach(e) + self.cfg.spacing_margin
        # Live 2026-11-23 the bot just circled outside their reach ("very passive"). Every
        # couple of seconds step into the edge of their reach and back out: a human swings at
        # it, and the dodge + whiff punish take it from there.
        if self._next_probe_at is None:
            self._next_probe_at = self.now + self.rng.uniform(*PROBE_EVERY_MS)
        if self.now >= self._next_probe_at:
            self._probe_until = self.now + PROBE_MS
            self._next_probe_at = self.now + PROBE_MS + self.rng.uniform(*PROBE_EVERY_MS)
            if abs(dist - hold) < 1.0:
                self._say("probing their range")
        if self.now < self._probe_until:
            hold = self._spacing_reach(e) - 0.4
        radial = max(-1.0, min(1.0, (dist - hold) / 1.0))     # + = move in, - = back off
        h = G.heading_to(p, e)
        walls = self._walls(p)
        if walls is not None:
            if radial < 0 and walls.free_run(p.loc_x, p.loc_y, h + math.pi) < 1.0:
                radial = 0.0           # wall behind us: hold here (it covers our back), strafe
            side_h = h + math.atan2(self._strafe_dir, 0.0)
            if walls.free_run(p.loc_x, p.loc_y, side_h) < 0.8:
                self._strafe_dir = -self._strafe_dir           # don't grind along into a corner
                self._strafe_flip_at = self.now + self.rng.uniform(700, 2200)
        # mix "toward them" and "around them" (90 deg) by how far off our distance is
        ang = math.atan2(self._strafe_dir * (1.0 - abs(radial)), radial)
        heading = h + ang
        if walls is not None:
            # Stay off the walls: 2026-11-23 vs player B the bot spent 42% of neutral within
            # 1 m of a wall (backed into it, then pinned and backstabbed); the user, beating
            # the same host 6-3 with this character, 15%. Pull toward the middle near walls.
            wd = walls.wall_distance(p.loc_x, p.loc_y)
            if wd < CENTER_PULL_M:
                cx, cy = walls.center
                w = (CENTER_PULL_M - wd) / CENTER_PULL_M          # 0 at 2 m, 1 at the wall
                vx = math.sin(heading) + w * 1.5 * math.sin(math.atan2(cx - p.loc_x, cy - p.loc_y))
                vy = math.cos(heading) + w * 1.5 * math.cos(math.atan2(cx - p.loc_x, cy - p.loc_y))
                heading = math.atan2(vx, vy)
        self._set_stick(pad, self.steer.stick_for_heading(heading, self.cam_yaw, magnitude=0.8))

    def _move_up(self, p, e, t, pad):
        if t < 90:
            self._set_stick(pad, self._toward(p, e))
        else:
            self._finish("atk", Atk.MOVE_UP)

    def _two_hand(self, p, e, t, pad):
        if t < 40:
            pad.buttons |= TRIANGLE
        else:
            self._finish("atk", Atk.TWO_HAND)

    def _switch_weapon(self, p, e, t, pad):
        if t < 30:
            pad.buttons |= R3
        elif t < 300:
            pad.ly = YBOTTOM
            pad.hat = DLEFT
        if t > 500:
            self._finish("atk", Atk.SWITCH_WEAPON)

    def _heal(self, p, e, t, pad):
        # Never chosen: metal-crow disabled healing as broken. Kept for completeness.
        if t < 30:
            pad.buttons |= SQUARE
        if p.sub != Sub.LOCK_IN and t > 100:
            self.t_atk = self.now                  # game didn't register it: retry
        if t > 1830:
            self._finish("atk", Atk.HEAL)

    def _pivot_bs(self, p, e, t, pad):
        """Get behind the enemy: round their side to a point just behind them, then turn in
        to face their back. Ends when a backstab is possible (instinct then takes it).
        The original sprinted at a heading 10 deg off the enemy, i.e. straight into them."""
        dist = G.distance(p, e)
        stalled = t > 1000 and self._pivot_behind_t is None    # not getting behind (wall, they turn)
        if self._bs_ready(p, e, dist) or t > 2500 or stalled:
            self._pivot_angle = self._pivot_behind_t = None
            return self._finish("atk", Atk.PIVOT_BS)
        if t == 0.0:
            self._toggle_t = None
        if p.locked_on and t < 40 and dist > G.BS_ATTEMPT_RANGE + 0.3:
            pad.buttons |= R3                      # unlock so we can run round them
        fx, fy = G.facing_vector(e)
        rx, ry = fy, -fx                           # their right-hand side
        relx, rely = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        ahead = relx * fx + rely * fy              # >0: we're in front of them
        side = 1.0 if relx * rx + rely * ry >= 0 else -1.0   # go round the side we're on
        # phase: None = going round their side, set = behind them (with hysteresis, or the
        # two waypoints fight at the boundary and the bot wobbles in place)
        if self._pivot_behind_t is None and ahead < -0.2:
            self._pivot_behind_t = self.now
        elif self._pivot_behind_t is not None and ahead > 0.5:
            self._pivot_behind_t = None            # they turned to face us: go round again
        if self._pivot_behind_t is None:           # aim beside them and well back
            tx = e.loc_x + rx * side * PIVOT_SIDE_M - fx * 0.7
            ty = e.loc_y + ry * side * PIVOT_SIDE_M - fy * 0.7
        else:
            if dist <= G.BS_ATTEMPT_RANGE + 0.3:
                # Behind and close: toggle-backstab. Lock on (snaps us to face their back) and
                # nudge toward them. Live 2026-11-22 the bot spent 3.3 s behind and within
                # 1.5 m but facing 60-180 deg off, waiting to reach an exact spot first.
                if not p.locked_on and (self._toggle_t is None or self.now - self._toggle_t > 300):
                    self._toggle_t = self.now
                if self._toggle_t is not None and self.now - self._toggle_t < 40:
                    pad.buttons |= R3
                # locked on we keep facing them while we strafe onto the spot right behind
                # them; from a diagonal, facing them is still >45 deg off their facing
                bx, by = e.loc_x - fx * PIVOT_BEHIND_M, e.loc_y - fy * PIVOT_BEHIND_M
                if math.hypot(bx - p.loc_x, by - p.loc_y) > 0.3:
                    far = math.hypot(bx - p.loc_x, by - p.loc_y) > 0.6
                    h, mag = math.atan2(bx - p.loc_x, by - p.loc_y), (0.85 if far else 0.5)
                else:
                    h, mag = G.heading_to(p, e), 0.3
                self._set_stick(pad, self.steer.stick_for_heading(h, self.cam_yaw, magnitude=mag))
                return
            tx, ty = e.loc_x - fx * PIVOT_BEHIND_M, e.loc_y - fy * PIVOT_BEHIND_M
        dx, dy = tx - p.loc_x, ty - p.loc_y
        self._set_stick(pad, self.steer.stick_for_heading(math.atan2(dx, dy), self.cam_yaw))
        if math.hypot(dx, dy) > SPRINT_MIN_RUN_M and t > 60:
            self._sprint(pad)                      # sprint while far from the waypoint
