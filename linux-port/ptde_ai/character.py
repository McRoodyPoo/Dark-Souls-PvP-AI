"""Per-character state and "subanimation" classification (port of CharacterStruct.c).

The subanimation is metal-crow's key idea: from the animation id plus how long it
has been playing, look up when the hurtbox will appear, and bucket the attack into
windup / closing / active / over. The dodge logic keys off these buckets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from . import tables as T


class Sub(IntEnum):
    """Ordering matters: the decision code compares these with < and >=."""
    WINDUP = 0
    WINDUP_CLOSING = 1
    WINDUP_GHOST_HIT = 2
    ACTIVE_DURING_HURTBOX = 11
    LOCK_IN = 12
    ACTIVE_HURTBOX_OVER = 13
    POISE_BROKEN = 14
    RECOVER = 20
    NEUTRAL = 30


# Animations whose windup spans two ids; their timers get added together.
COMBINE = {
    6207: (6407, 0), 6407: (6407, 1),   # left-hand combustion
    6307: (6507, 0), 6507: (6507, 1),   # right-hand combustion
    6222: (6422, 0), 6422: (6422, 1),   # left-hand wrath of the gods
    6322: (6522, 0), 6522: (6522, 1),   # right-hand wrath of the gods
}

# After this long into our own swing we can still turn (ghost hit). Original
# hardcoded it for one weapon (WeaponGhostHitTime_QFS).
GHOST_HIT_TIME = 0.22

TIMER_RESYNC_MS = 150

REQUIRED = ("loc_x", "loc_y", "rotation", "animation_type", "hp")


@dataclass
class Character:
    is_player: bool
    loc_x: float = 0.0
    loc_y: float = 0.0
    rotation_raw: float = 0.0       # radians, -pi..pi
    rotation_deg: float = 0.0       # (raw + pi) in degrees, 0..360
    animation_type: int = 0
    hp: int = 0
    stamina: int = 0
    r_weapon: int = 0
    l_weapon: int = 0
    anim_id: int = -1
    anim_id2: int = -1
    anim_id3: int = 0
    in_backstab: bool = False
    velocity: float = 0.0
    locked_on: bool = False
    two_handing: bool = False
    stam_recovery: int = 0
    poise: float = 0.0
    bleed_status: int = 0
    chr_type: int = -1
    team_type: int = -1
    weapon_range: float = 2.5
    sub: Sub = Sub.NEUTRAL
    dodge_time_remaining: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)


def current_attack(v: dict) -> tuple[int, str | None]:
    """(attack anim id, timer key) the timing tables are keyed by, as _classify picks it."""
    aid, aid2 = v.get("anim_id", -1), v.get("anim_id2", -1)
    if aid2 > 1000:
        return aid2, "anim_timer2"
    if aid > 1000:
        return (100 if aid % 1000 == 100 else aid), "anim_timer"   # kicks/parries share one
    return 0, None


class CharacterTracker:
    """Turns raw memory snapshots into Characters, keeping the per-character state
    the classification needs between ticks."""

    def __init__(self, is_player: bool, weapon_range: float):
        self.is_player = is_player
        self.weapon_range = weapon_range
        self.prev_sub = Sub.NEUTRAL
        self.waiting_for_timer = False   # enemy only: new animation, timer not yet reset
        self.waiting_since = 0.0
        self.now_ms = 0.0
        # learned hit times (HitMemory.hit_time) for attacks missing from the timing table:
        # live 2026-11-23 a lunge (anim id 514500) fell back to the 0.21 s default but hit at
        # ~0.65 s, so the bot read it as a whiff, ran in to punish, and ate it twice.
        self.timings = None

    def update(self, snap, history=None, now_ms: float = 0.0) -> Character | None:
        v = snap.values
        if any(k not in v for k in REQUIRED):
            return None
        c = Character(is_player=self.is_player, weapon_range=self.weapon_range, raw=v)
        c.loc_x, c.loc_y = v["loc_x"], v["loc_y"]
        c.rotation_raw = v["rotation"]
        c.rotation_deg = math.degrees(c.rotation_raw + math.pi)
        c.animation_type = v["animation_type"]
        c.hp = v["hp"]
        c.stamina = v.get("stamina", 0)
        c.r_weapon = v.get("r_weapon", 0)
        c.l_weapon = v.get("l_weapon", 0)
        c.anim_id = v.get("anim_id", -1)
        c.anim_id2 = v.get("anim_id2", -1)
        c.anim_id3 = v.get("anim_id3", 0)
        c.in_backstab = c.anim_id3 > 0
        c.velocity = v.get("velocity", 0.0)
        c.locked_on = bool(v.get("locked_on", 0))
        # a bitfield: 0x80 = two-handed (katana 160 / 0, Great Club 192 / 32 measured 2026-11-23).
        # Reading any non-zero as two-handed left the club build one-handed 28% of the time.
        c.two_handing = bool(v.get("two_handing", 0) & 0x80)
        c.stam_recovery = v.get("stam_recovery", 0)
        c.poise = v.get("poise", 0.0)
        c.bleed_status = v.get("bleed_status", 0)
        c.chr_type = v.get("chr_type", -1)
        c.team_type = v.get("team_type", -1)

        if history is not None:
            if self.is_player:
                history.push_ai_hp(c.hp, now_ms)
            else:
                history.push_anim_type_enemy(c.animation_type, now_ms)
                newest = c.anim_id if c.anim_id != -1 else c.anim_id2
                if history.push_anim_id_enemy(newest):
                    self.waiting_for_timer = True
                    self.waiting_since = now_ms

        self.now_ms = now_ms
        self._classify(c, v)
        self.prev_sub = c.sub
        return c

    def _classify(self, c: Character, v: dict) -> None:
        aid, aid2 = c.anim_id, c.anim_id2
        atk = T.ATTACK_ANIM.get(c.animation_type, 0)
        sub = self.prev_sub   # C kept the previous value when nothing below matched
        if v.get("hurtbox_active"):
            sub = Sub.ACTIVE_DURING_HURTBOX

        if aid in T.VULNERABLE_ANIM_ID:
            sub = Sub.LOCK_IN
        elif 2000 <= aid <= 2056:           # staggers, knockdowns, launches
            sub = Sub.POISE_BROKEN
        elif c.animation_type in T.DODGE_ANIM and aid != -1:
            sub = Sub.LOCK_IN
        elif atk in (2, 4, 5):
            cur_id, timer_key = current_attack(v)
            if cur_id and timer_key in v:
                timer = v[timer_key]
                # The game doesn't zero the timer the instant a new animation starts.
                # The C waited for it to drop below 0.1 forever, so an attack first seen
                # mid-swing was treated as just starting and never dodged. Give up after
                # TIMER_RESYNC_MS and trust the timer.
                in_window = self.now_ms - self.waiting_since < TIMER_RESYNC_MS
                if self.waiting_for_timer and timer > 0.1 and in_window:
                    timer = 0.0
                else:
                    self.waiting_for_timer = False
                if cur_id in COMBINE:
                    cur_id, part = COMBINE[cur_id]
                    if part == 1:
                        timer += v.get("anim_timer2", 0.0)
                hit_at = T.DODGE_TIMINGS.get(cur_id)
                if hit_at is None:
                    learned = self.timings.hit_time(cur_id) if self.timings is not None else None
                    hit_at = learned if learned is not None else T.DODGE_TIMING_DEFAULT
                delta = hit_at - timer
                c.dodge_time_remaining = delta
                if delta >= 1.0:
                    sub = Sub.NEUTRAL
                elif delta > 0.55:
                    sub = Sub.WINDUP
                elif delta >= 0.15:
                    sub = Sub.WINDUP_CLOSING
                elif delta >= 0:
                    sub = Sub.ACTIVE_DURING_HURTBOX
                else:
                    sub = Sub.ACTIVE_HURTBOX_OVER
                if self.is_player and timer > GHOST_HIT_TIME and delta >= -0.3:
                    sub = Sub.WINDUP_GHOST_HIT
        elif atk == 1:
            sub = Sub.WINDUP
        elif atk == 3:
            sub = Sub.ACTIVE_DURING_HURTBOX
        else:
            sub = Sub.NEUTRAL

        if v.get("ready_state"):          # player only: can transition to a new action
            sub = Sub.RECOVER
        c.sub = sub


def fightable(enemy: Character, self_chr_type: int | None = None) -> tuple[bool, str]:
    """Only engage the other side's player characters; never NPCs or allies."""
    from .offsets import (CHR_TYPE, FIGHTABLE_BY_SELF, FIGHTABLE_UNKNOWN_SELF, TEAM_TYPE)
    from .offsets import npc_target
    allowed = FIGHTABLE_BY_SELF.get(self_chr_type, FIGHTABLE_UNKNOWN_SELF)
    desc = (f"chr type {enemy.chr_type} ({CHR_TYPE.get(enemy.chr_type, 'NPC/unknown')}), "
            f"team {enemy.team_type} ({TEAM_TYPE.get(enemy.team_type, '?')})")
    raw = getattr(enemy, "raw", None) or {}
    npc = npc_target(enemy.chr_type, raw.get("entity_id", 0))
    if npc:
        return True, f"{npc} ({desc})"
    return enemy.chr_type in allowed, desc


def pick_target(slots: list[dict], self_chr_type, me_xy, current: int | None,
                switch_margin: float = 2.0, in_arena=None) -> int | None:
    """Slot of the nearest live character we may fight (fightable() rules), keeping the
    current one unless it's gone/dead/not fightable or another is switch_margin m closer.
    in_arena(x, y): only people inside the arena walls count (not someone waiting at their
    spawn or in the dugout)."""
    from .offsets import FIGHTABLE_BY_SELF, FIGHTABLE_UNKNOWN_SELF, npc_target
    allowed = FIGHTABLE_BY_SELF.get(self_chr_type, FIGHTABLE_UNKNOWN_SELF)
    import math as _m
    cands = {s["slot"]: _m.hypot(s["x"] - me_xy[0], s["y"] - me_xy[1]) for s in slots
             if (s["chr_type"] in allowed or npc_target(s["chr_type"], s.get("entity_id", 0)))
             and s["hp"] > 1                                  # 1 HP = lost the round, leaving
             and (in_arena is None or in_arena(s["x"], s["y"]))}
    if not cands:
        return None
    best = min(cands, key=cands.get)
    if current in cands and cands[current] <= cands[best] + switch_margin:
        return current
    return best

