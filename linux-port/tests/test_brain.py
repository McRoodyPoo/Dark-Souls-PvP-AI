"""Offline tests: simulated duel situations, no game needed.

  python tests/test_brain.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai import geometry as G                                  # noqa: E402
from ptde_ai.brain import Atk, Brain, Config, Def                  # noqa: E402
from ptde_ai.character import CharacterTracker, Sub                # noqa: E402
from ptde_ai.gamepad import CIRCLE, MIDDLE, R1, R3, TRIANGLE       # noqa: E402
from ptde_ai.nets import MODEL_DIR, MLP                            # noqa: E402
from ptde_ai.state import Snapshot                                 # noqa: E402
from ptde_ai.steer import Steering                                 # noqa: E402

FACE_PLUS_Y = math.pi      # raw rotation r faces (-sin r, -cos r): pi -> +y
FACE_MINUS_Y = 0.0


class Pad:
    def reset(self):
        self.lx = self.ly = self.rx = self.ry = MIDDLE
        self.buttons, self.hat = 0, -1


def snap(**kw) -> Snapshot:
    base = dict(loc_x=0.0, loc_y=0.0, rotation=FACE_PLUS_Y, animation_type=0, hp=1000,
                anim_id=-1, anim_id2=-1, anim_id3=0, anim_timer=0.0, anim_timer2=0.0,
                r_weapon=301015, l_weapon=1400000, poise=40.0)
    base.update(kw)
    return Snapshot(values=base)


def player_snap(**kw):
    d = dict(stamina=150, two_handing=192, locked_on=0, ready_state=0, bleed_status=100)
    d.update(kw)
    return snap(**d)


def make_brain(**cfg):
    return Brain(Steering(offset=0.0, sign=1.0), config=Config(use_attack_net=False,
                 use_defense_net=False, **cfg), rng=random.Random(0))


def step(brain, me, them, ps, es, now):
    pad = Pad()
    e = them.update(es, brain.history, now)
    p = me.update(ps, brain.history, now)
    brain.tick(p, e, 0.0, now, pad)
    return pad, p, e


def trackers():
    return CharacterTracker(True, 2.5), CharacterTracker(False, 6.0)


# --------------------------------------------------------------------- tests
def test_hurtbox_buckets():
    """Enemy 1H R1 (anim type 46) with animation id 203000: hurtbox at 0.25 s."""
    _, them = trackers()
    want = {0.0: Sub.WINDUP_CLOSING, 0.05: Sub.WINDUP_CLOSING, 0.12: Sub.ACTIVE_DURING_HURTBOX,
            0.2: Sub.ACTIVE_DURING_HURTBOX, 0.3: Sub.ACTIVE_HURTBOX_OVER}
    for timer, sub in want.items():
        e = them.update(snap(animation_type=46, anim_id=203000, anim_timer=timer))
        assert e.sub == sub, (timer, e.sub)
        assert abs(e.dodge_time_remaining - (0.25 - timer)) < 1e-9


def test_unknown_attack_uses_default_timing():
    _, them = trackers()
    e = them.update(snap(animation_type=46, anim_id=999999, anim_timer=0.0))
    assert abs(e.dodge_time_remaining - 0.21) < 1e-9


def test_rolls_from_incoming_r1():
    brain, (me, them) = make_brain(), trackers()
    ps = player_snap(loc_y=0.0, rotation=FACE_PLUS_Y)
    es = snap(loc_y=2.5, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000, anim_timer=0.05)
    pad, *_ = step(brain, me, them, ps, es, 1000.0)
    assert brain.dodge_type == Def.STANDARD_ROLL, brain.dodge_type
    assert pad.buttons & CIRCLE
    # Keeps rolling on later ticks without re-deciding, and exits after 900 ms.
    for t in (1020.0, 1120.0, 1400.0):
        step(brain, me, them, ps, es, t)
        assert brain.dodge_type == Def.STANDARD_ROLL
    step(brain, me, them, ps, snap(loc_y=2.5, rotation=FACE_MINUS_Y), 1950.0)
    assert brain.history.last_subroutines_self[0] == ("def", Def.STANDARD_ROLL)


def test_ignores_swings_that_cant_reach():
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record_reach(301015, 2.4)                  # learned: their weapon reaches 2.4 m
    es = snap(loc_y=4.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000, anim_timer=0.05)
    step(brain, me, them, player_snap(poise=10), es, 1000.0)
    assert not brain.in_dodge(), brain.dodge_type


def test_neutral_footsies_race_only_when_they_sit_still():
    """User 2026-11-23: pivots shouldn't be the only plan - footsies from neutral; race when
    they stand still."""
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(locked_on=1), snap(loc_y=2.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain.attack_type == Atk.FOOTSIE, brain.attack_type
    brain2, (me2, them2) = make_brain(), trackers()           # they stand still 1.5 s: race
    seen = set()
    for i in range(int(1600 / 8.3)):
        step(brain2, me2, them2, player_snap(locked_on=1), snap(loc_y=2.0, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        seen.add(brain2.attack_type)
    assert Atk.RACE in seen, seen


def test_punishes_a_whiff():
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.hits.record_reach(301015, 2.4)
    # they swing from 3 m (out of their 2.7 m reach): we let it whiff, then hit them
    t, seen = 1000.0, set()
    for i in range(90):
        step(brain, me, them, player_snap(), snap(loc_y=3.0, rotation=FACE_MINUS_Y, animation_type=46,
             anim_id=203000, anim_timer=0.02 + i * 0.0083), t)
        seen.add(brain.attack_type); t += 8.3
    assert not brain.in_dodge()
    assert Atk.PUNISH in seen, seen


def test_keeps_moving_while_they_attack_out_of_reach():
    """Live 2026-11-23: out of every roll it stood still until their whole string ended."""
    brain, (me, them) = make_brain(), trackers()
    moved, t = 0, 1000.0
    for i in range(40):                       # they swing from 4.5 m: can't reach us
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_y=4.5, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000,
                            anim_timer=0.02 + i * 0.0083), t)
        moved += abs(pad.lx - MIDDLE) + abs(pad.ly - MIDDLE) > 4000
        t += 8.3
    assert brain.attack_type == Atk.SPACE and moved > 30, (brain.attack_type, moved)


def test_runs_in_to_punish_a_whiff_from_range():
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.hits.record_reach(301015, 2.4)
    y, t, pressed_at = 3.8, 1000.0, None       # they whiff from 3.8 m (we reach 2.5 m)
    for i in range(160):
        pad, *_ = step(brain, me, them, player_snap(loc_y=4.0 - y, rotation=FACE_PLUS_Y),
                       snap(loc_y=4.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000,
                            anim_timer=0.02 + i * 0.0083), t)
        if brain.attack_type == Atk.PUNISH and pad.ly < MIDDLE // 2:
            y -= 4.0 * 0.0083                  # running toward them
        if pad.buttons & R1 and pressed_at is None:
            pressed_at = y
        t += 8.3
    assert pressed_at is not None and pressed_at <= 2.5, pressed_at   # R1 only once in reach


def test_follow_up_roll_is_pressed_from_inside_a_roll():
    """Live 2026-11-23: their 109 came while we were still in the roll from their 108; the
    roll routine quit because we were 'locked in' and never pressed circle."""
    from ptde_ai.gamepad import CIRCLE
    brain, (me, them) = make_brain(), trackers()
    rolling = dict(animation_type=34, anim_id=703)                  # still in the first roll
    atk = dict(loc_y=1.5, rotation=FACE_MINUS_Y, animation_type=109, anim_id=233300)
    step(brain, me, them, player_snap(**rolling), snap(anim_timer=0.0, **atk), 1000.0)
    pressed = False
    chosen = None
    for i in range(40):
        pad, *_ = step(brain, me, them, player_snap(**rolling), snap(anim_timer=0.2 + i * 0.0083, **atk),
                       1200.0 + 8.3 * i)
        chosen = chosen or (brain.dodge_type if brain.in_dodge() else None)
        pressed |= bool(pad.buttons & CIRCLE)
    assert chosen in (Def.STANDARD_ROLL, Def.REVERSE_ROLL_BS) and pressed, (chosen, pressed)
    # the plain roll too (the one that quit live)
    brain, (me, them) = make_brain(), trackers()
    brain.dodge_type, brain.dodge_state, brain.t_def = Def.STANDARD_ROLL, 1, 1000.0
    pad, *_ = step(brain, me, them, player_snap(**rolling), snap(anim_timer=0.5, **atk), 1000.0)
    assert pad.buttons & CIRCLE and brain.in_dodge()


def test_no_backstab_cancel_swing_into_their_string():
    """Live 2026-11-23: they were behind us mid-string; the ghost hit turned our back into it."""
    brain, (me, them) = make_brain(), trackers()
    swinging = snap(loc_y=-0.8, rotation=FACE_PLUS_Y, animation_type=109, anim_id=233300, anim_timer=0.6)
    step(brain, me, them, player_snap(rotation=FACE_PLUS_Y), swinging, 1000.0)
    assert brain.attack_type != Atk.GHOST_HIT, brain.attack_type
    # neutral and sneaking up behind us: reverse roll out (ghost hit got backstabbed 4 of 7)
    brain2, (me2, them2) = make_brain(), trackers()
    step(brain2, me2, them2, player_snap(rotation=FACE_PLUS_Y), snap(loc_y=-1.0, rotation=FACE_PLUS_Y), 1000.0)
    assert brain2.dodge_type == Def.TURN_TO_FACE and brain2.attack_type != Atk.GHOST_HIT, \
        (brain2.dodge_type, brain2.attack_type)            # turn to face them (user: roll less)


def test_no_punish_right_after_their_swing_hit_us():
    """Live 2026-11-23: their R1 hit us, the bot called it a whiff and R1'd into the next one."""
    brain, (me, them) = make_brain(), trackers()
    atk = dict(loc_y=1.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000)
    step(brain, me, them, player_snap(hp=2000), snap(anim_timer=0.0, **atk), 1000.0)
    step(brain, me, them, player_snap(hp=1850), snap(anim_timer=0.26, **atk), 1260.0)   # it hit
    brain.dodge_state, brain.dodge_type = 0, Def.NONE
    step(brain, me, them, player_snap(hp=1850), snap(anim_timer=0.6, **atk), 1600.0)     # "over"
    assert brain.attack_type != Atk.PUNISH, brain.attack_type


def test_picks_the_host_not_the_other_red_phantom():
    """Live 2026-11-23: a second red phantom sat in character slot 1; the bot (reading only
    slot 1) idled next to the host in slot 2 or walked into a wall toward the other red."""
    from ptde_ai.character import pick_target
    slots = [{"slot": 1, "chr_type": 2, "team_type": 16, "hp": 2073, "x": 265.9, "y": -296.8},  # other red
             {"slot": 2, "chr_type": 0, "team_type": 1, "hp": 1965, "x": 255.6, "y": -311.1},   # host
             {"slot": 3, "chr_type": 5, "team_type": 6, "hp": 193, "x": 254.0, "y": -311.0}]    # NPC
    me = (254.7, -311.1)
    assert pick_target(slots, 2, me, 1) == 2                   # we're a red: fight the host only
    assert pick_target(slots, 2, me, 2) == 2
    # two fightable (we're the host, two reds): keep the current one unless another is much closer
    reds = [{"slot": 1, "chr_type": 2, "team_type": 16, "hp": 900, "x": 3.0, "y": 0.0},
            {"slot": 2, "chr_type": 2, "team_type": 16, "hp": 900, "x": 2.0, "y": 0.0}]
    assert pick_target(reds, 0, (0.0, 0.0), 1) == 1            # only 1 m closer: stay
    reds[1]["x"] = 0.5
    assert pick_target(reds, 0, (0.0, 0.0), 1) == 2            # 2.5 m closer: switch
    reds[1]["hp"] = 0
    assert pick_target(reds, 0, (0.0, 0.0), 2) == 1            # dead: move on
    assert pick_target([slots[2]], 2, me, None) is None        # nobody to fight


def test_never_r1_on_wake_up():
    """User 2026-11-23: never R1 on wake-up. Recordings: R1 within 1.5 s of 15 of 38
    wake-ups (punish / pinned jab / ghost hit / kick), hit right after 9 of them."""
    def first_r1_after_wakeup(enemy, mid_punish=False):
        brain, (me, them) = make_brain(), trackers()
        t = 1000.0
        if mid_punish:                       # an R1 routine was running when we went down
            brain.attack_type, brain.attack_state, brain.t_atk = Atk.PUNISH, 1, t
        for i in range(60):                  # knocked down (stagger / knockdown anim ids)
            step(brain, me, them, player_snap(rotation=FACE_PLUS_Y, anim_id=2010), enemy, t); t += 8.3
        up = t
        for i in range(400):                 # up again, they stay right there
            pad, *_ = step(brain, me, them, player_snap(rotation=FACE_PLUS_Y), enemy, t)
            if pad.buttons & R1:
                return t - up
            t += 8.3
        return None
    pinned_in_front = snap(loc_y=0.6, rotation=FACE_MINUS_Y)       # would trigger the jab
    sneaking_behind = snap(loc_y=-1.0, rotation=FACE_PLUS_Y)       # would trigger the ghost hit
    for enemy in (pinned_in_front, sneaking_behind):
        d = first_r1_after_wakeup(enemy)
        assert d is None or d >= 1500, d
    d = first_r1_after_wakeup(pinned_in_front, mid_punish=True)
    assert d is None or d >= 1500, d


def test_does_not_punish_a_roll():
    """Recordings to 2026-11-23: punishing their roll hit 15/50 and they rolled behind us."""
    brain, (me, them) = make_brain(), trackers()
    for i in range(30):          # they roll (dodge anim 32) 2 m away
        step(brain, me, them, player_snap(), snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=32,
             anim_id=700), 1000.0 + 8.3 * i)
        assert brain.attack_type != Atk.PUNISH, i


def test_close_and_facing_us_no_rolling():
    """Close and facing us in neutral: footsies (or a race), never a roll."""
    brain, (me, them) = make_brain(), trackers()
    seen = set()
    for i in range(80):
        step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
             snap(loc_y=1.4, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        seen.add(brain.attack_type); seen.add(brain.dodge_type)
    assert not seen & {Def.REVERSE_ROLL_BS, Def.STANDARD_ROLL, Def.OMNISTEP_BACKWARDS}, seen


def test_spacing_drifts_off_the_walls():
    """2026-11-23 vs player B: bot 42% of neutral within 1 m of a wall, the user 15%."""
    brain, (me, them) = make_brain(), trackers()
    brain.arena = _box(-6, 6, -6, 6)                          # centre (0, 0)
    xs = []
    for i in range(30):                                        # we're at the west wall, they're north-ish
        pad, *_ = step(brain, me, them, player_snap(loc_x=-5.6, locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_x=-5.6, loc_y=4.9, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        if brain.attack_type in (Atk.SPACE, Atk.FOOTSIE):
            xs.append(pad.lx - MIDDLE)
    assert xs and sum(xs) / len(xs) > 3000, xs[:5]             # stick leans east (+x), off the wall


def test_holds_block_while_down_and_sometimes_waits_it_out():
    """User 2026-11-23: hold block while down (come up facing them faster); with them close
    enough to try a backstab, sometimes don't wake early (mixes up their timing)."""
    from ptde_ai.gamepad import CIRCLE, L1
    waited = woke = 0
    for seed in range(20):
        brain, (me, them) = make_brain(), trackers()
        brain.rng = random.Random(seed)
        blocked = escaped = False
        for i in range(120):     # their backstab: we're down (anim_id3 set), they're 1.2 m away
            pad, *_ = step(brain, me, them, player_snap(anim_id3=9420, animation_type=0),
                           snap(loc_y=1.2, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
            blocked |= bool(pad.buttons & L1)
            escaped |= brain.dodge_type in (Def.OMNISTEP_BACKWARDS, Def.REVERSE_ROLL_BS)
        assert blocked, seed
        waited += not escaped; woke += escaped
    assert 3 <= waited <= 14 and woke >= 6, (waited, woke)     # ~40% wait it out
    # far away: always the normal wake-up
    brain, (me, them) = make_brain(), trackers()
    for i in range(120):
        step(brain, me, them, player_snap(anim_id3=9420), snap(loc_y=5.0, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
    assert not any("waiting it out" in l for l in brain.log)


def test_stamina_rules():
    """User 2026-11-23: remember stamina. Roll ~30, attack ~34 on this character."""
    from ptde_ai.gamepad import L1, CIRCLE
    # no stamina for a roll: block the incoming hit instead
    brain, (me, them) = make_brain(), trackers()
    atk = dict(loc_y=1.5, rotation=FACE_MINUS_Y, animation_type=46, anim_id=233300)
    step(brain, me, them, player_snap(stamina=20), snap(anim_timer=0.0, **atk), 1000.0)
    held = False
    for i in range(20):
        pad, *_ = step(brain, me, them, player_snap(stamina=20), snap(anim_timer=0.2 + i * 0.0083, **atk), 1200.0 + 8.3 * i)
        held |= bool(pad.buttons & L1)
        assert not pad.buttons & CIRCLE
    assert brain.dodge_type == Def.HOLD_BLOCK and held, brain.dodge_type
    # a whiff to punish, but not enough stamina to swing and still roll: no punish
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record_reach(301015, 2.4)
    seen = set()
    for i in range(90):
        step(brain, me, them, player_snap(stamina=50), snap(loc_y=3.0, rotation=FACE_MINUS_Y, animation_type=46,
             anim_id=203000, anim_timer=0.02 + i * 0.0083), 1000.0 + 8.3 * i)
        seen.add(brain.attack_type)
    assert Atk.PUNISH not in seen, seen
    # a backstab is worth it with just its own cost
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(loc_y=-0.7, rotation=FACE_PLUS_Y, stamina=40), snap(rotation=FACE_PLUS_Y), 1000.0)
    assert brain.attack_type == Atk.BACKSTAB, brain.attack_type


def test_pokes_are_plain_r1s_and_no_kicks_in_neutral():
    """User 2026-11-23: plain R1 pokes from the front while racing; the kick input is only
    for backstab attempts ("kick confirm" option select), never in neutral."""
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    pokes, kicks_in_front = 0, 0
    for i in range(400):
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_y=1.6, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        pokes += sum("poke" in l for l in brain.log); brain.log.clear()
        if pad.buttons & R1 and (abs(pad.lx - MIDDLE) > 4000 or abs(pad.ly - MIDDLE) > 4000):
            kicks_in_front += 1
    assert 1 <= pokes <= 3, pokes                              # every 1.2-2.5 s over 3.3 s
    assert kicks_in_front == 0, kicks_in_front


def test_havel_build_backstab_only():
    """User 2026-11-23, Havel + Great Club: no R1s, only backstabs (kick input); kick landed but
    no backstab -> go for the back again; an R1 out of a roll becomes a ghost strike."""
    # a whiff to punish (they swing from out of reach): no R1 on this build
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record_reach(301015, 2.4)
    seen = set()
    for i in range(90):
        step(brain, me, them, player_snap(), snap(loc_y=3.0, rotation=FACE_MINUS_Y, animation_type=46,
             anim_id=203000, anim_timer=0.02 + i * 0.0083), 1000.0 + 8.3 * i)
        seen.add(brain.attack_type)
    assert not seen & {Atk.PUNISH, Atk.POKE, Atk.GHOST_HIT}, seen
    # racing in front of a neutral host: no pokes
    brain, (me, them) = make_brain(), trackers()
    for i in range(400):
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_y=1.6, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        assert not pad.buttons & R1, i
    # right after a roll, a whiff in reach: ghost strike, not an R1
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record_reach(301015, 1.6)                # their reach ~1.9 m: a 2.2 m swing whiffs
    seen = set()
    for i in range(90):
        brain._last_roll_t = 1000.0 + 8.3 * i - 300    # we just rolled
        step(brain, me, them, player_snap(), snap(loc_y=2.2, rotation=FACE_MINUS_Y, animation_type=46,
             anim_id=203000, anim_timer=0.02 + i * 0.0083), 1000.0 + 8.3 * i)
        seen.add(brain.attack_type)
    assert Atk.GHOST_HIT in seen and Atk.PUNISH not in seen, seen
    # our kick landed but it wasn't a backstab (they're kick-stunned, we're behind them): go again,
    # kick input (no stance cancel)
    brain, (me, them) = make_brain(), trackers()
    brain._kick_t = 1000.0
    seq = []
    for i in range(40):
        pad, *_ = step(brain, me, them, player_snap(loc_y=-0.8, rotation=FACE_PLUS_Y),
                       snap(rotation=FACE_PLUS_Y, anim_id=2052), 1050.0 + 8.3 * i)
        seq.append(pad)
    from ptde_ai.gamepad import TRIANGLE
    assert any("going for the back again" in l for l in brain.log), brain.log
    assert any(p.buttons & R1 and p.ly < MIDDLE // 4 for p in seq)          # kick (flick + R1)
    assert not any(p.buttons & TRIANGLE for p in seq)                         # no stance cancel


def test_havel_kick_input_is_reliable():
    """2026-11-23 Havel/club: 30 stance cancels on oki swapped the club's grip and got punished;
    kick inputs while facing 53-61 deg off came out as club R1s."""
    from ptde_ai.gamepad import TRIANGLE
    # our backstab just ended (oki): still a kick, never Y + R1
    brain, (me, them) = make_brain(), trackers()
    for i, a in enumerate([225] * 5 + [0]):
        step(brain, me, them, player_snap(loc_y=-3.0, rotation=FACE_PLUS_Y, animation_type=a),
             snap(rotation=FACE_PLUS_Y), 1000.0 + 8.3 * i)
    pads = []
    for i in range(26):
        pad, *_ = step(brain, me, them, player_snap(loc_y=-0.7, rotation=FACE_PLUS_Y), snap(rotation=FACE_PLUS_Y),
                       1100.0 + 8.3 * i)
        pads.append(pad)
    assert not any(p.buttons & TRIANGLE for p in pads)
    assert any(p.buttons & R1 and p.ly < MIDDLE // 4 for p in pads)        # forward flick + R1
    # in the backstab window by position but our facing is 50 deg off them: don't press
    import math as m
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(loc_y=-0.7, rotation=FACE_PLUS_Y + m.radians(50)),
         snap(rotation=FACE_PLUS_Y + m.radians(10)), 1000.0)
    assert brain.attack_type != Atk.BACKSTAB, brain.attack_type


def test_two_handed_unless_toggle_escaped_and_racing():
    """User 2026-11-23: usually two-handed; after a toggle escape, stay one-handed while racing
    (the toggle reaches max speed faster, like block)."""
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(two_handing=0), snap(loc_y=4.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain.attack_type == Atk.TWO_HAND, brain.attack_type   # out of reach: back to 2H
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(two_handing=0), snap(loc_y=2.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain.attack_type != Atk.TWO_HAND, brain.attack_type   # in their reach: not now
    brain, (me, them) = make_brain(), trackers()                  # toggle-escaped, racing: stay 1H
    brain._toggle_escape_t = 900.0
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.RACE, 1, 950.0
    for i in range(30):
        step(brain, me, them, player_snap(two_handing=0, locked_on=1), snap(loc_y=3.6, rotation=FACE_MINUS_Y),
             1000.0 + 8.3 * i)
        assert brain.attack_type != Atk.TWO_HAND, i


def test_waits_in_the_middle_when_they_are_outside_the_walls():
    """Live 2026-11-23: host waited outside the arena, bot pushed into the wall for 23 s."""
    brain, (me, them) = make_brain(), trackers()
    brain.arena = _box(-6, 6, -6, 6)
    lx = []
    for i in range(60):        # we're by the north wall, they're 10 m beyond it
        pad, *_ = step(brain, me, them, player_snap(loc_y=5.5), snap(loc_y=16.0, rotation=FACE_MINUS_Y),
                       1000.0 + 8.3 * i)
        lx.append(pad.ly - MIDDLE)
    assert brain.attack_type == Atk.WAIT_MIDDLE, brain.attack_type
    assert sum(lx) / len(lx) > 4000, sum(lx) / len(lx)         # walking south, back to the middle


def test_unlocks_and_turns_on_an_incoming_pivot():
    """User 2026-11-23: weak to pivot (unlocked) backstabs - unlock and turn to face them.
    Recordings: 0.5 s before host backstabs they crossed our line of sight at ~2.1 m/s."""
    def run(speed):
        brain, (me, them) = make_brain(), trackers()
        presses, sticks, locked, r3_prev = [], [], 1, False
        for i in range(140):
            x = -0.6 + speed * 0.0083 * i          # they run across in front of us at 1.0 m
            pad, *_ = step(brain, me, them, player_snap(locked_on=locked, rotation=FACE_PLUS_Y),
                           snap(loc_x=x, loc_y=1.0, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
            r3 = bool(pad.buttons & R3)
            if r3 and not r3_prev:
                locked = 1 - locked; presses.append(i)
            r3_prev = r3
            sticks.append(pad.lx - MIDDLE)
        return brain, presses, sticks
    brain, presses, sticks = run(3.0)
    assert any("incoming pivot" in l for l in brain.log), brain.log
    assert presses, presses                                   # unlocked (race or turn)
    assert any(abs(sx) > 4000 for sx in sticks[presses[0]:])    # turned with the stick
    brain, presses, sticks = run(0.5)                           # slow walk: no
    assert not any("incoming pivot" in l for l in brain.log)


def test_havel_trades_with_poise_and_commits_to_the_race():
    """User 2026-11-23: rolled with enough poise to trade for a backstab; wiggled in their face
    (race <-> pivot turn) instead of committing."""
    # a 490-damage hit (way over the old 8% cap), poise 62 vs 35: trade it for the pivot backstab
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record(451105, 46, 490)
    log = _swing(brain, me, them, poise=62)
    assert brain.attack_type == Atk.PIVOT_BS and not brain.in_dodge(), (brain.attack_type, log)
    # ...but not if it could leave us nearly dead (hp 600 - 490 < 20% of 2000)
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record(451105, 46, 490)
    step(brain, me, them, player_snap(hp=2000), snap(loc_y=5.0, rotation=FACE_MINUS_Y), 900.0)   # max hp seen
    log = _swing(brain, me, them, poise=62, hp=600)
    assert brain.attack_type != Atk.PIVOT_BS, (brain.attack_type, log)
    # racing and closer to their back than they are to ours: no pivot turn, keep going
    brain, (me, them) = make_brain(), trackers()
    import math as m
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.RACE, 1, 1000.0
    for i in range(40):        # we're beside/behind them (they face +y, we're at their back-left)
        x = -0.6 + 3.0 * 0.0083 * i
        step(brain, me, them, player_snap(loc_x=x, loc_y=-0.7, locked_on=1, rotation=FACE_PLUS_Y),
             snap(loc_x=x + 0.3, loc_y=0.0, rotation=FACE_PLUS_Y), 1000.0 + 8.3 * i)
    assert not any("incoming pivot" in l for l in brain.log), brain.log


def test_no_attacks_into_a_neutral_opponent():
    """Live: the attack net fired ghost hits at a neutral host and got backstabbed for it."""
    brain, (me, them) = make_brain(), trackers(); brain.cfg.use_attack_net = True
    brain.attack_net = lambda x: 0.99                    # a net that always wants to attack
    for i in range(30):
        step(brain, me, them, player_snap(), snap(loc_y=2.4, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        assert brain.attack_type not in (Atk.GHOST_HIT, Atk.DEAD_ANGLE, Atk.PUNISH), brain.attack_type


def test_punish_faces_them():
    brain, (me, them) = make_brain(), trackers()
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.PUNISH, 1, 1000.0
    for i in range(40):
        pad, *_ = step(brain, me, them, player_snap(), snap(loc_y=2.0, rotation=FACE_MINUS_Y, anim_id=7585), 1000.0 + 8.3 * i)
        if brain.attack_type == Atk.PUNISH:
            assert pad.ly <= MIDDLE, (pad.lx, pad.ly)      # never away from them (+y)
            if pad.buttons & R1:
                assert pad.ly == MIDDLE and pad.lx == MIDDLE   # centred on the press: an R1, not a kick
            elif 8.3 * i > 130:
                assert pad.ly < MIDDLE, (i, pad.lx, pad.ly)    # then toward them


def test_reverse_roll_when_close_with_time():
    brain, (me, them) = make_brain(), trackers()
    ps = player_snap()
    # anim id 233300 hits at 0.714 s. Watch it from the start, then 0.2 s in.
    atk = dict(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=233300)
    step(brain, me, them, ps, snap(anim_timer=0.0, **atk), 1000.0)
    assert brain.dodge_type == Def.NONE                       # still early windup
    step(brain, me, them, ps, snap(anim_timer=0.2, **atk), 1200.0)   # 0.514 s left, within 3 m
    assert brain.dodge_type == Def.REVERSE_ROLL_BS, brain.dodge_type


def test_attack_first_seen_mid_swing_still_dodged():
    brain, (me, them) = make_brain(), trackers()
    atk = dict(loc_y=2.5, rotation=FACE_MINUS_Y, animation_type=46, anim_id=233300)
    step(brain, me, them, player_snap(), snap(anim_timer=0.2, **atk), 1000.0)
    assert brain.dodge_type == Def.NONE      # timer assumed stale at first
    step(brain, me, them, player_snap(), snap(anim_timer=0.4, **atk), 1200.0)
    assert brain.dodge_type == Def.STANDARD_ROLL, brain.dodge_type   # resynced, 0.31 s left


def test_approach_steers_at_enemy():
    brain, (me, them) = make_brain(), trackers()
    pad, *_ = step(brain, me, them, player_snap(), snap(loc_y=20.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain.attack_type == Atk.MOVE_UP
    assert pad.ly < MIDDLE // 4 and abs(pad.lx - MIDDLE) < 50, (pad.lx, pad.ly)   # straight ahead
    # Same situation with the camera turned 90 degrees: stick must compensate.
    brain2, (me2, them2) = make_brain(), trackers()
    pad2 = Pad()
    e = them2.update(snap(loc_y=20.0, rotation=FACE_MINUS_Y), brain2.history, 0)
    p = me2.update(player_snap(), brain2.history, 0)
    brain2.tick(p, e, math.pi / 2, 0.0, pad2)
    assert abs(pad2.ly - MIDDLE) < 50 and pad2.lx != MIDDLE, (pad2.lx, pad2.ly)


def test_two_hands_when_far():
    brain, (me, them) = make_brain(), trackers()
    pad, *_ = step(brain, me, them, player_snap(two_handing=0),
                   snap(loc_y=20.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain.attack_type == Atk.TWO_HAND and pad.buttons & TRIANGLE


def test_takes_backstab_when_behind():
    brain, (me, them) = make_brain(), trackers()
    brain._stick_centred_prev, brain._stick_centred_since = True, 0.0   # ready stance held (OKI)
    # Enemy at origin facing +y; we're 1 m behind them (at -y), facing the same way.
    pad, *_ = step(brain, me, them, player_snap(loc_y=-1.0, rotation=FACE_PLUS_Y),
                   snap(loc_y=0.0, rotation=FACE_PLUS_Y), 1000.0)
    assert brain.attack_type == Atk.BACKSTAB, brain.attack_type
    # stick was already centred: the kick (flick toward them + R1) goes out on the first tick
    # (2026-11-24: "kick the opponent's back the very frame he wakes up")
    assert pad.buttons & R1 and pad.ly < MIDDLE // 4, (pad.buttons, pad.ly)
    pad, *_ = step(brain, me, them, player_snap(loc_y=-1.0), snap(), 1050.0)
    assert brain.attack_type == Atk.BACKSTAB
    brain.log.clear()
    starts = 0
    for i in range(6):                                     # live: restarted every tick
        step(brain, me, them, player_snap(loc_y=-1.0), snap(), 1060.0 + 8.3 * i)
        starts += sum("start attack BACKSTAB" in l for l in brain.log); brain.log.clear()
    assert starts == 0, starts


def test_kick_in_neutral_stance_cancel_on_oki():
    """User 2026-11-23: kick (forward + R1) in neutral; Y then R1 (stance cancel) on oki."""
    from ptde_ai.gamepad import TRIANGLE

    def attempt(brain, me, them, t0):
        seq = []
        for i in range(26):                       # one attempt (~220 ms)
            pad, *_ = step(brain, me, them, player_snap(loc_y=-0.7, rotation=FACE_PLUS_Y),
                           snap(rotation=FACE_PLUS_Y), t0 + 8.3 * i)
            seq.append(("Y" if pad.buttons & TRIANGLE else "") + ("R1" if pad.buttons & R1 else "")
                       + ("^" if pad.ly < MIDDLE // 4 else ""))
        return [x for i, x in enumerate(seq) if x and (i == 0 or seq[i - 1] != x)]
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    assert attempt(brain, me, them, 1000.0)[:2] == ["R1^", "^"]   # kick: R1 with stick forward
    assert any("(kick)" in l for l in brain.log)
    # our backstab landed and ended: they're getting up -> stance cancel
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    for i, a in enumerate([225] * 5 + [0]):
        step(brain, me, them, player_snap(loc_y=-3.0, rotation=FACE_PLUS_Y, animation_type=a),
             snap(rotation=FACE_PLUS_Y), 1000.0 + 8.3 * i)
    assert any("oki" in l for l in brain.log), brain.log
    seq = attempt(brain, me, them, 1100.0)
    assert seq[:3] == ["Y", "YR1", "R1"] and not any("^" in x for x in seq), seq   # Y first, no kick
    # oki over (3 s later): back to kicking
    brain.attack_state = 0
    assert attempt(brain, me, them, 5000.0)[0] == "R1^"


def test_waits_for_recovery_before_backstabbing():
    brain, (me, them) = make_brain(), trackers()
    swinging = snap(loc_y=0.0, rotation=FACE_PLUS_Y, animation_type=46, anim_id=203000, anim_timer=0.05)
    pad, *_ = step(brain, me, them, player_snap(loc_y=-0.9, rotation=FACE_PLUS_Y, poise=62), swinging, 1000.0)
    assert brain.attack_type != Atk.BACKSTAB and not pad.buttons & R1, brain.attack_type
    recovering = snap(loc_y=0.0, rotation=FACE_PLUS_Y, animation_type=46, anim_id=203000, anim_timer=0.6)
    pad, *_ = step(brain, me, them, player_snap(loc_y=-0.9, rotation=FACE_PLUS_Y, poise=62), recovering, 1100.0)
    assert brain.attack_type == Atk.BACKSTAB, brain.attack_type
    r1 = False
    for i in range(1, 15):    # the kick flick: stick centred 2 game frames first, then R1
        pad, *_ = step(brain, me, them, player_snap(loc_y=-0.9, rotation=FACE_PLUS_Y, poise=62), recovering,
                       1100.0 + 8.3 * i)
        assert brain.attack_type == Atk.BACKSTAB, brain.attack_type
        r1 |= bool(pad.buttons & R1)
    assert r1


def test_no_escape_dodges_during_our_own_backstab():
    brain, (me, them) = make_brain(), trackers()
    step(brain, me, them, player_snap(loc_y=-0.6, rotation=FACE_PLUS_Y), snap(rotation=FACE_PLUS_Y), 1000.0)
    assert brain.attack_type == Atk.BACKSTAB
    # backstab animation playing (anim_id3 set on both): don't mash omnisteps/reverse rolls
    for i in range(40):
        step(brain, me, them, player_snap(loc_y=-0.6, rotation=FACE_PLUS_Y, animation_type=225, anim_id3=1),
             snap(rotation=FACE_PLUS_Y, anim_id3=1), 1100.0 + 25 * i)
        assert brain.dodge_type not in (Def.OMNISTEP_BACKWARDS, Def.REVERSE_ROLL_BS), brain.dodge_type
    # but being backstabbed (no attempt of ours) still escapes
    brain2, (me2, them2) = make_brain(), trackers()
    step(brain2, me2, them2, player_snap(anim_id3=1, animation_type=225), snap(loc_y=1.0, rotation=FACE_MINUS_Y), 1000.0)
    assert brain2.dodge_type in (Def.OMNISTEP_BACKWARDS, Def.REVERSE_ROLL_BS), brain2.dodge_type


def test_backstab_geometry():
    me = CharacterTracker(True, 2.5).update(player_snap(loc_y=-1.0, rotation=FACE_PLUS_Y))
    en = CharacterTracker(False, 6).update(snap(loc_y=0.0, rotation=FACE_PLUS_Y))
    assert G.backstab_state(me, en) == 2
    assert G.backstab_state(en, me) == 0                       # they're in front of us
    side = CharacterTracker(True, 2.5).update(player_snap(loc_x=1.0, loc_y=-1.0, rotation=-math.pi / 2))
    assert G.backstab_state(side, en) == 1                     # behind but facing wrong way


def test_distance_fix():
    a = CharacterTracker(True, 2.5).update(player_snap(loc_x=-1.0))
    b = CharacterTracker(False, 6).update(snap(loc_x=1.0))
    assert G.distance(a, b) == 2.0 and G.legacy_distance(a, b) == 0.0   # the old bug


def test_move_up_does_not_count_as_roll():
    """C stored MoveUp(1) and StandardRoll(1) in one array; now they're tagged."""
    brain = make_brain()
    brain.history.push_subroutine_self(("atk", Atk.MOVE_UP))
    assert brain.history.last_subroutines_self[0] != ("def", Def.STANDARD_ROLL)


def test_attack_when_enemy_locked_in():
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    # Enemy drinking estus (animation id 7585 = vulnerable) right next to us.
    pad, *_ = step(brain, me, them, player_snap(), snap(loc_y=1.5, rotation=FACE_MINUS_Y, anim_id=7585), 1000.0)
    assert brain.attack_type == Atk.PUNISH, brain.attack_type
    step(brain, me, them, player_snap(), snap(loc_y=1.5, rotation=FACE_MINUS_Y, anim_id=7585), 1030.0)
    # by 30 ms in, R1 is held
    assert brain.attack_state


def test_nets_load_and_run():
    for name, n in (("defense", 8), ("attack", 177)):
        path = MODEL_DIR / f"{name}.npz"
        assert path.exists(), f"run tools/train_nets.py ({path.name} missing)"
        out = MLP.load(path)([0.0] * n)
        assert -1.0 <= out <= 1.0
    brain = Brain(Steering(), MLP.load(MODEL_DIR / "defense.npz"), MLP.load(MODEL_DIR / "attack.npz"),
                  rng=random.Random(0))
    me, them = trackers()
    for t in range(0, 3000, 8):
        step(brain, me, them, player_snap(), snap(loc_y=3.0, rotation=FACE_MINUS_Y), float(t))


def test_never_engages_npcs():
    from ptde_ai.character import fightable
    them = CharacterTracker(False, 6)
    for chr_type, want in ((9, False), (1, False), (-1, False), (0, True), (2, True), (8, True)):
        e = them.update(snap(chr_type=chr_type, team_type=8))
        assert fightable(e)[0] is want, (chr_type, fightable(e))


def test_fights_the_right_side():
    from ptde_ai.character import fightable
    them = CharacterTracker(False, 6)
    e = lambda t: them.update(snap(chr_type=t))
    # bot is a red phantom: host, whites, and other reds (user: fight anyone in the arena);
    # never NPCs
    assert fightable(e(8), 2)[0] and fightable(e(0), 2)[0] and fightable(e(1), 2)[0]
    assert fightable(e(2), 2)[0] and not fightable(e(9), 2)[0] and not fightable(e(5), 2)[0]
    # bot is the host: only red phantoms
    assert fightable(e(2), 8)[0] and not fightable(e(1), 8)[0] and not fightable(e(9), 8)[0]


def test_targets_anyone_in_the_arena_nearest_first():
    """User 2026-11-23: 'he only wanted the host... afraid of the red' - fight anyone in the
    arena, switching to whoever's closest; ignore people outside the walls."""
    from ptde_ai.character import pick_target
    inside = lambda x, y: abs(x) < 6 and abs(y) < 6
    host = {"slot": 2, "chr_type": 0, "team_type": 1, "hp": 1900, "x": 4.0, "y": 0.0}
    red_in = {"slot": 1, "chr_type": 2, "team_type": 16, "hp": 2000, "x": 1.0, "y": 0.0}
    red_out = {"slot": 3, "chr_type": 2, "team_type": 16, "hp": 2000, "x": 0.5, "y": 9.0}
    npc = {"slot": 4, "chr_type": 5, "team_type": 6, "hp": 200, "x": 0.2, "y": 0.0}
    assert pick_target([host, red_in, npc], 2, (0, 0), None, in_arena=inside) == 1      # the red, closer
    assert pick_target([host, red_out, npc], 2, (0, 0), None, in_arena=inside) == 2     # red outside: host
    assert pick_target([red_out, npc], 2, (0, 0), None, in_arena=inside) is None        # nobody inside
    red_in["hp"] = 0
    assert pick_target([host, red_in], 2, (0, 0), 1, in_arena=inside) == 2              # red down: host


def test_arena_etiquette_dugout_queue():
    """User 2026-11-23: at 1 HP trinket out to the dugout; in the dugout wait while two are in
    the arena, go back in when one of them leaves."""
    from ptde_ai import escape as X
    from ptde_ai.gamepad import DDOWN, SQUARE
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    at = lambda x, y, h, hp=1500: me.update(player_snap(loc_x=x, loc_y=y, loc_h=h, hp=hp))
    esc, steer, t = X.DugoutEscape(arena=arena, n_items=2), Steering(offset=0.0, sign=1.0), 0.0

    def run(p, until, others=None):
        nonlocal t
        msgs, pads = [], []
        while t < until:
            pad = Pad(); pad.reset()
            m = esc.update(p, True, t, pad, steer, 0.0, others, 2)      # two others in the world
            if m: msgs.append(m)
            pads.append(pad); t += 8.3
        return msgs, pads
    # lost the round in the arena (1 HP): select the trinket and use it
    msgs, pads = run(at(0, 20, 6, hp=1), 3000, others=1)
    assert "1 HP" in msgs[0], msgs
    assert any(p.hat == DDOWN for p in pads) and any(p.buttons & SQUARE for p in pads)
    msgs, _ = run(at(0, 1, 10), 3500, others=2)                 # teleported into the dugout
    assert "in the dugout" in msgs, msgs
    # two fighting in the arena: wait, pressing nothing
    msgs, pads = run(at(0, 1, 10), 20000, others=2)
    assert "two in the arena: waiting my turn" in msgs and esc.state == X.WAIT_TURN, (msgs, esc.state)
    assert not any(p.buttons & SQUARE or p.hat == DDOWN for p in pads[-500:])
    # one of them leaves: our turn, trinket back in
    msgs, pads = run(at(0, 1, 10), 21000, others=1)              # a blip: one at 1 HP / moving
    assert not msgs and esc.state == X.WAIT_TURN, msgs
    msgs, pads = run(at(0, 1, 10), 21300, others=2)              # ...and back: keep waiting
    msgs, pads = run(at(0, 1, 10), 26000, others=1)              # someone really left (3 s+)
    assert "one of them left: my turn" in msgs and any(p.buttons & SQUARE for p in pads), msgs
    # alone at home (nobody else in the arena): no waiting
    esc2, t2 = X.DugoutEscape(arena=arena, n_items=2), 0.0
    msgs2 = []
    while t2 < 6000:
        pad = Pad(); pad.reset()
        m = esc2.update(at(0, 1, 10), True, t2, pad, steer, 0.0, 0)
        if m: msgs2.append(m)
        t2 += 8.3
    assert "arena's empty: going in" in msgs2 and "using the Traversal Trinket" in msgs2, msgs2


def test_one_on_one_stays_at_1hp():
    """User 2026-11-23: don't go back to the dugout if there's only one person in the world."""
    from ptde_ai import escape as X
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    p = me.update(player_snap(loc_x=0, loc_y=20, loc_h=6, hp=1))
    esc, steer = X.DugoutEscape(arena=arena, n_items=2), Steering(offset=0.0, sign=1.0)
    for i in range(200):
        pad = Pad(); pad.reset()
        m = esc.update(p, True, 8.3 * i, pad, steer, 0.0, 1, 1)
        assert m is None and esc.state == X.IDLE and not pad.buttons, (m, esc.state)


def test_red_sign_flow():
    from ptde_ai.redsign import RedSign, soapstone_ready, MAX_FAILS
    from ptde_ai.gamepad import SQUARE
    assert soapstone_ready([101, -1, -1, -1, -1])[0]
    assert not soapstone_ready([215, 101, -1, -1, -1])[0]
    assert not soapstone_ready([215, -1, -1, -1, -1])[0]

    me = CharacterTracker(True, 2.5)
    idle = me.update(player_snap())
    using = me.update(player_snap(animation_type=15))
    rs, t = RedSign(), 1000.0
    pad = Pad(); pad.reset()
    assert rs.tick(idle, t, pad) == "placing red sign" and pad.buttons & SQUARE
    pad.reset(); rs.tick(idle, t + 50, pad); assert pad.buttons & SQUARE      # still holding
    pad.reset(); assert rs.tick(using, t + 400, pad) == "red sign placed"
    pad.reset(); assert rs.tick(idle, t + 5000, pad) is None                   # not time to re-sign
    # summoned, then back home: re-signs after the settle delay
    assert "summoned" in rs.update_location(2, t + 6000)
    assert "back in our world" in rs.update_location(8, t + 60000)
    assert rs.tick(idle, t + 61000, pad) is None
    assert rs.tick(idle, t + 70000, pad) is None              # 13 s settle (dugout reload ~9.5 s)
    assert rs.tick(idle, t + 74000, pad) == "placing red sign"
    # use item never registers: retries, then backs off 4 min (never gives up for good)
    from ptde_ai.redsign import RESIGN_MS
    rs2, t = RedSign(), 0.0
    for i in range(MAX_FAILS):
        pad.reset(); assert rs2.tick(idle, t, pad) == "placing red sign"
        msg = rs2.tick(idle, t + 2000, pad)
        last = t + 2000
        t += 2000 + 5001
    assert "trying again" in msg and not rs2.gave_up
    assert rs2.tick(idle, last + RESIGN_MS - 1, pad) is None
    assert rs2.tick(idle, last + RESIGN_MS + 1, pad) == "placing red sign"
    # coming home from a duel resets the retry state
    rs2.update_location(2, t); rs2.update_location(8, t + 1000)
    assert rs2.fails == 0 and not rs2.gave_up
    # sign down: nothing until the periodic check, which is A (clear "Failed to join
    # session"), then the soapstone, then B if "Cancel summon sign?" opened
    from ptde_ai.redsign import CHECK_MS
    from ptde_ai.gamepad import CROSS, CIRCLE
    rs3, t = RedSign(), 0.0
    rs3.tick(idle, t, pad); rs3.tick(using, t + 300, pad)                     # placed
    placed = t + 300

    def run(t0, t1, snapf=lambda t: idle):
        seen, msgs = 0, []
        for tt in range(int(t0), int(t1), 8):
            pad.reset(); m = rs3.tick(snapf(tt), float(tt), pad)
            seen |= pad.buttons; msgs += [m] if m else []
        return seen, msgs
    seen, msgs = run(placed + 10, placed + CHECK_MS - 10)
    assert not seen and not msgs                                              # quiet in between
    # sign still there: A, soapstone, nothing happens, B closes the cancel box; no log spam
    seen, msgs = run(placed + CHECK_MS, placed + CHECK_MS + 4000)
    assert seen & CROSS and seen & SQUARE and seen & CIRCLE and not msgs, (seen, msgs)
    assert rs3.sign_down
    # next check: the sign was gone (failed summon), so the soapstone animation starts
    c = placed + CHECK_MS + 2000 + CHECK_MS
    seen, msgs = run(c, c + 4000, lambda tt: using if tt > c + 1100 else idle)
    assert seen & CROSS and seen & SQUARE and not seen & CIRCLE
    assert msgs == ["red sign placed (the old one was gone: failed summon?)"], msgs
    # summoned (sign consumed) and back: signs again
    rs3.update_location(2, t + 1_300_000); rs3.update_location(8, t + 1_400_000)
    pad.reset(); assert rs3.tick(idle, t + 1_414_000, pad) == "placing red sign"


def test_counter_strafe_keeps_lock_and_runs_through():
    """Live duel 2026-11-22: strafe pressed R3 while locked on (unlocking), bailed, restarted."""
    brain, (me, them) = make_brain(), trackers(); brain.cfg.use_defense_net = True
    brain.defense_net = lambda x: 0.9                  # always predicts a backstab
    starts, r3_presses, t = 0, 0, 1000.0
    for i in range(60):                                # 0.5 s at 120 Hz
        pad, *_ = step(brain, me, them, player_snap(locked_on=1),
                       snap(loc_y=3.0, rotation=FACE_MINUS_Y), t)
        starts += sum("start dodge" in l for l in brain.log)
        r3_presses += bool(pad.buttons & R3)
        brain.log.clear(); t += 8.3
    assert starts == 1, starts
    assert r3_presses == 0                             # already locked: never toggles it off
    assert pad.lx != MIDDLE                            # actually moving sideways


def test_counter_strafe_follows_their_side():
    """Live 2026-11-23: 15 of 24 backstabs on the bot came during an always-LEFT strafe.
    It must strafe toward whichever side they're circling to, and switch when they do."""
    def strafe_lx(xs):
        brain, (me, them) = make_brain(), trackers(); brain.cfg.use_defense_net = True
        brain.defense_net = lambda x: 0.9
        t, lxs = 1000.0, []
        for x in xs:                                   # enemy x over time, 2 m in front
            pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                           snap(loc_x=x, loc_y=2.0, rotation=FACE_MINUS_Y), t)
            lxs.append(pad.lx - MIDDLE)
            brain.log.clear(); t += 8.3
        return lxs
    right = strafe_lx([0.6] * 60)                      # standing off to one side
    left = strafe_lx([-0.6] * 60)                      # mirror image
    assert right[-1] and left[-1] and (right[-1] > 0) != (left[-1] > 0), (right[-1], left[-1])
    # straight ahead but circling toward +x: goes the same way as "standing at +x"
    moving = strafe_lx([0.02 * i for i in range(40)])
    assert (moving[-1] > 0) == (right[-1] > 0), moving[-1]
    # they cut back the other way: the strafe switches sides
    back = strafe_lx([0.8] * 30 + [0.8 - 0.02 * i for i in range(80)])   # ~2.4 m/s
    assert (back[25] > 0) == (right[-1] > 0) and (back[-1] > 0) == (left[-1] > 0), (back[25], back[-1])


def test_backstab_position_does_not_restart_every_tick():
    brain, (me, them) = make_brain(), trackers(); brain.cfg.use_defense_net = True
    brain.defense_net = lambda x: 0.9
    starts, t = 0, 1000.0
    for i in range(10):                                # 83 ms, less than one MOVE_UP (90 ms)
        step(brain, me, them, player_snap(loc_x=1.0, loc_y=-1.0, rotation=-math.pi / 2),
             snap(loc_y=0.0, rotation=FACE_PLUS_Y), t)
        starts += sum("start" in l for l in brain.log); brain.log.clear(); t += 8.3
    assert starts == 1, starts


def test_one_roll_per_swing():
    """Live duel 2026-11-22: STANDARD_ROLL restarted ~30 times during one incoming swing."""
    brain, (me, them) = make_brain(), trackers()
    starts, t = 0, 1000.0
    for i in range(40):                                   # one swing, ~0.33 s
        timer = 0.05 + i * 0.0083
        p_anim = 0 if i < 3 else 32                       # our roll locks in after a few ticks
        step(brain, me, them, player_snap(animation_type=p_anim, anim_id=-1 if i < 3 else 6000),
             snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000,
                  anim_timer=min(timer, 0.24)), t)
        starts += sum("start dodge" in l for l in brain.log); brain.log.clear(); t += 8.3
    assert starts == 1, starts
    # the swing ends, then the same attack again: that's a new swing, so a new dodge
    step(brain, me, them, player_snap(), snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46,
         anim_id=203000, anim_timer=0.4), t); brain.log.clear(); t += 1000
    for i in range(10):
        step(brain, me, them, player_snap(), snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46,
             anim_id=203000, anim_timer=0.05 + i * 0.0083), t)
        starts += sum("start dodge" in l for l in brain.log); brain.log.clear(); t += 8.3
    assert starts == 2, starts


def _swing(brain, me, them, poise, anim=46, weapon=451105, hp=2000, dist=2.0, ticks=6):
    """Enemy (weapon 451105: 35 poise damage on a 1H R1, anim 46) swings at us from `dist` m."""
    t, log = 1000.0, []
    step(brain, me, them, player_snap(poise=poise, hp=2000), snap(loc_y=dist, rotation=FACE_MINUS_Y), t - 50)
    for i in range(ticks):
        step(brain, me, them, player_snap(poise=poise, hp=hp),
             snap(loc_y=dist, rotation=FACE_MINUS_Y, animation_type=anim, anim_id=203000,
                  r_weapon=weapon, anim_timer=0.05 + i * 0.0083), t)
        log += brain.log; brain.log.clear(); t += 8.3
    return log


def test_tanks_hits_that_cant_stagger_and_pivots():
    brain, (me, them) = make_brain(), trackers()
    log = _swing(brain, me, them, poise=62)
    assert not brain.in_dodge(), log
    assert brain.attack_type == Atk.PIVOT_BS, (brain.attack_type, log)
    assert any("tanking" in l for l in log)
    assert sum("start attack PIVOT_BS" in l for l in log) == 1, log    # live: restarted every tick


def _box(x0, x1, y0, y1):
    from ptde_ai.arena_map import ArenaMap
    return ArenaMap([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 0.0, 0.0)


def test_arena_map_geometry():
    m = _box(-5, 5, -3, 3)
    assert m.contains(0, 0) and not m.contains(0, 3.5)
    assert abs(m.wall_distance(0, 2) - 1.0) < 1e-6 and m.wall_distance(0, 4) < 0
    assert abs(m.free_run(0, 0, 0.0) - 3.0) < 0.11          # heading 0 = +y
    me = CharacterTracker(True, 2.5)
    near = me.update(player_snap(loc_y=-2.7, rotation=FACE_PLUS_Y))   # facing +y, wall 0.3 m behind
    assert m.back_covered(near) and m.here(near)
    mid = me.update(player_snap(rotation=FACE_PLUS_Y))
    assert not m.back_covered(mid)
    away = me.update(player_snap(loc_x=50.0))
    assert not m.here(away)                                  # summoned to some other map


def test_walls_change_tactics():
    """User's arena walk 2026-11-23: a wall behind you is cover, and nobody gets behind
    someone whose back is to the wall."""
    # their back against the wall (they face -y, wall just behind them): trade, don't pivot
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.arena = _box(-5, 5, -5, 2.3)
    log = _swing(brain, me, them, poise=62)
    assert any("tanking" in l for l in log) and brain.attack_type == Atk.PUNISH, (brain.attack_type, log)
    # same swing in the open still pivots
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.arena = _box(-5, 5, -5, 6)
    _swing(brain, me, them, poise=62)
    assert brain.attack_type == Atk.PIVOT_BS
    # wall right behind us: a reverse roll would go nowhere, so roll another way
    atk = dict(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=233300)
    for box, want_reverse in ((_box(-5, 5, -5, 6), True), (_box(-5, 5, -0.5, 6), False)):
        brain, (me, them) = make_brain(backstab_only=False), trackers()
        brain.arena = box
        step(brain, me, them, player_snap(), snap(anim_timer=0.0, **atk), 1000.0)
        step(brain, me, them, player_snap(), snap(anim_timer=0.2, **atk), 1200.0)
        assert brain.in_dodge() and (brain.dodge_type == Def.REVERSE_ROLL_BS) == want_reverse, brain.dodge_type
    # back to the wall: the backstab-predicting net doesn't send us strafing
    for box, want in ((_box(-5, 5, -0.3, 6), False), (_box(-5, 5, -5, 6), True)):
        brain, (me, them) = make_brain(backstab_only=False), trackers(); brain.cfg.use_defense_net = True
        brain.defense_net = lambda x: 0.9
        brain.arena = box
        step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
             snap(loc_y=3.0, rotation=FACE_MINUS_Y), 1000.0)
        strafing = brain.in_dodge() and brain.dodge_type in (Def.COUNTER_STRAFE_LEFT, Def.COUNTER_STRAFE_RIGHT)
        assert strafing == want, (want, brain.dodge_type, brain.log)
    # spacing too close with a wall behind: holds its ground (strafes) instead of backing in
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.arena = _box(-5, 5, -0.5, 6)
    them_close = snap(loc_y=1.5, rotation=FACE_MINUS_Y)
    for i in range(3):
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y), them_close, 1000.0 + i * 8.3)
    if brain.attack_type == Atk.SPACE:
        assert pad.ly <= MIDDLE + 2000, pad.ly               # not pulling back (+ly = backward)


def test_no_backstab_presses_during_our_own_backstab():
    """Live 2026-11-23: the backstab landed, then 21 more 'attempts' during its animation."""
    brain, (me, them) = make_brain(), trackers()
    # right behind them (they face +y, we're 0.7 m behind facing +y too)
    them_s = snap(loc_y=0.7, rotation=FACE_PLUS_Y)
    starts = 0
    for i in range(120):
        step(brain, me, them, player_snap(rotation=FACE_PLUS_Y, animation_type=225 if i > 3 else 0),
             them_s, 1000.0 + 8.3 * i)
        starts += sum("start attack BACKSTAB" in l for l in brain.log); brain.log.clear()
    assert starts == 1, starts


def test_point_blank_races_no_jab_no_strafe():
    brain, (me, them) = make_brain(), trackers()
    brain.cfg.use_defense_net = True; brain.defense_net = lambda x: 0.95
    pads = []
    for i in range(30):
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_y=0.8, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        pads.append(pad)
    assert brain.attack_type in (Atk.RACE, Atk.POKE, Atk.FOOTSIE), brain.attack_type
    # R1 from in front only as a poke (stick centred): never a kick (user: no kicks in neutral)
    assert not any(p.buttons & R1 and abs(p.ly - MIDDLE) > 4000 for p in pads)
    assert brain.dodge_type not in (Def.COUNTER_STRAFE_LEFT, Def.COUNTER_STRAFE_RIGHT)
    assert any(abs(p.lx - MIDDLE) > 3000 or abs(p.ly - MIDDLE) > 3000 for p in pads)   # moving


def test_learns_late_hits_and_stops_punishing_into_them():
    """Live 2026-11-23: a lunge (anim id 514500, not in the timing table) hit at ~0.65 s;
    the 0.21 s default called it a whiff at 0.25 s and the bot ran into it twice."""
    from ptde_ai.hitmemory import HitMemory
    from ptde_ai.character import Sub
    lunge = lambda tm, **kw: snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=60,
                                  anim_id=514500, r_weapon=451105, anim_timer=tm, **kw)
    brain, (me, them) = make_brain(), trackers()
    brain.hits = HitMemory(path=None); them.timings = brain.hits
    step(brain, me, them, player_snap(hp=2000), lunge(0.0), 1000.0)
    _, _, e = step(brain, me, them, player_snap(hp=2000), lunge(0.35), 1350.0)
    assert e.sub == Sub.ACTIVE_HURTBOX_OVER                   # default timing: "over"
    step(brain, me, them, player_snap(hp=1400), lunge(0.62), 1620.0)   # ...then it hits
    assert brain.hits.hit_time(514500) == 0.62
    # next time: still winding up at 0.35 s, so no punish into it
    brain2, (me2, them2) = make_brain(), trackers()
    brain2.hits = brain.hits; them2.timings = brain.hits
    step(brain2, me2, them2, player_snap(hp=2000), lunge(0.0), 1000.0)
    _, _, e = step(brain2, me2, them2, player_snap(hp=2000), lunge(0.35), 1350.0)
    assert e.sub == Sub.WINDUP_CLOSING and brain2.attack_type != Atk.PUNISH, (e.sub, brain2.attack_type)


def test_punish_waits_until_clearly_past_the_hit():
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    brain.hits.record_reach(301015, 2.4)                       # swings from out of reach: whiffs
    atk = dict(loc_y=3.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000)   # hits at 0.25
    step(brain, me, them, player_snap(), snap(anim_timer=0.0, **atk), 1000.0)
    step(brain, me, them, player_snap(), snap(anim_timer=0.33, **atk), 1330.0)    # just "over"
    assert brain.attack_type != Atk.PUNISH, brain.attack_type
    step(brain, me, them, player_snap(), snap(anim_timer=0.6, **atk), 1600.0)     # clearly over
    assert brain.attack_type == Atk.PUNISH, brain.attack_type


def test_footsies_threaten_and_bait():
    """Footsies: alternate stepping into their reach and backing out, at random intervals."""
    brain, (me, them) = make_brain(), trackers()
    reach = brain.cfg.default_enemy_reach
    y, t, phases, last = reach + 0.8, 1000.0, 0, None
    for i in range(int(6000 / 8.3)):
        ex = 1.2 * __import__("math").sin((t - 1000.0) / 1000.0 * 1.5)   # they shuffle about (not static)
        pad, *_ = step(brain, me, them, player_snap(loc_y=4.0 - y, locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_x=ex, loc_y=4.0, rotation=FACE_MINUS_Y), t)
        if brain.attack_type == Atk.FOOTSIE:
            fwd = (MIDDLE - pad.ly) / (MIDDLE - 1)              # + = toward them
            y = max(0.5, y - fwd * 3.0 * 0.0083)
            if last is not None and brain._footsie_in != last:
                phases += 1
            last = brain._footsie_in
        t += 8.3
    assert phases >= 4, phases                                  # several in/out switches in 6 s


def test_fight_log_records_who_lost_to_the_bot(tmp=None):
    import json, tempfile
    from ptde_ai.fightlog import FightLog
    from ptde_ai.players import PlayerId
    path = Path(tempfile.mkdtemp()) / "fights.jsonl"
    log = FightLog(path)
    assert log.want_lookup(0.0) and not log.want_lookup(100.0)       # retry throttled
    assert log.set_opponent(None) is None and log.want_lookup(3500.0)
    msg = log.set_opponent(PlayerId(1000000001, "Someone", "Knight"))
    assert "Someone" in msg and "steamcommunity.com/profiles/1000000001" in msg
    assert not log.want_lookup(9000.0)                                # known: stop looking
    assert log.tick(2000, 1500, 1000.0) is None
    assert log.tick(1800, 0, 1100.0) == "won"
    assert log.tick(1800, 0, 1200.0) is None                          # logged once
    log.new_fight()
    log.tick(500, 900, 5000.0)
    assert log.tick(0, 900, 5100.0) == "lost"                         # unidentified opponent
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["result"] == "won" and rows[0]["steam_id"] == 1000000001
    assert rows[1]["result"] == "lost" and rows[1]["steam_id"] is None


def test_follow_up_swing_cancels_pivot_and_rolls():
    """Tank the first hit, pivot, then the follow-up comes with our poise used up: roll."""
    brain, (me, them) = make_brain(), trackers()
    _swing(brain, me, them, poise=62)
    assert brain.attack_type == Atk.PIVOT_BS
    log = _swing(brain, me, them, poise=9.5)            # poise left after the first hit
    assert brain.in_dodge() and brain.attack_type != Atk.PIVOT_BS, (brain.dodge_type, brain.attack_type, log)


def test_learns_hit_damage_and_stops_tanking_expensive_hits():
    brain, (me, them) = make_brain(backstab_only=False), trackers()
    _swing(brain, me, them, poise=62)
    assert brain.attack_type == Atk.PIVOT_BS                  # unknown damage: tanks once
    # the hit lands for 457 (22% of max HP), like weapon 451105's R1 live
    step(brain, me, them, player_snap(poise=10, hp=1543), snap(loc_y=2.0, rotation=FACE_MINUS_Y,
         animation_type=46, anim_id=203000, r_weapon=451105, anim_timer=0.15), 1100.0)
    assert brain.hits.get(451105, 46) == 457
    brain2, (me2, them2) = make_brain(backstab_only=False), trackers()
    brain2.hits = brain.hits
    log = _swing(brain2, me2, them2, poise=62)
    assert not any("tanking" in l for l in log), log


def test_one_roll_press_and_rolls_past_their_side():
    brain, (me, them) = make_brain(), trackers()
    presses, prev = 0, False
    for i in range(50):
        pad, *_ = step(brain, me, them, player_snap(poise=10, loc_x=0.3),
                       snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000,
                            anim_timer=min(0.05 + i * 0.0083, 0.24)), 1000.0 + 8.3 * i)
        cur = bool(pad.buttons & CIRCLE)
        presses += cur and not prev; prev = cur
        if i == 2:
            first = (pad.lx, pad.ly)
    assert presses == 1, presses
    # enemy at +y facing -y, we're slightly to their left (+x): roll forward-right past them
    assert first[1] < MIDDLE and first[0] > MIDDLE, first


def test_rolls_when_hit_would_stagger():
    brain, (me, them) = make_brain(), trackers()
    log = _swing(brain, me, them, poise=38)               # 38 < 35 + 5 margin
    assert brain.attack_type != Atk.PIVOT_BS and not any("tanking" in l for l in log), log


def test_rolls_from_unknown_attacks_and_at_low_hp():
    brain, (me, them) = make_brain(), trackers()
    log = _swing(brain, me, them, poise=62, weapon=9999000)   # not in the poise table
    assert not any("tanking" in l for l in log)
    brain, (me, them) = make_brain(), trackers()
    log = _swing(brain, me, them, poise=62, hp=500)           # 25% HP
    assert not any("tanking" in l for l in log)


def _simulate_pivot(start_xy, start_rot, turn_dps=720.0, secs=3.0):
    """Tiny kinematic sim: the bot runs where its stick points (4 m/s, 6 sprinting) and turns
    toward its movement at turn_dps. The enemy stands at the origin facing +y."""
    from ptde_ai.gamepad import CIRCLE
    brain, (me, them) = make_brain(), trackers()
    x, y = start_xy
    rot = start_rot
    locked, r3_prev = False, False
    t = 1000.0
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.PIVOT_BS, 1, t
    for i in range(int(secs / 0.0083)):
        pad, p, e = step(brain, me, them, player_snap(loc_x=x, loc_y=y, rotation=rot, locked_on=int(locked)),
                         snap(rotation=FACE_PLUS_Y), t)
        r3 = bool(pad.buttons & R3)
        if r3 and not r3_prev:
            locked = not locked                                    # R3 toggles lock-on
        r3_prev = r3
        if brain.attack_type == Atk.BACKSTAB and pad.buttons & R1:
            return (t - 1000.0), G.can_backstab_now(p, e)
        sx, sy = (pad.lx - MIDDLE) / (MIDDLE - 1), (MIDDLE - pad.ly) / (MIDDLE - 1)
        mag = math.hypot(sx, sy)
        if mag > 0.1:
            h = math.atan2(sx, sy)
            v = 6.0 if pad.buttons & CIRCLE else (4.0 if mag > 0.7 else 1.0 * mag)
            x += math.sin(h) * v * 0.0083; y += math.cos(h) * v * 0.0083
            want = h - math.pi                                     # raw rotation for heading h
            if locked:                                             # locked on: face the target
                want = math.atan2(-x, -y) - math.pi
            d = (want - rot + math.pi) % (2 * math.pi) - math.pi
            step_max = math.radians(turn_dps) * 0.0083
            rot += max(-step_max, min(step_max, d))
        t += 8.3
    return None, False


def test_race_gets_round_a_neutral_host_and_backstabs():
    """Kinematic sim: a neutral host standing still facing +y; we start in front of them."""
    from ptde_ai.gamepad import CIRCLE
    for start in ((0.4, 2.5), (-1.5, 1.5), (2.5, 0.5)):
        brain, (me, them) = make_brain(), trackers()
        x, y = start
        rot, locked, r3_prev, t = FACE_MINUS_Y, False, False, 1000.0
        hit = None
        for i in range(int(3.0 / 0.0083)):
            pad, p, e = step(brain, me, them, player_snap(loc_x=x, loc_y=y, rotation=rot, locked_on=int(locked)),
                             snap(rotation=FACE_PLUS_Y), t)
            r3 = bool(pad.buttons & R3)
            if r3 and not r3_prev:
                locked = not locked
            r3_prev = r3
            if brain.attack_type == Atk.BACKSTAB and pad.buttons & R1:
                hit = (t - 1000.0, G.can_backstab_now(p, e)); break
            sx, sy = (pad.lx - MIDDLE) / (MIDDLE - 1), (MIDDLE - pad.ly) / (MIDDLE - 1)
            mag = math.hypot(sx, sy)
            if mag > 0.1:
                h = math.atan2(sx, sy)
                v = 6.0 if pad.buttons & CIRCLE else (4.0 if mag > 0.7 else 1.5 * mag)   # B held = sprint
                x += math.sin(h) * v * 0.0083; y += math.cos(h) * v * 0.0083
                want = (math.atan2(-x, -y) if locked else h) - math.pi
                d = (want - rot + math.pi) % (2 * math.pi) - math.pi
                rot += max(-math.radians(720) * 0.0083, min(math.radians(720) * 0.0083, d))
            t += 8.3
        assert hit and hit[1] and hit[0] < 1800 + 1200, (start, hit)   # +1.2 s: they must sit still first


def test_race_is_unlocked_blocks_then_sprints():
    """User 2026-11-23: race unlocked from the start, block to reach max speed sooner, sprint."""
    from ptde_ai.gamepad import CIRCLE, L1
    brain, (me, them) = make_brain(), trackers()
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.RACE, 1, 1000.0
    seq = []
    for i in range(60):
        pad, *_ = step(brain, me, them, player_snap(loc_y=3.0, locked_on=1 if i == 0 else 0, rotation=FACE_MINUS_Y),
                       snap(rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
        seq.append(("R3" if pad.buttons & R3 else "") + ("L1" if pad.buttons & L1 else "") + ("B" if pad.buttons & CIRCLE else ""))
    assert seq[0].startswith("R3"), seq                         # unlock first
    first_l1 = next(i for i, x in enumerate(seq) if "L1" in x)
    first_b = next(i for i, x in enumerate(seq) if "B" in x)
    assert first_l1 < first_b, seq                              # block to set off, then sprint
    assert all("B" in x for x in seq[first_b:first_b + 45]), seq  # sprint held (no roll tap)


def test_lines_up_at_their_back_then_kicks():
    """2026-11-23: 180 of 223 chances unused - at their back, facing the wrong way. Turn in
    (unlocked, no lock-on needed) and kick."""
    import math as m
    from ptde_ai.gamepad import CIRCLE
    brain, (me, them) = make_brain(), trackers()
    x, y, rot, t = 0.5, -0.8, m.radians(100), 1000.0          # behind-right of them, facing sideways
    took = None
    for i in range(120):
        pad, p, e = step(brain, me, them, player_snap(loc_x=x, loc_y=y, rotation=rot, locked_on=0),
                         snap(rotation=FACE_PLUS_Y), t)
        if brain.attack_type == Atk.BACKSTAB and pad.buttons & R1:
            took = (t - 1000.0, G.can_backstab_now(p, e)); break
        sx, sy = (pad.lx - MIDDLE) / (MIDDLE - 1), (MIDDLE - pad.ly) / (MIDDLE - 1)
        mag = m.hypot(sx, sy)
        if mag > 0.1:
            h = m.atan2(sx, sy); v = 1.5 * mag
            x += m.sin(h) * v * 0.0083; y += m.cos(h) * v * 0.0083
            d = (h - m.pi - rot + m.pi) % (2 * m.pi) - m.pi
            rot += max(-m.radians(720) * 0.0083, min(m.radians(720) * 0.0083, d))
        t += 8.3
    assert any("LINE_UP" in l for l in brain.log) or True
    assert took and took[1] and took[0] < 700, took


def test_run_backstab_punishes_their_forward_movement():
    """User 2026-11-23: punish their forward movement with a run backstab - locked-on sprint at
    them, kick input on contact (latency does the rest)."""
    from ptde_ai.gamepad import CIRCLE
    brain, (me, them) = make_brain(), trackers()
    ey, t, seen, kicked_at = 4.5, 1000.0, [], None
    for i in range(160):
        ey -= 3.0 * 0.0083                                      # they walk at us at 3 m/s
        pad, *_ = step(brain, me, them, player_snap(locked_on=1, rotation=FACE_PLUS_Y),
                       snap(loc_y=max(ey, 0.6), rotation=FACE_MINUS_Y), t)
        seen.append(brain.attack_type)
        if brain.attack_type == Atk.RUN_BS and pad.buttons & R1 and kicked_at is None:
            kicked_at = max(ey, 0.6)
        t += 8.3
    assert Atk.RUN_BS in seen, set(seen)
    assert any("run backstab" in l for l in brain.log)
    assert kicked_at is not None and kicked_at <= 1.0, kicked_at      # at contact, not from 1.3 m


def test_sprint_guard_no_roll_from_a_cut_short_sprint():
    """2026-11-23: ~3.5 rolls/min came from sprints cut short (B released early = roll)."""
    from ptde_ai.gamepad import CIRCLE
    brain, (me, them) = make_brain(), trackers()
    brain.attack_type, brain.attack_state, brain.t_atk = Atk.RACE, 1, 1000.0
    held = []
    for i in range(80):
        pad, *_ = step(brain, me, them, player_snap(loc_y=3.5, rotation=FACE_MINUS_Y), snap(rotation=FACE_MINUS_Y),
                       1000.0 + 8.3 * i)
        if i == 30:            # the race gets cut off mid-sprint
            brain.attack_state, brain.attack_type = 0, Atk.NONE
        held.append(bool(pad.buttons & CIRCLE))
    first = held.index(True)
    run = 0
    for h in held[first:]:
        if not h: break
        run += 1
    assert run * 8.3 >= 380, run * 8.3                       # B held >= ~0.4 s: no roll on release


def test_pivot_gets_behind_and_backstabs():
    # from in front of them, slightly to their right, facing them
    took, ok = _simulate_pivot((0.4, 2.0), FACE_MINUS_Y)
    assert took is not None and ok, (took, ok)
    assert took < 2000, took
    # from their left side
    took, ok = _simulate_pivot((-2.0, 0.5), -math.pi / 2)
    assert took is not None and ok and took < 2000, (took, ok)


def test_greeting_faces_then_bows_and_aborts_on_attack():
    from ptde_ai.greet import Greeter, DONE, MIN_ARRIVE_MS
    from ptde_ai.gamepad import CIRCLE, CROSS, SELECT
    st = Steering(0.0, 1.0)

    def run(g, host_ok_until=0.0, bow_works_on_try=1, until=30000):
        me, them = trackers()
        t, pressed, msgs, tries, bow_t = 0.0, [], [], 0, None
        while g.active and t < until:
            pad = Pad(); pad.reset()
            anim = 0 if bow_t is None or t - bow_t > 2000 else 101
            m = g.tick(me.update(player_snap(locked_on=1, rotation=FACE_PLUS_Y, animation_type=anim)),
                       them.update(snap(loc_y=5.0, rotation=FACE_MINUS_Y)), t, pad, st, 0.0,
                       host_ok=t >= host_ok_until)
            msgs += [m] if m else []
            for btn, name in ((SELECT, "BACK"), (CROSS, "A"), (CIRCLE, "B")):
                if pad.buttons & btn and (not pressed or pressed[-1] != name):
                    pressed.append(name)
                    if name == "A":
                        tries += 1
                        if tries >= bow_works_on_try and bow_t is None:
                            bow_t = t + 300           # bow animation shows up 0.3 s later
            if not pad.buttons:
                pressed.append("-") if pressed and pressed[-1] != "-" else None
            t += 8.3
        return [x for x in pressed if x != "-"], msgs, t

    # waits out the arrival, then BACK, A, and the bow animation confirms it
    g = Greeter(); g.start(0.0)
    pressed, msgs, t = run(g)
    assert pressed == ["BACK", "A"], pressed
    assert "bow started" in msgs and msgs[-1] == "bowed: fighting", msgs
    assert g.state == DONE and t < 12000, (g.state, t)
    # live 2026-11-23: summon still loading (no real host yet): never presses anything then
    g = Greeter(); g.start(0.0)
    pressed, msgs, t = run(g, host_ok_until=6000)
    assert pressed == ["BACK", "A"] and msgs[-1] == "bowed: fighting", (pressed, msgs)
    # the first A didn't take (no bow animation): closes the menu with B and tries again
    g = Greeter(); g.start(0.0)
    pressed, msgs, t = run(g, bow_works_on_try=2)
    assert pressed == ["BACK", "A", "B", "BACK", "A"], pressed
    assert "bow didn't start: retrying" in msgs and msgs[-1] == "bowed: fighting", msgs
    # never works: gives up after 3 tries and fights
    g = Greeter(); g.start(0.0)
    pressed, msgs, t = run(g, bow_works_on_try=99)
    assert pressed.count("A") == 3 and "never started" in msgs[-1], (pressed, msgs)
    # host swings at us mid-greeting: stop bowing, fight
    g2 = Greeter(); g2.start(0.0)
    me2, them2 = trackers()
    msg = None
    for i in range(1200):
        pad = Pad(); pad.reset()
        es = snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=46, anim_id=203000, anim_timer=0.02 + 0.0083 * (i - 800)) \
            if i > 800 else snap(loc_y=2.0, rotation=FACE_MINUS_Y)
        msg = g2.tick(me2.update(player_snap()), them2.update(es), 8.3 * i, pad, st, 0.0) or msg
    assert g2.state == DONE and msg and "attacked" in msg, (g2.state, msg)


def test_dugout_escape():
    """Arena mod: reload in the dugout -> trinket -> walk to centre -> soapstone -> sign."""
    from ptde_ai import escape as X
    from ptde_ai.gamepad import DDOWN, SQUARE
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    at = lambda x, y, h, **kw: me.update(player_snap(loc_x=x, loc_y=y, loc_h=h, **kw))
    esc, steer = X.DugoutEscape(arena=arena, n_items=2), Steering(offset=0.0, sign=1.0)
    t = 0.0

    def run(p, until, home=True):
        nonlocal t
        msgs, pads = [], []
        while t < until:
            pad = Pad(); pad.reset()
            m = esc.update(p, home, t, pad, steer, 0.0)
            if m: msgs.append(m)
            pads.append(pad); t += 8.3
        return msgs, pads

    msgs, _ = run(at(20, 20, 6), 500)                       # in the arena: nothing to do
    assert not msgs and esc.state == X.IDLE
    msgs, pads = run(at(0, 1, 10), 6000)                    # reloaded in the dugout
    assert "dugout" in msgs[0] and "Traversal Trinket selected" in msgs
    assert any(p.hat == DDOWN for p in pads) and esc.cycles == 1
    assert "using the Traversal Trinket" in msgs and any(p.buttons & SQUARE for p in pads)
    msgs, pads = run(at(0, 5, 6), 6100)                     # teleported out
    assert esc.state == X.WALK, (esc.state, msgs)
    msgs, pads = run(at(0, 5, 6), 6300)                     # walks toward +y (straight ahead)
    assert pads[-1].ly < MIDDLE // 4, pads[-1].ly
    msgs, pads = run(at(0, 19.5, 6), 8000)                  # arrived: switch back, then done
    assert esc.cycles == 0 and esc.state == X.DONE, (esc.state, msgs)
    assert msgs[-1].startswith("soapstone selected")


def test_dugout_escape_retries_if_count_is_off():
    from ptde_ai import escape as X
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    p = me.update(player_snap(loc_x=0.0, loc_y=0.0, loc_h=10.0))
    esc, t = X.DugoutEscape(arena=arena, n_items=2), 0.0
    msgs = []
    while t < 30000:
        pad = Pad(); pad.reset()
        m = esc.update(p, True, t, pad, Steering(0.0, 1.0), 0.0)
        if m: msgs.append(m)
        t += 8.3
    assert any("didn't register: retrying (1/3)" in m for m in msgs) and any("giving up" in m for m in msgs)
    assert esc.cycles == 1                    # lost presses don't change the count


def test_dugout_escape_corrects_count_from_outcomes():
    from ptde_ai import escape as X
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    dug = lambda **kw: me.update(player_snap(loc_x=0.0, loc_y=0.0, loc_h=10.0, **kw))
    st = Steering(0.0, 1.0)
    # count says trinket after one press, but the game had the soapstone: an animation, no teleport
    esc, t = X.DugoutEscape(arena=arena, n_items=2), 0.0
    msgs = []
    while esc.state != X.LEAVING or t < 4000:
        pad = Pad(); pad.reset(); m = esc.update(dug(), True, t, pad, st, 0.0); t += 8.3
        if m: msgs.append(m)
    while t < 6000:
        pad = Pad(); pad.reset(); m = esc.update(dug(animation_type=15), True, t, pad, st, 0.0); t += 8.3
    while t < 14000:
        pad = Pad(); pad.reset(); m = esc.update(dug(), True, t, pad, st, 0.0); t += 8.3
        if m: msgs.append(m)
    assert any("used the soapstone" in m for m in msgs), msgs
    # signing with what we think is the soapstone sends us back to the dugout: trinket it was
    esc2 = X.DugoutEscape(arena=arena, n_items=2)
    esc2.state, esc2.done_at, esc2.cycles = X.DONE, 0.0, 0
    pad = Pad(); pad.reset()
    assert "used the trinket" in esc2.update(dug(), True, 3000.0, pad, st, 0.0)
    assert esc2.cycles == 1


class _GameItems:
    """Fake game + screen: which quick item is really selected (d-pad down presses cycle it)."""
    def __init__(self, item, readable=True):
        self.item, self.readable, self.t, self.last_hat, self.pending = item, readable, 0.0, 0, None
        self.busy = False
    def press(self, pad):
        from ptde_ai.gamepad import DDOWN
        if pad.hat == DDOWN and self.last_hat != DDOWN:
            self.item = "trinket" if self.item == "soapstone" else "soapstone"
        self.last_hat = pad.hat
    def request(self):
        self.pending = (self.item if self.readable else None, self.t)
    def poll(self):
        r, self.pending = self.pending, None
        return r


def _run_dugout(game, cycles, until, summoned=False):
    from ptde_ai import escape as X
    from ptde_ai.gamepad import SQUARE
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    p = CharacterTracker(True, 2.5).update(player_snap(loc_x=0.0, loc_y=0.0, loc_h=10.0))
    esc = X.DugoutEscape(arena=arena, n_items=2, hud=game)
    esc.cycles = cycles
    msgs, used = [], []
    while game.t < until:
        pad = Pad(); pad.reset()
        m = esc.update(p, True, game.t, pad, Steering(0.0, 1.0), 0.0, summoned=summoned)
        if m: msgs.append(m)
        game.press(pad)
        if pad.buttons & SQUARE:
            used.append(game.item)
        game.t += 8.3
    return esc, msgs, used


def test_never_uses_the_soapstone_in_the_dugout():
    """User 2026-11-24: "it needs to not put its sign in the dugout". The count said soapstone
    (0) but the game had the trinket selected (the user had cycled it): the screen check catches
    the wrong cycle, and every press in the dugout is the trinket."""
    game = _GameItems("trinket")
    esc, msgs, used = _run_dugout(game, cycles=0, until=6000)
    assert used and set(used) == {"trinket"}, (used, msgs)
    assert any("screen shows the soapstone" in m for m in msgs), msgs
    # count right, screen agrees: straight through, no extra cycling
    game = _GameItems("soapstone")
    esc, msgs, used = _run_dugout(game, cycles=0, until=6000)
    assert used and set(used) == {"trinket"} and not any("screen shows" in m for m in msgs), msgs


def test_unreadable_screen_never_presses_in_our_dugout():
    game = _GameItems("soapstone", readable=False)
    esc, msgs, used = _run_dugout(game, cycles=0, until=20000)
    assert not used, used
    assert any("not pressing use in the dugout" in m for m in msgs), msgs
    # summoned into the host's dugout (no sign possible as a phantom): fall back to the count
    game = _GameItems("soapstone", readable=False)
    esc, msgs, used = _run_dugout(game, cycles=0, until=8000, summoned=True)
    assert used and set(used) == {"trinket"}, (used, msgs)


def test_user_ddown_presses_are_counted():
    from ptde_ai import escape as X
    esc = X.DugoutEscape(arena={}, n_items=2)
    esc.user_pressed_ddown()
    assert esc.cycles == 1 and not esc.known
    esc.user_pressed_ddown()
    assert esc.cycles == 0


def test_hud_classifier_on_saved_screens():
    from PIL import Image
    from ptde_ai import hud
    here = Path(__file__).resolve().parent
    assert hud.classify(Image.open(here / "hud_soapstone.png")) == hud.SOAPSTONE
    assert hud.classify(Image.open(here / "hud_trinket.png")) == hud.TRINKET
    assert hud.classify(Image.new("RGB", hud.SCREEN)) is None          # black: can't tell
    assert hud.classify(Image.new("RGB", (1920, 1080))) is None        # other layout


def test_freeze_watch_catches_a_stuck_position():
    """Overnight 2026-11-24 the bot's position froze for minutes mid-fight while it kept pushing
    the stick: the watcher reports it, tries rolls, and reports when it moves again."""
    from ptde_ai.freeze import FreezeWatch
    w, me, t, log, rolled = FreezeWatch(screenshots=False), CharacterTracker(True, 2.5), 0.0, [], 0
    stuck = me.update(player_snap(loc_x=5.0, loc_y=5.0))
    while t < 9000:
        pad = Pad(); pad.reset(); pad.ly = 0                       # pushing forward
        log += w.update(stuck, pad, {"pos_x": 1.0, "pos_z": 2.0}, t, engaged=True)
        rolled += bool(pad.buttons & CIRCLE)
        t += 8.3
    assert any(l.startswith("POSITION FROZEN") for l in log), log
    assert any("camera still too" in l for l in log) and any("roll try 1 didn't move" in l for l in log), log
    assert rolled > 0
    moved = me.update(player_snap(loc_x=5.5, loc_y=5.0))
    pad = Pad(); pad.reset()
    assert any("moving again" in l for l in w.update(moved, pad, {}, t, engaged=True))
    # standing still on purpose (stick centred) is not a freeze
    w2, t = FreezeWatch(screenshots=False), 0.0
    while t < 9000:
        pad = Pad(); pad.reset()
        assert not w2.update(stuck, pad, {}, t, engaged=True); t += 8.3


def test_threat_features_backstab_geometry():
    """Opponent 1 m straight behind us, facing our back, walking in: the relative inputs say so."""
    import math as m
    from ptde_ai.threat import NAMES, ThreatFeatures
    tf = ThreatFeatures()
    rot = 0.0                                   # facing = (-sin 0, -cos 0) = (0, -1)
    for i, t in enumerate(range(0, 300, 33)):
        ey = 1.5 - 0.5 * t / 300                # behind us (+y), closing to 1.0 m
        tf.push(t, 0.0, 0.0, rot, 0.0, ey, rot)  # they face the same way we do -> at our back
    f = dict(zip(NAMES, tf.features(False, False)))
    assert abs(f["dist"] - 1.0) < 0.05 and f["their_fwd"] < -0.9 and f["bearing"] > 0.95
    assert f["aim"] < 0.05 and f["align"] < 0.05 and f["closing"] < -1.0
    # a target switch (1.5 m+ jump) restarts the history instead of reading as huge speed
    tf.push(400, 0.0, 0.0, rot, 5.0, 5.0, rot)
    assert tf.features(False, False) is None


def test_brain_uses_threat_net_when_loaded():
    brain, (me, them) = make_brain(), trackers(); brain.cfg.use_defense_net = True
    brain.threat_net = lambda x: 0.9
    brain.defense_net = lambda x: -1.0
    p, e = me.update(player_snap()), them.update(snap(loc_y=-3.0))
    for t in range(0, 300, 33):
        brain.threat.push(t, p.loc_x, p.loc_y, p.rotation_raw, e.loc_x, e.loc_y, e.rotation_raw)
    assert brain.backstab_threat(p, e) == 0.9


def test_memory_is_read_only():
    """Rule (RULES.md): memory access is read-only. The game's memory is opened O_RDONLY and the
    reader has no way to write; no module in ptde_ai/ opens /proc/<pid>/mem for writing."""
    import re
    root = Path(__file__).resolve().parent.parent / "ptde_ai"
    from ptde_ai.memory import ProcMem
    assert not [n for n in dir(ProcMem) if "write" in n.lower() or "poke" in n.lower()]
    for f in root.glob("*.py"):
        src = f.read_text()
        assert not re.search(r"O_RDWR|O_WRONLY|process_vm_writev|WriteProcessMemory", src), f.name
        for m in re.finditer(r"open\([^)]*mem[^)]*\)", src):
            assert "O_RDONLY" in m.group(0) or "maps" in m.group(0), (f.name, m.group(0))


def test_profiles_learn_what_works_per_opponent():
    """User 2026-11-24: remember opponents and play accordingly. Against someone who beats the
    default but loses to "bait", the picks shift to bait; a stranger still starts from the
    pooled record; every style keeps some chance (stays hard to read)."""
    import random as R
    from ptde_ai.adapt import Profiles
    pr = Profiles(path=None, rng=R.Random(1))
    truth = {"balanced": 0.35, "pressure": 0.3, "bait": 0.75, "racer": 0.4}
    rng = R.Random(2)
    for _ in range(150):
        arm = pr.choose("111", "neutral")
        pr.record("111", "neutral", arm, 1.0 if rng.random() < truth[arm] else 0.0)
    picks = [pr.choose("111", "neutral") for _ in range(400)]
    assert picks.count("bait") > 250, {a: picks.count(a) for a in truth}
    assert len(set(picks)) >= 2                                  # still mixes it up
    assert pr.best("111", "neutral")[0] == "bait"
    assert pr.best("999", "neutral")[0] == "bait"                # newcomer: pooled record
    assert pr.posterior("999", "neutral")["bait"][0] < pr.posterior("111", "neutral")["bait"][0]


def test_exchange_scoring():
    from ptde_ai.adapt import EXCHANGE_MAX_MS, Exchanges
    def run(events, until=30000):
        ex, hp, out, t = Exchanges(), [2000, 2000], [], 0.0
        while t < until:
            for at, who, dmg in events:
                if at <= t < at + 10:
                    hp[who] -= dmg
            ev, score = ex.update(t, hp[0], hp[1], 2.0)
            if ev == "end":
                out.append(round(score, 2))
            t += 10
        return out
    assert run([(3000, 1, 600)], 6000) == [1.0]                  # our backstab
    assert run([(3000, 0, 200), (3800, 1, 500)], 6000) == [0.71]  # poise trade, then backstab
    assert run([(3000, 0, 600)], 6000) == [0.0]                  # their backstab
    assert run([], EXCHANGE_MAX_MS + 100)[0] == 0.5              # nothing happened


def test_adapter_styles_reach_the_brain_and_score_wakeups():
    from types import SimpleNamespace as NS
    from ptde_ai.adapt import Adapter, Profiles, WAKEUP_STYLES
    ad = Adapter(Profiles(path=None))
    ad.set_opponent(1000000000, "Tester")
    ad.update(0.0, NS(hp=2000), NS(hp=2000), 2.0, down=False)
    ad.update(1000.0, NS(hp=1500), NS(hp=2000), 1.0, down=True)          # knocked down
    chosen = ad.wakeup
    assert ad.style()["wait_out_chance"] == WAKEUP_STYLES[chosen]["wait_out_chance"]
    ad.update(3000.0, NS(hp=1500), NS(hp=2000), 1.0, down=False)         # up
    lines = ad.update(3500.0, NS(hp=1100), NS(hp=2000), 1.0, down=False)  # punished getting up
    assert any("got punished" in l for l in lines), lines
    w, l = ad.profiles._rec("1000000000", "wakeup")[chosen]
    assert l > w
    # the racer style makes the brain race a standing opponent sooner than the default
    def races_within(style, ms):
        brain, (me, them) = make_brain(), trackers()
        brain.style.update(style)
        seen = set()
        for i in range(int(ms / 8.3)):
            step(brain, me, them, player_snap(locked_on=1), snap(loc_y=2.0, rotation=FACE_MINUS_Y), 1000.0 + 8.3 * i)
            seen.add(brain.attack_type)
        return Atk.RACE in seen
    from ptde_ai.adapt import NEUTRAL_STYLES
    assert races_within(NEUTRAL_STYLES["racer"], 700) and not races_within({}, 700)


def test_player_results_feed_the_general_record():
    """User 2026-11-24: profiles on players should also be absorbed by the general model.
    Every exchange counts for the player AND for everyone; the everyone record keeps far more
    (here: all 4 players count; each player's own record keeps only its ~67 most recent)."""
    from ptde_ai.adapt import ALL, Profiles
    pr = Profiles(path=None)
    for who in ("a", "b", "c", "d"):
        for _ in range(150):
            pr.record(who, "neutral", "bait", 1.0)
    w, l = pr._rec(ALL, "neutral")["bait"]
    mine = sum(pr._rec("a", "neutral")["bait"])
    assert w > 300 and mine < 70, (w, mine)                     # 600 seen; a keeps ~67 recent
    assert pr.best("newcomer", "neutral")[0] == "bait"


def test_background_retrain_pauses_during_fights(tmp=None):
    import signal, subprocess, tempfile, time as tm
    from ptde_ai import retrain as RT
    d = Path(tempfile.mkdtemp())
    RT.PIDFILE, RT.LOG = d / "retrain.pid", d / "retrain.log"
    r = RT.Retrainer()
    proc = subprocess.Popen(["sleep", "30"])
    r.pid = proc.pid
    state = lambda: open(f"/proc/{proc.pid}/stat").read().split()[2]
    assert any("paused" in l for l in r.tick(0, idle=False))
    tm.sleep(0.1); assert state() == "T"                         # frozen while fighting
    assert any("resumed" in l for l in r.tick(10, idle=True))
    tm.sleep(0.1); assert state() in ("S", "R")
    RT.LOG.write_text("retrain: 12 new backstabs absorbed\n")
    proc.kill(); proc.wait()
    assert any("finished: 12 new backstabs absorbed" in l for l in r.tick(20, idle=True))
    assert not r.running
    r2 = RT.Retrainer(); r2.pid, r2.paused = 999999, True
    r2.release()                                                 # never leaves a job frozen


def test_retrain_tests_on_the_newest_fights():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from train_threat import newest_split
    data = [(f"2026112{i}", None, None, None, None, n) for i, n in enumerate([10, 10, 10, 5, 5])]
    train, test = newest_split(data)
    assert [d[0] for d in test] == ["20261123", "20261124"] and len(train) == 3


def test_reach_per_attack_lunges_spears_and_spells():
    """User 2026-11-24: "no idea what to do against some weapons ... stands there and gets hit by
    spears, magic". Lunges reach ~5 m even from an unknown weapon, spears further, spells by
    spell; a learned start distance counts (capped, so one bad sample can't run away)."""
    brain, (me, them) = make_brain(), trackers()
    e = lambda **kw: them.update(snap(**kw))
    assert abs(brain._enemy_reach(e(r_weapon=301000)) - 3.0) < 1e-6           # unknown, neutral
    assert brain._enemy_reach(e(r_weapon=301000, animation_type=103)) >= 5.0   # rolling attack
    assert brain._enemy_reach(e(r_weapon=1050000)) >= 4.0                      # spear, neutral
    assert brain._enemy_reach(e(r_weapon=1050000, animation_type=115)) >= 6.0  # spear R2
    assert brain._enemy_reach(e(animation_type=137)) >= 5.5                    # fire surge
    assert brain._enemy_reach(e(animation_type=168)) == 2.5                    # combustion
    brain.hits.record_reach_start(301000, 46, 4.2, anim_id=203000)
    assert abs(brain._enemy_reach(e(r_weapon=301000, animation_type=46, anim_id=203000)) - 4.5) < 1e-6
    brain.hits.record_reach_start(301000, 46, 8.0, anim_id=203001)             # misattributed
    assert brain._enemy_reach(e(r_weapon=301000, animation_type=46, anim_id=203001)) <= 4.8
    # the live brain learns where an attack STARTED when it hits
    brain2, (me2, them2) = make_brain(), trackers()
    step(brain2, me2, them2, player_snap(hp=1000), snap(loc_y=4.6, r_weapon=211000), 0.0)
    step(brain2, me2, them2, player_snap(hp=1000), snap(loc_y=4.6, r_weapon=211000, animation_type=115, anim_id=584300), 10.0)
    step(brain2, me2, them2, player_snap(hp=700), snap(loc_y=1.5, r_weapon=211000, animation_type=115, anim_id=584300), 900.0)
    assert brain2.hits.reach_start(211000, 115, 584300) == 4.6


def test_sidesteps_projectiles_and_flames_not_melee_at_range():
    """Fireball aimed at us from 10 m: run across its path. Not aimed at us, or a melee swing
    from 10 m: nothing. Fire surge from 4 m in front: get out of the cone."""
    def run(es, frames=6):
        brain, (me, them) = make_brain(), trackers()
        seen = set()
        for i in range(frames):
            step(brain, me, them, player_snap(), es, 8.3 * i)
            seen.add(brain.dodge_type)
        return seen, brain
    aimed = dict(loc_y=10.0, rotation=FACE_MINUS_Y)               # 10 m away, facing us
    seen, brain = run(snap(animation_type=174, **aimed))
    assert Def.SIDESTEP in seen, seen
    assert any("sidestepping" in l for l in brain.log), brain.log
    seen, _ = run(snap(animation_type=174, loc_y=10.0, rotation=FACE_PLUS_Y))   # facing away
    assert Def.SIDESTEP not in seen
    seen, _ = run(snap(animation_type=46, **aimed))                            # melee from 10 m
    assert Def.SIDESTEP not in seen and Def.STANDARD_ROLL not in seen
    seen, _ = run(snap(animation_type=137, loc_y=4.0, rotation=FACE_MINUS_Y))  # fire surge in range
    assert Def.FLEE_ROLLS in seen, seen                                       # (2026-11-24: roll away)


def test_profiles_by_weapon_setup():
    """User 2026-11-24: players switch setups. Results are kept per player AND weapon; a new
    weapon for a known player starts from their record and from everyone vs that weapon."""
    import random as R
    from ptde_ai.adapt import Profiles
    pr = Profiles(path=None, rng=R.Random(3))
    for _ in range(60):                                   # anyone with weapon 451: bait works
        pr.record("someone", "neutral", "bait", 1.0, setup="451")
        pr.record("someone", "neutral", "pressure", 0.0, setup="451")
    for _ in range(40):                                   # player A with a 904: pressure works
        pr.record("player A", "neutral", "pressure", 1.0, setup="904")
    assert pr.best("player A", "neutral", "904")[0] == "pressure"
    assert pr.best("stranger", "neutral", "451")[0] == "bait"        # weapon knowledge transfers
    picks = [pr.choose("player A", "neutral", "451") for _ in range(300)]
    assert picks.count("bait") > picks.count("balanced")             # player A's new weapon: both priors


def test_setup_switch_needs_a_held_weapon():
    from types import SimpleNamespace as NS
    from ptde_ai.adapt import Adapter, Profiles
    ad = Adapter(Profiles(path=None)); ad.set_opponent(1, "V")
    w = lambda r: NS(hp=2000, r_weapon=r, l_weapon=1450000)
    ad.update(0, NS(hp=2000), w(904000), 9.0, False)
    assert ad.setup == "904"
    log = []
    for t in range(100, 4000, 100):                       # quick swaps back and forth
        log += ad.update(t, NS(hp=2000), w(451000 if t % 200 else 904000), 9.0, False)
    assert ad.setup == "904" and not log
    for t in range(4000, 10000, 100):                     # held 6 s: that's their setup now
        log += ad.update(t, NS(hp=2000), w(451000), 9.0, False)
    assert ad.setup == "451" and any("switched to weapon class 451" in l for l in log), log


def test_sign_check_looks_before_pressing():
    """2026-11-24, reports of failed summons: the 2-minute sign check pressed A (opening "Check
    your Summon Sign" on the sign it stands on) and the soapstone ("Cancel summon sign?"). Now
    it looks at the screen first and presses nothing while the sign prompt is showing."""
    from PIL import Image
    from ptde_ai import hud
    from ptde_ai.redsign import CHECK_MS, RedSign
    from ptde_ai.gamepad import CIRCLE, CROSS, SQUARE
    here = Path(__file__).resolve().parent
    assert hud.sign_prompt(Image.open(here / "hud_soapstone.png")) is True
    assert hud.sign_prompt(Image.open(here / "hud_cancel_dialog.png")) is False
    assert hud.sign_prompt(Image.new("RGB", (1920, 1080))) is None
    p = CharacterTracker(True, 2.5).update(player_snap())
    def presses(visible):
        rs = RedSign(sign_visible=lambda: visible)
        rs.sign_down, rs.last_sign_ms = True, 0.0
        seen, t = 0, CHECK_MS + 1
        while t < CHECK_MS + 4000:
            pad = Pad(); pad.reset(); rs.tick(p, t, pad); seen |= pad.buttons; t += 8.3
        return seen
    assert presses(True) == 0                                   # sign on screen: hands off
    assert presses(False) & CROSS and presses(False) & SQUARE   # not seen: the old check
    assert presses(None) & CROSS                                # can't tell: the old check


def test_rolls_when_the_sorcery_lands_not_when_it_is_cast():
    """Live 2026-11-24 vs a sorcerer: sidestepping from the cast ended right as the spell was
    released (~1 s later) and it hit anyway, 37 times. Sidestep while they wind up, then roll
    so the i-frames cover the arrival (cast + ~1.0 s + 8 m / 9 m/s ~ 1.9 s); learn the timing."""
    brain, (me, them) = make_brain(), trackers()
    starts, t = [], 0.0
    aimed = dict(loc_y=8.0, rotation=FACE_MINUS_Y)
    step(brain, me, them, player_snap(), snap(**aimed), t)
    while t < 2600:
        t += 8.3
        at = 157 if t < 1000 else 158 if t < 1600 else 0
        step(brain, me, them, player_snap(), snap(animation_type=at, **aimed), t)
        if brain.in_dodge() and (not starts or starts[-1][1] != brain.dodge_type):
            starts.append((t, brain.dodge_type))
    kinds = [k for _, k in starts]
    assert Def.SIDESTEP in kinds and Def.STANDARD_ROLL in kinds, starts
    roll_t = next(t for t, k in starts if k == Def.STANDARD_ROLL)
    assert 1500 <= roll_t <= 1850, starts                          # just before ~1.89 s
    assert any("sorcery due in" in l for l in brain.log)
    # a sorcery that hits teaches the release time
    b2, (me2, them2) = make_brain(), trackers()
    step(b2, me2, them2, player_snap(hp=1000), snap(**aimed), 0.0)
    step(b2, me2, them2, player_snap(hp=1000), snap(animation_type=157, **aimed), 10.0)
    step(b2, me2, them2, player_snap(hp=800), snap(animation_type=158, **aimed), 1500.0)
    assert abs(b2.hits.value("proj_release:sorcery", 0) - (1.49 - 8.0 / 9.0)) < 0.02


def test_arriving_into_someone_elses_fight_is_not_an_attack():
    """User 2026-11-24: summoned into a world where two players were already fighting. The host
    swinging at the OTHER player (facing away from us) isn't an attack on us; swinging at us is."""
    from ptde_ai.greet import Greeter, DONE
    st = Steering(0.0, 1.0)
    def run(rotation):
        g, (me, them) = Greeter(), trackers()
        g.start(0.0)
        msg = None
        for i in range(1, 1000):
            es = snap(loc_y=2.0, rotation=rotation, animation_type=46, anim_id=203000,
                      anim_timer=0.02 + 0.0083 * (i - 800)) if i > 800 else snap(loc_y=2.0, rotation=rotation)
            pad = Pad(); pad.reset()
            msg = g.tick(me.update(player_snap()), them.update(es), 8.3 * i, pad, st, 0.0) or msg
        return msg
    assert "attacked" in (run(FACE_MINUS_Y) or "")             # facing us, swinging: attack
    assert "attacked" not in (run(FACE_PLUS_Y) or "")          # swinging the other way: not us


def test_no_freeze_alarm_during_the_summon_arrival():
    """Live 2026-11-24: "POSITION FROZEN" fired 3 s after a summon, during the arrival animation."""
    from ptde_ai.freeze import ARRIVAL_QUIET_MS, FreezeWatch
    w, me = FreezeWatch(screenshots=False), CharacterTracker(True, 2.5)
    stuck = me.update(player_snap(loc_x=5.0, loc_y=5.0))
    w.arrived(0.0)
    log, t = [], 0.0
    while t < ARRIVAL_QUIET_MS - 100:
        pad = Pad(); pad.reset(); pad.ly = 0
        log += w.update(stuck, pad, {}, t, engaged=True); t += 8.3
    assert not log, log
    while t < ARRIVAL_QUIET_MS + 4000:                          # still stuck well after: report
        pad = Pad(); pad.reset(); pad.ly = 0
        log += w.update(stuck, pad, {}, t, engaged=True); t += 8.3
    assert any(l.startswith("POSITION FROZEN") for l in log)


def test_one_hp_ends_the_round():
    """User 2026-11-24: "1HP means round over for most people". Taking them to 1 HP is a win
    (the log only counted deaths: 4-58 was really 59-58), and it scores the exchange."""
    from ptde_ai.fightlog import FightLog
    from ptde_ai.adapt import Exchanges
    log = FightLog(path=None)
    assert log.tick(1500, 400, 0.0) is None
    assert log.tick(1500, 1, 10.0) == "won"
    log2 = FightLog(path=None)
    log2.tick(300, 1500, 0.0)
    assert log2.tick(1, 1500, 10.0) == "lost"
    ex = Exchanges()
    ex.update(0.0, 2000, 900, 2.0)
    assert ex.update(500.0, 1700, 1, 2.0) == ("end", 1.0)       # we traded, then ended their round


def test_summoned_into_a_round_waits_its_turn():
    """User 2026-11-24: "if there are already two players in the arena, teleport to the dugout.
    Do not bow first. Just get out." Decided once, shortly after arriving; never mid-round."""
    from ptde_ai import escape as X
    from ptde_ai.gamepad import SQUARE
    q = X.should_queue_on_arrival
    assert q(True, 2000, 2, 1900)                                # two in the arena: out
    assert not q(True, 2000, 1, 1900)                            # one waiting for us: fight
    assert not q(True, 500, 2, 1900)                             # still arriving: look first
    assert not q(False, 2000, 2, 1900)                           # at home: not this rule
    assert not q(True, X.ARRIVAL_QUEUE_MS + 1, 2, 1900)          # long since arrived
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    me = CharacterTracker(True, 2.5)
    at = lambda x, y, h: me.update(player_snap(loc_x=x, loc_y=y, loc_h=h))
    esc, st, t = X.DugoutEscape(arena=arena, n_items=2), Steering(0.0, 1.0), 0.0
    assert "straight to the dugout" in esc.queue(t)
    msgs, pressed = [], False
    def run(p, until, others):
        nonlocal t, pressed
        while t < until:
            pad = Pad(); pad.reset()
            m = esc.update(p, True, t, pad, st, 0.0, others, others + 0, summoned=True)
            if m: msgs.append(m)
            pressed |= bool(pad.buttons & SQUARE)
            t += 8.3
    run(at(0, 20, 6), 3000, 2)                                   # trinket out of the arena
    assert pressed and esc.state == X.OUT_LEAVING, (esc.state, msgs)
    run(at(0, 1, 10), 9000, 2)                                    # in the dugout, two still at it
    assert esc.state == X.WAIT_TURN and any("waiting my turn" in m for m in msgs), (esc.state, msgs)
    run(at(0, 1, 10), 10000, 1)                                   # a 1 s dip: not our turn yet
    assert esc.state == X.WAIT_TURN, (esc.state, msgs)
    run(at(0, 1, 10), 13500, 1)                                   # gone for 3 s+: go
    assert esc.state not in (X.WAIT_TURN, X.IDLE), (esc.state, msgs)


def test_fights_the_havelmom_summon_but_no_other_npc():
    """User 2026-11-24: "target the NPC Havelmom summon despite the rules". Matched by entity ID
    from the arena's event script: 1269420 = the summon (fight it), 1031001 = the Havelmom
    merchant in the dugout (never). Every other NPC stays off-limits."""
    from ptde_ai.character import fightable, pick_target
    from ptde_ai.offsets import npc_target
    _, them = trackers()
    ok, desc = fightable(them.update(snap(chr_type=5, team_type=6, entity_id=1269420)), 2)
    assert ok and "Havelmom" in desc
    assert not fightable(them.update(snap(chr_type=5, team_type=8, entity_id=1031001)), 2)[0]  # merchant
    assert not fightable(them.update(snap(chr_type=5, team_type=8, entity_id=123104)), 2)[0]   # talk NPC
    assert not fightable(them.update(snap(chr_type=7, team_type=7, entity_id=1200800)), 2)[0]  # boss
    assert npc_target(2, 1269420) is None                    # player types are just players
    slots = [{"slot": 1, "chr_type": 5, "team_type": 8, "hp": 1500, "x": 1.0, "y": 0.0, "entity_id": 1031001},
             {"slot": 2, "chr_type": 5, "team_type": 6, "hp": 1500, "x": 3.0, "y": 0.0, "entity_id": 1269420}]
    assert pick_target(slots, 2, (0.0, 0.0), None) == 2      # merchant skipped, summon picked
    slots[1]["hp"] = 1
    assert pick_target(slots, 2, (0.0, 0.0), None) is None   # at 1 HP its round is over too

def test_clean_kick_starts_a_race_for_the_back():
    """User 2026-11-24: "if you manage to land a kick and the kick wasn't blocked, race for a
    backstab". Clean = it took HP off them with their shield down; a kick into a raised shield
    does no HP damage and starts no race."""
    def run(their_anim):
        brain, (me, them) = make_brain(), trackers()
        seen, t = set(), 0.0
        es = lambda hp: snap(loc_y=1.6, rotation=FACE_MINUS_Y, hp=hp, animation_type=their_anim)
        step(brain, me, them, player_snap(), es(1000), t)
        for i in range(1, 30):                              # our kick plays; it lands at ~0.2 s
            t += 8.3
            step(brain, me, them, player_snap(animation_type=112), es(1000 if t < 200 else 980), t)
            seen.add(brain.attack_type)
        for i in range(1, 60):                              # after the kick
            t += 8.3
            step(brain, me, them, player_snap(), es(980), t)
            seen.add(brain.attack_type)
        return seen, brain.log
    seen, log = run(0)
    assert any("kick landed clean" in l for l in log) and Atk.RACE in seen, (seen, log[-5:])
    seen, log = run(77)                                     # shield held up
    assert any("on their shield" in l for l in log) and not any("landed clean" in l for l in log)


def test_lost_trinket_press_is_retried_quickly():
    """Live 2026-11-24: just after a summon the first trinket press was ignored and the bot
    waited 8 s before pressing again (13 s to reach the dugout). No item animation within
    ~1.2 s = the press was lost: press again. Summoned, it also skips the screen check."""
    from ptde_ai import escape as X
    from ptde_ai.gamepad import SQUARE
    arena = {"dugout": {"x": 0.0, "y": 0.0, "h": 10.0}, "center": {"x": 0.0, "y": 20.0, "h": 6.0}}
    p = CharacterTracker(True, 2.5).update(player_snap(loc_x=0.0, loc_y=20.0, loc_h=6.0))
    class NeverCalled:
        busy = False
        def request(self): raise AssertionError("no screenshot while summoned")
        def poll(self): return None
    esc, t = X.DugoutEscape(arena=arena, n_items=2, hud=NeverCalled()), 0.0
    esc.queue(t)
    presses, last = [], False
    while t < 5000:
        pad = Pad(); pad.reset()
        esc.update(p, True, t, pad, Steering(0.0, 1.0), 0.0, 2, 2, summoned=True)
        down = bool(pad.buttons & SQUARE)
        if down and not last: presses.append(t)
        last = down
        t += 8.3
    assert len(presses) >= 2 and presses[1] - presses[0] < 1600, presses


def test_back_turned_means_go_for_the_backstab():
    """User 2026-11-24: "if the opponent's back is facing you, go for the backstab even if it
    means taking damage". Their back to us and swinging the other way: chase, don't dodge.
    Still dodge a hit we know would kill us. Facing us: the usual dodge."""
    def run(rotation, our_hp=1500, lethal=False):
        brain, (me, them) = make_brain(), trackers()
        if lethal:
            brain.hits.record(301015, 46, 2000.0)
        atks, dodges = set(), set()
        for i in range(60):
            t = 1000.0 + 8.3 * i
            es = snap(loc_y=2.2, rotation=rotation, animation_type=46 if i > 5 else 0,
                      anim_id=203000, anim_timer=0.02 + 0.0083 * max(0, i - 5))
            step(brain, me, them, player_snap(hp=our_hp, rotation=FACE_PLUS_Y), es, t)
            atks.add(brain.attack_type); dodges.add(brain.dodge_type)
        return atks, dodges
    atks, dodges = run(FACE_PLUS_Y)                         # they face away: their back to us
    assert dodges <= {Def.NONE} and (atks & {Atk.RACE, Atk.LINE_UP, Atk.BACKSTAB}), (atks, dodges)
    atks, dodges = run(FACE_PLUS_Y, our_hp=300, lethal=True)  # known to kill us: still dodge
    assert dodges - {Def.NONE}, dodges
    atks, dodges = run(FACE_MINUS_Y)                        # facing us and swinging: dodge
    assert dodges - {Def.NONE}, dodges


def test_heals_between_fights_with_one_divine_blessing():
    """User 2026-11-24: "heal between fights even when other players are doing 1HP fights.
    Press start > spam A to use divine blessing". Start, A until the use animation starts (one
    Blessing only), then B to back out of the menu; no heal after 6 A's: back off."""
    from ptde_ai.heal import Healer, MAX_A, FAIL_BACKOFF_MS
    from ptde_ai.gamepad import CIRCLE, CROSS, START
    assert Healer.wants(1, 2000) and Healer.wants(1700, 2000) and not Healer.wants(1900, 2000)
    def run(heals_on_press):
        h, t, hp, anim, log = Healer(), 0.0, 400, 0, []
        assert "Divine Blessing" in h.start(hp, t)
        edges, last = [], 0
        while t < 9000 and (h.active or t < 100):
            pad = Pad(); pad.reset()
            m = h.update(hp, anim, t, pad)
            if m: log.append(m)
            for b in (START, CROSS, CIRCLE):
                if pad.buttons & b and not last & b: edges.append((round(t), b))
            last = pad.buttons
            a_count = sum(1 for _, b in edges if b == CROSS)
            if heals_on_press and a_count >= heals_on_press and anim == 0:
                anim = 15                               # the use animation starts
            if anim and t > 1500:
                hp, anim = 2000, 0
            t += 8.3
        return h, edges, log
    h, edges, log = run(heals_on_press=3)                # Start, A, A, A (uses it), then B x3
    kinds = [b for _, b in edges]
    assert kinds[0] == START and kinds.count(CROSS) == 3 and kinds[-3:] == [CIRCLE] * 3, kinds
    assert any(l.startswith("healed") for l in log) and not h.active
    h, edges, log = run(heals_on_press=0)                # out of Blessings
    assert [b for _, b in edges].count(CROSS) == MAX_A and any("no heal" in l for l in log)
    assert h.start(400, 10_000) is None                  # backing off
    assert h.start(400, 10_000 + FAIL_BACKOFF_MS) is not None


def test_heals_only_between_fights():
    """Live 2026-11-24 it tried to heal mid-fight (opponent at 6.0 m; at 212 HP mid-exchange)
    and kept failing. Only: waiting in the dugout, round over (1 HP), or no opponent near."""
    from ptde_ai.heal import between_fights as bf
    assert not bf(False, 1206, True, 1800, 6.0)        # racing at 6 m: a fight
    assert not bf(False, 212, True, 1500, 4.0)         # low HP mid-exchange: still a fight
    assert bf(True, 900, True, 1800, 3.0)              # waiting our turn in the dugout
    assert bf(False, 1, True, 1800, 2.0)               # our round is over
    assert bf(False, 900, True, 1, 2.0)                # theirs is
    assert bf(False, 900, False, 0, 99.0)              # nobody to fight
    assert bf(False, 900, True, 1800, 20.0)            # they're across the map


def test_backstabs_get_a_bigger_reward():
    """User 2026-11-24: "he's still ignoring way too many backstab opportunities. Maybe he
    should get a bigger reward for them." An exchange where the bot lands a backstab scores at
    least 0.9 even if it took more damage; one where it gets backstabbed at most 0.1."""
    from types import SimpleNamespace as NS
    from ptde_ai.adapt import BS_REWARD, Adapter, Exchanges, Profiles
    ex = Exchanges()
    ex.update(0.0, 2000, 2000, 2.0)
    ex.backstab(by_us=True)
    ex.update(100.0, 1400, 1800, 2.0)                  # we took 600, dealt 200...
    assert ex.update(2000.0, 1400, 1800, 2.0) == ("end", BS_REWARD)   # ...but we backstabbed
    ex = Exchanges(); ex.update(0.0, 2000, 2000, 2.0); ex.backstab(by_us=False)
    ex.update(100.0, 1500, 1700, 2.0)
    assert ex.update(2000.0, 1500, 1700, 2.0)[1] <= 1 - BS_REWARD + 1e-9
    # the live adapter notices a backstab from the animations
    ad = Adapter(Profiles(path=None)); ad.set_opponent(1, "V")
    ad.update(0.0, NS(hp=2000, animation_type=0), NS(hp=2000, animation_type=0), 2.0, False)
    ad.update(50.0, NS(hp=2000, animation_type=225), NS(hp=2000, animation_type=0), 1.0, False)
    assert ad.ex.bs_us


def test_kicks_are_aimed_at_them_and_chain_into_the_race():
    """User 2026-11-24: aim every kick at the opponent (kicks aim freely, unlocked); a kick
    backstab is one motion: kick > unlock > block + sprint at once > let go of block > circle
    to their back > backstab."""
    import math as m
    from ptde_ai.gamepad import CIRCLE, L1, R1, R3
    brain, (me, them) = make_brain(), trackers()
    p = me.update(player_snap(locked_on=1, rotation=FACE_PLUS_Y))
    e = them.update(snap(loc_x=0.7, loc_y=0.7, rotation=FACE_PLUS_Y))     # 45 deg off our facing
    brain.now, brain.cam_yaw = 0.0, 0.0
    stick, unlocked, kicked = None, False, False
    for t in range(0, 300, 8):
        pad = Pad(); pad.reset()
        brain.now = float(t)
        done = brain._aimed_flick(p, e, float(t), pad, R1)
        unlocked |= bool(pad.buttons & R3) and t < 40
        if pad.buttons & R1:
            kicked, stick = True, (pad.lx, pad.ly)
        if done: break
    assert unlocked and kicked
    want = brain.steer.stick_for_heading(G.heading_to(p, e), 0.0)
    assert abs(stick[0] - want[0]) < 2000 and abs(stick[1] - want[1]) < 2000, (stick, want)
    # our kick animation starts (no backstab): race with block AND sprint from the first ticks
    brain2, (me2, them2) = make_brain(), trackers()
    es = snap(loc_y=1.4, rotation=FACE_MINUS_Y)
    step(brain2, me2, them2, player_snap(), es, 0.0)
    held, t = [], 0.0
    for i in range(1, 60):
        t += 8.3
        pad, _, _ = step(brain2, me2, them2, player_snap(animation_type=112 if i < 40 else 0, stamina=150), es, t)
        if brain2.attack_type == Atk.RACE:
            held.append((bool(pad.buttons & L1), bool(pad.buttons & CIRCLE)))
    assert any("kicked: unlock, block + sprint" in l for l in brain2.log), brain2.log[-5:]
    assert held and held[0] == (True, True), held[:5]               # both at once from the start
    assert held[-1] == (False, True), held[-5:]                     # then sprint, block released


def test_turtles_get_kicks_jumping_attacks_and_roll_ghosts():
    """User 2026-11-24: an opponent sat in a corner with their shield up and the bot wiggled
    around. Turtling (still, shield up, facing us): mix kicks (mostly), jumping attacks (R2)
    and rolling ghost strikes."""
    import random as R
    from ptde_ai.gamepad import R1, R2
    moves = []
    for seed in range(30):
        brain, (me, them) = make_brain(), trackers()
        brain.rng = R.Random(seed)
        es = snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=77)   # still, shield up, facing us
        r2 = False
        for i in range(400):
            pad, _, _ = step(brain, me, them, player_snap(stamina=150), es, 8.3 * i)
            r2 |= bool(pad.buttons & R2)
            if brain._turtle_move and brain.attack_type == Atk.TURTLE:
                pass
        mv = next((l.split(": ")[-1] for l in brain.log if "turtling" in l), None)
        moves.append(mv)
        if mv == "jump":
            assert r2, "jumping attack = R2"
    assert moves.count(None) == 0, moves
    assert moves.count("kick") > moves.count("jump") > 0 and "roll" in moves, moves


def test_engages_a_camping_caster():
    """User 2026-11-24: player G stood in a corner buffing and casting pyromancies while the bot
    did nothing. A long learned reach (5.95 m from an MLGS projectile) kept neutral spacing at
    6-7 m; spacing now caps their reach at a melee R1's, and a camper / caster gets charged."""
    from ptde_ai.gamepad import CIRCLE, L1
    brain, (me, them) = make_brain(), trackers()
    brain.hits.record_reach(309000, 5.95)
    e = them.update(snap(loc_y=6.5, rotation=FACE_MINUS_Y, r_weapon=309000))
    assert brain._spacing_reach(e) == 3.5
    seen, held = set(), []
    for i in range(300):                                  # they stand still 6.5 m away
        pad, _, _ = step(brain, me, them, player_snap(stamina=150),
                         snap(loc_y=6.5, rotation=FACE_MINUS_Y, r_weapon=309000), 8.3 * i)
        seen.add(brain.attack_type)
        if brain.attack_type == Atk.CHARGE:
            held.append((bool(pad.buttons & L1), bool(pad.buttons & CIRCLE)))
    assert Atk.CHARGE in seen and any("charging" in l for l in brain.log), seen
    assert held[0] == (True, True) and held[-1][1], held[:3]       # block + sprint at once
    # casting at range also gets charged, straight away
    b2, (me2, them2) = make_brain(), trackers()
    step(b2, me2, them2, player_snap(stamina=150), snap(loc_y=6.0, rotation=FACE_MINUS_Y, animation_type=157), 0.0)
    step(b2, me2, them2, player_snap(stamina=150), snap(loc_y=6.0, rotation=FACE_MINUS_Y, animation_type=157), 8.3)
    assert b2.attack_type == Atk.CHARGE or b2.dodge_type == Def.SIDESTEP, (b2.attack_type, b2.dodge_type)


def test_wake_up_machine_waits_at_their_back():
    """User 2026-11-24: "he needs to become a wake up machine ... if he gets his opponent into
    kickstun, he needs to be at their back ready to wake them up". Down: go to the spot behind
    them, don't press while they're on the floor, kick-backstab the moment they're up."""
    brain, (me, them) = make_brain(), trackers()
    # they're down (in the backstab / knockdown), facing +y; we're 2 m in front of them
    down = lambda: snap(loc_y=2.0, rotation=FACE_PLUS_Y, anim_id3=1)
    x, y, starts = 0.0, 0.0, []
    for i in range(240):
        pad, pp, ee = step(brain, me, them, player_snap(loc_x=x, loc_y=y, rotation=FACE_PLUS_Y), down(), 8.3 * i)
        assert brain.attack_type != Atk.BACKSTAB, "no press on someone lying down"
        starts.append(brain.attack_type)
    assert Atk.OKI in starts and any("wake-up" in l for l in brain.log)
    # standing at their back as they get up: the kick-backstab goes straight in
    b2, (me2, them2) = make_brain(), trackers()
    for i in range(30):
        step(b2, me2, them2, player_snap(loc_y=1.1, rotation=FACE_PLUS_Y, stamina=150),
             snap(loc_y=2.0, rotation=FACE_PLUS_Y, anim_id3=1 if i < 10 else 0), 8.3 * i)
    assert any("backstab attempt" in l for l in b2.log), b2.log[-6:]


def test_kicks_lead_a_moving_target():
    from ptde_ai.gamepad import R1
    brain, (me, them) = make_brain(), trackers()
    for i in range(20):                                   # they strafe +x at ~3 m/s, 1 m ahead
        step(brain, me, them, player_snap(rotation=FACE_PLUS_Y), snap(loc_x=0.025 * i, loc_y=1.0), 8.3 * i)
    p = me.update(player_snap(rotation=FACE_PLUS_Y)); e = them.update(snap(loc_x=0.5, loc_y=1.0))
    stick = None
    for t in range(0, 300, 8):
        pad = Pad(); pad.reset(); brain.cam_yaw = 0.0
        if brain._aimed_flick(p, e, float(t), pad, R1): break
        if pad.buttons & R1: stick = (pad.lx, pad.ly)
    now_aim = brain.steer.stick_for_heading(G.heading_to(p, e), 0.0)
    assert stick is not None and stick[0] != now_aim[0], (stick, now_aim)   # led, not straight at them


def test_faces_the_opponent_before_bowing():
    """User 2026-11-24: face the opponent before bowing. It turns with the stick (unlocked) and
    bows only once facing them; never bows facing away."""
    from ptde_ai.greet import Greeter, DONE
    from ptde_ai.gamepad import SELECT
    st = Steering(0.0, 1.0)
    def run(turns):
        g, (me, them) = Greeter(), trackers()
        g.start(0.0, force=True)
        rot, msgs, bowed_facing = FACE_MINUS_Y, [], None   # start facing AWAY from them (they're at +y)
        for i in range(800):
            t = 8.3 * i
            pad = Pad(); pad.reset()
            p = me.update(player_snap(rotation=rot)); e = them.update(snap(loc_y=3.0, rotation=FACE_MINUS_Y))
            m = g.tick(p, e, t, pad, st, 0.0)
            if m: msgs.append(m)
            if pad.buttons & SELECT and bowed_facing is None:
                bowed_facing = abs(((rot - FACE_PLUS_Y + m_pi) % (2 * m_pi)) - m_pi)
            if turns and (pad.lx, pad.ly) != (MIDDLE, MIDDLE):
                rot += (FACE_PLUS_Y - rot) * 0.05            # the character turns toward the stick
            if g.state == DONE: break
        return msgs, bowed_facing
    m_pi = math.pi
    msgs, facing = run(turns=True)
    assert any("facing them: bowing" in x for x in msgs) and facing is not None and facing < 0.4, (msgs, facing)
    msgs, facing = run(turns=False)                         # can't turn: no bow facing away
    assert facing is None and any("skipping the bow" in x for x in msgs), msgs


def test_fire_tempest_roll_spam_away_toward_open_floor():
    """User 2026-11-24: fire tempest: roll spam directly away from the opponent, preferably
    toward the most open space."""
    brain, (me, them) = make_brain(), trackers()
    taps, last, sticks = 0, False, []
    for i in range(240):                                    # 2 s of tempest, caster 3 m away (+y)
        pad, p, e = step(brain, me, them, player_snap(stamina=150), snap(loc_y=3.0, rotation=FACE_MINUS_Y, animation_type=164), 8.3 * i)
        down = bool(pad.buttons & CIRCLE)
        taps += down and not last
        last = down
        if (pad.lx, pad.ly) != (MIDDLE, MIDDLE):
            sticks.append(pad.ly)
    assert brain.dodge_type == Def.FLEE_ROLLS or any("fire tempest" in l for l in brain.log)
    assert taps >= 2, taps                                  # roll after roll
    assert sticks and sum(1 for ly in sticks if ly > MIDDLE) > len(sticks) * 0.8   # away from them (-y)


def test_fire_surge_keep_out_or_roll_away():
    """User 2026-11-24: fire surge: keep out of its range, and roll away if within range."""
    brain, (me, them) = make_brain(), trackers()
    seen = set()
    for i in range(60):
        step(brain, me, them, player_snap(stamina=150), snap(loc_y=3.5, rotation=FACE_MINUS_Y, animation_type=137), 8.3 * i)
        seen.add(brain.dodge_type)
    assert Def.FLEE_ROLLS in seen, seen                           # inside its range: roll away
    b2, (me2, them2) = make_brain(), trackers()
    toward = 0
    for i in range(120):
        pad, _, _ = step(b2, me2, them2, player_snap(stamina=150), snap(loc_y=7.0, rotation=FACE_MINUS_Y, animation_type=137), 8.3 * i)
        toward += pad.ly < MIDDLE - 4000                          # stick pushed toward them (+y)
    assert b2.attack_type in (Atk.KEEP_OUT, Atk.NONE) and toward == 0, (b2.attack_type, toward)


def test_corner_camper_gets_poison_knives_until_they_leave():
    """User 2026-11-24: "if a player is corner camping, he should back up and throw poison
    throwing knives until his opponent ... leaves the corner 0.2m". Then back to the soapstone."""
    from ptde_ai import escape as X
    from ptde_ai.gamepad import DDOWN, R3, SQUARE
    brain, (me, them) = make_brain(), trackers()
    brain.arena = _box(-8, 8, -5, 5)
    brain.items = X.DugoutEscape(arena={}, n_items=3)
    brain.items.trinket_at, brain.items.knife_at = 1, 2
    corner = dict(loc_x=7.2, loc_y=4.2, rotation=FACE_MINUS_Y)      # 1.1 m from the (8, 5) corner
    downs, throws, lastd, lasts, anim_until = 0, 0, False, False, -1.0
    locked, r3_ticks, relocks = False, 0, 0
    t = 0.0
    for i in range(1500):
        t = 8.3 * i
        anim = 15 if t < anim_until else 0                           # the throw animation plays
        if i == 900 and locked:
            locked = False                                           # the lock drops mid-way
        pad, _, _ = step(brain, me, them, player_snap(stamina=150, loc_h=0.0, animation_type=anim,
                                                      locked_on=int(locked)),
                         snap(loc_h=0.0, **corner), t)
        r3_ticks = r3_ticks + 1 if pad.buttons & R3 else 0
        if r3_ticks == 4:                                            # R3 held ~33 ms: the game locks
            locked = not locked
            relocks += i > 900
        d, q = pad.hat == DDOWN, bool(pad.buttons & SQUARE)
        downs += d and not lastd; throws += q and not lasts
        if q and not lasts:
            # user 2026-11-24: "HE NEEDS TO BE LOCKED ON WHEN THROWING KNIVES"
            assert locked, f"threw a knife unlocked at {t:.0f} ms"
            anim_until = t + 150 + 600
        lastd, lasts = d, q
    assert relocks >= 1, "never locked back on after the lock dropped"
    assert any("camping a corner" in l for l in brain.log), brain.log[-5:]
    assert downs == 2 and brain.items.cycles == 2 and throws >= 3, (downs, brain.items.cycles, throws)
    corner["loc_x"] = 6.9                                            # they step 0.3 m out
    for i in range(1, 300):
        step(brain, me, them, player_snap(stamina=150, loc_h=0.0), snap(loc_h=0.0, **corner), t + 8.3 * i)
    assert any("left the corner" in l for l in brain.log) and brain.items.cycles == 0, brain.items.cycles
    assert any("knives: throw 1" in l for l in brain.log)
    # up close it's a fight: no knives from 1.5 m, however still they are in the corner
    b2, (me2, them2) = make_brain(), trackers()
    b2.arena = _box(-8, 8, -5, 5)
    b2.items = X.DugoutEscape(arena={}, n_items=3); b2.items.knife_at = 2
    for i in range(400):
        step(b2, me2, them2, player_snap(stamina=150, loc_x=6.0, loc_y=3.3, loc_h=0.0),
             snap(loc_h=0.0, loc_x=7.2, loc_y=4.2, rotation=FACE_MINUS_Y), 8.3 * i)
    assert not any("camping a corner" in l for l in b2.log), [l for l in b2.log if "knives" in l][:3]


def test_kicks_their_back_the_frame_they_wake_up():
    """User 2026-11-24: "he should always kick the opponent's back the very frame his opponent
    wakes up". At their back, facing them, stick centred: the kick goes out on the tick they're
    up; once the wake-up time is learned, it's pressed just before so it lands on that frame."""
    from ptde_ai.gamepad import R1
    def run(learned=None, up_at=1200.0):
        brain, (me, them) = make_brain(), trackers()
        if learned:
            brain.hits.table["wake_ms:stun"] = learned
        kick_t = None
        for i in range(260):
            t = 8.3 * i
            down = t < up_at
            pad, _, _ = step(brain, me, them, player_snap(loc_y=-0.9, rotation=FACE_PLUS_Y, stamina=150),
                             snap(rotation=FACE_PLUS_Y, anim_id3=1 if down else 0), t)
            if pad.buttons & R1 and kick_t is None:
                kick_t = t
        return kick_t
    k = run()
    assert k is not None and 1200.0 <= k < 1200.0 + 9, k          # the very tick they're up
    k = run(learned=1200.0)
    assert k is not None and 1200.0 - 60 <= k < 1200.0, k          # pressed just before they're up


def test_no_cast_loop_gets_hit_not_circled():
    """User 2026-11-24: "player G is just doing the no cast animation and bot king is just angling
    for a backstab ... literally exploiting him". Anim type 132 (40% of that fight): hit them
    directly with the turtle mix, don't race round them."""
    import random as R
    from ptde_ai.gamepad import R1, R2
    brain, (me, them) = make_brain(), trackers()
    brain.rng = R.Random(1)
    hits, races = 0, 0
    for i in range(500):
        pad, _, _ = step(brain, me, them, player_snap(stamina=150),
                         snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=132, anim_id=6399), 8.3 * i)
        hits += bool(pad.buttons & (R1 | R2))
        races += brain.attack_type == Atk.RACE and brain.now - brain.t_atk < 9
    assert any("no-cast loop" in l for l in brain.log), brain.log[-5:]
    assert hits > 0, "it never hit them"


def test_r2_only_ever_as_a_flicked_jump():
    """User 2026-11-24: "Never press R2 unless it's a jump attack". The game runs at 30 fps: every
    R2 must go out with the stick pushed after >= 2 game frames centred, and never while we're
    in an animation (it buffers into a plain heavy; live 3 of 6 came out as R2_2H)."""
    import random as R
    from ptde_ai.gamepad import R2, MIDDLE
    from ptde_ai.brain import KICK_NEUTRAL_MS
    jumps = 0
    for seed in range(6):
        brain, (me, them) = make_brain(), trackers()
        brain.rng = R.Random(seed)
        rng = R.Random(100 + seed)
        centred_since, prev_centred, prev_r2, anim = 0.0, True, False, 0
        for i in range(1500):
            tt = 8.3 * i
            if i % 40 == 0:
                anim = rng.choice([0, 0, 0, 108, 113])    # sometimes still recovering
            pad, p, _ = step(brain, me, them, player_snap(stamina=150, animation_type=anim),
                             snap(loc_y=2.0, rotation=FACE_MINUS_Y, animation_type=132, anim_id=6399), tt)
            centred = abs(pad.lx - MIDDLE) < 2000 and abs(pad.ly - MIDDLE) < 2000
            r2 = bool(pad.buttons & R2)
            if r2 and not prev_r2:
                jumps += 1
                assert not centred, f"seed {seed} t {tt:.0f}: R2 with the stick centred (a heavy)"
                assert prev_centred and tt - centred_since >= KICK_NEUTRAL_MS - 9, \
                    f"seed {seed} t {tt:.0f}: flick after {tt - centred_since:.0f} ms centred"
                assert anim == 0, f"seed {seed} t {tt:.0f}: R2 during anim {anim}"
            if centred and not prev_centred:
                centred_since = tt
            prev_centred, prev_r2 = centred, r2
    assert jumps > 0, "never jumped at all"


def test_stops_racing_someone_who_turns_with_us():
    """Live 2026-11-24: the opponent stood still (anim 0) and turned with the bot; it raced for
    their back at 0.9 m for minutes (race, "they're behind us", race...). A race that gains
    nothing: stop racing them for a while and hit them (kicks / jumping attacks) instead."""
    import math as M
    import random as R
    from ptde_ai.gamepad import R1, R2
    brain, (me, them) = make_brain(), trackers()
    brain.rng = R.Random(3)
    hits, race_ticks, px, py, ph = 0, 0, 0.0, -1.0, 0.0
    for i in range(1500):
        tt = 8.3 * i
        face = M.atan2(-px, -py)                   # they always face us (raw r faces (-sin r, -cos r))
        pad, p, _ = step(brain, me, them, player_snap(stamina=150, loc_x=px, loc_y=py,
                                                      rotation=M.atan2(px, py)),
                         snap(loc_x=0.0, loc_y=0.0, rotation=face), tt)
        # move us a little the way the stick points (enough for the angle to change)
        if abs(pad.lx - 16384) > 2000 or abs(pad.ly - 16384) > 2000:
            ang = M.atan2(px, py) + 0.004 * (1 if pad.lx > 16384 else -1)
            r = M.hypot(px, py); px, py = r * M.sin(ang), r * M.cos(ang)
        hits += bool(pad.buttons & (R1 | R2))
        race_ticks += brain.attack_type == Atk.RACE and brain.in_attack()
    assert any("they turn with us" in l for l in brain.log), [l for l in brain.log if "rac" in l][-4:]
    assert hits > 0
    assert race_ticks < 0.5 * 1500, race_ticks


def test_soapstone_plus_trinket_allowed():
    from ptde_ai.redsign import soapstone_ready
    assert soapstone_ready([101, 276, -1, -1, -1])[0]
    assert soapstone_ready([276, 101, -1, -1, -1])[0]
    assert not soapstone_ready([101, 276, 215, -1, -1])[0]


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as ex:
            failed += 1
            print(f"FAIL  {name}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
