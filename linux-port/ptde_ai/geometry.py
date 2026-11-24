"""Spatial helpers (port of HelperUtil.c), with the bugs fixed.

Conventions (derived from metal-crow's backstab code, which is map-independent):
  * rotation_deg = (raw_rotation + pi) in degrees, 0..360, as ReadPlayer stores it.
  * A character with raw rotation r faces the world direction (-sin r, -cos r).
  * heading(dx, dy) = atan2(dx, dy), so a character's facing heading is r + pi.

legacy_* functions reproduce the original math exactly. They exist only to build
neural-net inputs, because the training data was recorded with those formulas.
Control code uses the corrected versions.
"""
from __future__ import annotations

import math

BACKSTAB_DEGREE_RANGE = 60.0   # attacker's facing must be within +-60 deg of the target's
BACKSTAB_RANGE = 1.5           # max distance for a backstab to trigger
REAL_BEHIND_RANGE = 0.1        # how far behind the target's plane counts as "behind"
# Stricter window for actually pressing R1 to backstab. Live 2026-11-22, all plain R1s:
# 1.3 m / 40 deg; 0.5-0.7 m behind but 33-52 deg off their back axis with facing ~43 deg;
# and 1.16 m dead behind during their windup. Position angle matters, not just facing.
# From the user's 18 landed backstabs vs player B (2026-11-23): distance median 0.87 m
# (max 1.18), 20 deg off their back (max 52), facing difference 22 deg (max 56). The old
# 1.0 m / 25 / 25 window passed up chances a human takes.
BS_ATTEMPT_RANGE = 1.3     # user 2026-11-24: a bit more for players bent over (was 1.15; user's landed max 1.18)
BS_ATTEMPT_DEGREES = 50.0        # our facing vs theirs (user's landed max 56)
BS_ATTEMPT_POS_DEGREES = 50.0    # where we stand, measured from their back (user's landed max 52)
# (the input is a kick: a miss just kicks them, so the window can be as wide as a human's)


def distance(a, b) -> float:
    # Original took fabsf() of each coordinate first, which is wrong whenever the
    # two characters straddle an axis at 0.
    return math.hypot(a.loc_x - b.loc_x, a.loc_y - b.loc_y)


def facing_vector(c) -> tuple[float, float]:
    r = c.rotation_raw
    return -math.sin(r), -math.cos(r)


def facing_heading(c) -> float:
    return math.atan2(*facing_vector(c))


def heading_to(a, b) -> float:
    """World heading from a toward b."""
    return math.atan2(b.loc_x - a.loc_x, b.loc_y - a.loc_y)


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def can_backstab_now(attacker, target, dist: float | None = None) -> bool:
    """Tight check used before pressing R1 for a backstab."""
    if dist is None:
        dist = distance(attacker, target)
    fx, fy = facing_vector(target)
    relx, rely = attacker.loc_x - target.loc_x, attacker.loc_y - target.loc_y
    behind = -(relx * fx + rely * fy)
    side = relx * fy - rely * fx
    return (behind > REAL_BEHIND_RANGE and dist <= BS_ATTEMPT_RANGE
            and math.degrees(math.atan2(abs(side), behind)) <= BS_ATTEMPT_POS_DEGREES
            and angle_diff_deg(target.rotation_deg, attacker.rotation_deg) <= BS_ATTEMPT_DEGREES)


def backstab_state(attacker, target, dist: float | None = None) -> int:
    """0 = not behind target, 1 = behind target, 2 = in position to backstab.

    Same rule as BackstabDetection_CounterClockwise, but measured along the target's
    true facing instead of snapping it to the nearest of four axes.
    """
    if dist is None:
        dist = distance(attacker, target)
    fx, fy = facing_vector(target)
    # Distance of the attacker in front of the target (negative = behind).
    ahead = (attacker.loc_x - target.loc_x) * fx + (attacker.loc_y - target.loc_y) * fy
    if ahead >= -REAL_BEHIND_RANGE:
        return 0
    if (angle_diff_deg(target.rotation_deg, attacker.rotation_deg) <= BACKSTAB_DEGREE_RANGE
            and dist <= BACKSTAB_RANGE):
        return 2
    return 1


# --- legacy formulas, for neural-net features only ---------------------------

def legacy_distance(a, b) -> float:
    return math.hypot(abs(abs(a.loc_x) - abs(b.loc_x)), abs(abs(a.loc_y) - abs(b.loc_y)))


def legacy_angle_delta_from_front(player, enemy) -> float:
    dx = abs(abs(player.loc_x) - abs(enemy.loc_x))
    dy = abs(abs(player.loc_y) - abs(enemy.loc_y))
    r = player.rotation_deg
    try:
        if 45 < r < 135 or 225 < r < 315:
            return math.atan(dy / dx)
        return math.atan(dx / dy)
    except ZeroDivisionError:
        return math.pi / 2


def legacy_rotation_difference(player, enemy) -> float:
    return abs(player.rotation_deg - enemy.rotation_deg)
