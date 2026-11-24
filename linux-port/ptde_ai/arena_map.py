"""Walkable area of the arena, from the user's walk along its walls (2026-11-23).

The polygon is where a character's CENTRE can go (the walk hugged the walls and rolled
into the corners), so anything outside it is wall. Used by the brain to:
  * treat a wall right behind the bot as cover: nobody can stand there to backstab it;
  * not try to get behind an enemy whose back is against a wall;
  * not roll or back off into a wall.
Only applies while we're actually in this arena (hosts may summon the bot elsewhere).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import geometry as G

BACK_COVER_M = 0.6        # wall within this behind us: no room for a backstabber
NEAR_MARGIN_M = 1.5       # how far outside the polygon still counts as "in the arena"
HEIGHT_TOL_M = 3.0


@dataclass
class ArenaMap:
    polygon: list[tuple[float, float]]
    h_min: float
    h_max: float

    @classmethod
    def from_arena(cls, arena: dict) -> "ArenaMap | None":
        w = arena.get("walkable") if arena else None
        if not w or len(w.get("polygon", [])) < 3:
            return None
        return cls([tuple(p) for p in w["polygon"]], w["h_min"], w["h_max"])

    @property
    def area(self) -> float:
        a = 0.0
        for (x1, y1), (x2, y2) in zip(self.polygon, self.polygon[1:] + self.polygon[:1]):
            a += x1 * y2 - x2 * y1
        return abs(a) / 2

    # ------------------------------------------------------------ geometry
    def contains(self, x: float, y: float) -> bool:
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def wall_distance(self, x: float, y: float) -> float:
        """Distance to the nearest wall: positive inside the walkable area, negative outside."""
        best = math.inf
        n = len(self.polygon)
        for i in range(n):
            (x1, y1), (x2, y2) = self.polygon[i], self.polygon[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            u = 0.0 if L == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L))
            best = min(best, math.hypot(x - (x1 + u * dx), y - (y1 + u * dy)))
        return best if self.contains(x, y) else -best

    def free_run(self, x: float, y: float, heading: float, limit: float = 4.0) -> float:
        """How far we can go along a world heading (atan2(dx, dy)) before hitting a wall."""
        sx, sy = math.sin(heading), math.cos(heading)
        d, step = 0.0, 0.1
        while d < limit:
            if not self.contains(x + sx * (d + step), y + sy * (d + step)):
                return d
            d += step
        return limit

    @property
    def center(self) -> tuple[float, float]:
        n = len(self.polygon)
        return (sum(x for x, _ in self.polygon) / n, sum(y for _, y in self.polygon) / n)

    # --------------------------------------------------------------- tactics
    def here(self, c) -> bool:
        """Is this character in this arena (not summoned to some other map)?"""
        h = c.raw.get("loc_h") if hasattr(c, "raw") else None
        if h is not None and not (self.h_min - HEIGHT_TOL_M <= h <= self.h_max + HEIGHT_TOL_M):
            return False
        return self.wall_distance(c.loc_x, c.loc_y) > -NEAR_MARGIN_M

    def back_covered(self, c, depth: float = BACK_COVER_M) -> bool:
        """A wall directly behind c: nobody can stand in the backstab spot."""
        fx, fy = G.facing_vector(c)
        return not self.contains(c.loc_x - fx * depth, c.loc_y - fy * depth)

    def behind_reachable(self, target, depth: float = 0.8) -> bool:
        """Can anyone stand `depth` metres behind target (where a backstab comes from)?"""
        return not self.back_covered(target, depth)
