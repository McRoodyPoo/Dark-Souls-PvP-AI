"""Turn the arena walk (tools/record_arena_walk.py) into the walkable area in arena.json.

  python tools/build_arena_map.py [recordings/arena_walk.json]

The walk hugs the walls and rolls into the corners, so the convex hull of the trace is
the area a character's centre can reach. Nearly collinear hull points are dropped.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ptde_ai.arena_map import ArenaMap   # noqa: E402


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]          # counter-clockwise


def simplify(poly, tol=0.15):
    """Drop hull points that sit within tol metres of the line through their neighbours."""
    changed = True
    while changed and len(poly) > 3:
        changed = False
        for i in range(len(poly)):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
            ac = math.dist(a, c)
            if ac == 0:
                continue
            d = abs((c[0] - a[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (c[1] - a[1])) / ac
            if d < tol:
                poly.pop(i)
                changed = True
                break
    return poly


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "recordings" / "arena_walk.json"
    walk = json.loads(src.read_text())
    poly = simplify(hull([(p[0], p[1]) for p in walk]))
    hs = sorted(p[2] for p in walk)
    arena_path = ROOT / "arena.json"
    arena = json.loads(arena_path.read_text())
    arena["walkable"] = {"polygon": [[round(x, 3), round(y, 3)] for x, y in poly],
                         "h_min": round(hs[0], 2), "h_max": round(hs[-1], 2)}
    arena_path.write_text(json.dumps(arena, indent=1))
    m = ArenaMap.from_arena(arena)
    print(f"{len(walk)} points -> {len(poly)}-point walkable polygon, area {m.area:.1f} m^2, "
          f"height {hs[0]:.2f}..{hs[-1]:.2f}")
    for i, (x, y) in enumerate(poly):
        nx, ny = poly[(i + 1) % len(poly)]
        print(f"  ({x:7.2f}, {y:7.2f}) -> wall {math.dist((x, y), (nx, ny)):5.2f} m")
    c = arena.get("center")
    if c:
        print(f"centre is {m.wall_distance(c['x'], c['y']):.2f} m from the nearest wall")


if __name__ == "__main__":
    main()
