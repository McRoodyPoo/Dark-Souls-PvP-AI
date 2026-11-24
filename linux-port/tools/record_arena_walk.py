"""Record your character's position while you walk the arena's edges (read-only).

  python tools/record_arena_walk.py          # Ctrl+C when done

Walk to each corner, roll into it a few times, and slide along the walls between them.
Writes recordings/arena_walk.json; tools/build_arena_map.py turns it into the walkable
area in arena.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ptde_ai.memory import ProcMem           # noqa: E402
from ptde_ai.process import find_game_pid    # noqa: E402
from ptde_ai.state import StateReader        # noqa: E402

OUT = ROOT / "recordings" / "arena_walk.json"


def main() -> None:
    reader = StateReader(ProcMem(find_game_pid()))
    center = json.loads((ROOT / "arena.json").read_text()).get("center")
    pts, last_print, last = [], 0.0, None
    print("recording your position; walk the corners and walls, Ctrl+C when done")
    try:
        while True:
            v = reader.player().values
            x, y, h = v.get("loc_x"), v.get("loc_y"), v.get("loc_h")
            if None not in (x, y, h) and (last is None or math.hypot(x - last[0], y - last[1]) > 0.02):
                pts.append([round(x, 3), round(y, 3), round(h, 3), round(time.time(), 2)])
                last = (x, y)
            if time.time() - last_print > 2 and last:
                d = math.hypot(last[0] - center["x"], last[1] - center["y"]) if center else float("nan")
                print(f"{len(pts):6d} points  at ({last[0]:.1f}, {last[1]:.1f})  {d:4.1f} m from centre",
                      flush=True)
                last_print = time.time()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    OUT.write_text(json.dumps(pts))
    print(f"\nsaved {len(pts)} points to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
