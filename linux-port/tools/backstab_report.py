"""List every backstab attempt in the recordings with its geometry and what happened.

  python tools/backstab_report.py            # all recordings
  python tools/backstab_report.py <file>...  # specific ones

A real backstab shows up as a big enemy HP drop with a special animation id on both
characters; a plain R1 shows the bot's normal attack id (e.g. 29x000) and ~200-300 damage.
Use this to pin down the game's real backstab window (geometry.BS_ATTEMPT_*).
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptde_ai import geometry as G               # noqa: E402
from ptde_ai.recorder import REC_DIR, load      # noqa: E402


class _C:
    def __init__(self, d):
        self.loc_x, self.loc_y = d["loc_x"], d["loc_y"]
        self.rotation_raw = d["rotation"]
        self.rotation_deg = math.degrees(d["rotation"] + math.pi)


def report(path: str) -> None:
    ticks = [r for r in load(path) if "mode" in r]
    for i, r in enumerate(ticks):
        if not (r.get("attack") == "BACKSTAB" and r["pad"][4] & 0x20):
            continue
        if i and ticks[i - 1].get("attack") == "BACKSTAB" and ticks[i - 1]["pad"][4] & 0x20:
            continue
        p, e = _C(r["p"]), _C(r["e"])
        fx, fy = G.facing_vector(e)
        relx, rely = p.loc_x - e.loc_x, p.loc_y - e.loc_y
        pos = math.degrees(math.atan2(abs(relx * fy - rely * fx), -(relx * fx + rely * fy)))
        nxt = ticks[i:i + 240]                     # backstab damage lands ~1.2 s later
        bot_ids, en_ids = [], []
        for q in nxt:
            if not bot_ids or bot_ids[-1] != q["p"]["anim_id"]:
                bot_ids.append(q["p"]["anim_id"])
            if not en_ids or en_ids[-1] != q["e"]["anim_id"]:
                en_ids.append(q["e"]["anim_id"])
        dmg = r["e"]["hp"] - min(q["e"]["hp"] for q in nxt)
        landed = any(q["p"]["animation_type"] == 225 for q in nxt[:30])   # 225 = backstab animation
        print(f"{Path(path).name} t={r['t'] / 1000:.2f}  dist {G.distance(p, e):.2f} m  "
              f"pos {pos:3.0f} deg  facing {G.angle_diff_deg(e.rotation_deg, p.rotation_deg):3.0f} deg  "
              f"their state {r['e_sub']:2d} (anim {r['e']['animation_type']})  -> {'BACKSTAB' if landed else 'miss    '} dmg {dmg:4d}  "
              f"bot ids {bot_ids[:4]}  their ids {en_ids[:4]}")


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(str(REC_DIR / "*.jsonl.gz")))
    for f in files:
        report(f)
