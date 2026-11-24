"""Seed models/hit_damage.json from fight recordings: every HP drop of the bot while the
opponent is in an attack animation is attributed to that (weapon class, attack), and
how far into the attack it landed is kept for attacks missing from the timing table."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptde_ai import tables as T                  # noqa: E402
from ptde_ai.character import current_attack     # noqa: E402
from ptde_ai.hitmemory import HitMemory          # noqa: E402
from ptde_ai.recorder import REC_DIR, load       # noqa: E402

mem = HitMemory()
n = 0
for path in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
    prev = None
    for r in load(path):
        if "mode" not in r:
            prev = None
            continue
        if prev and 0 < r["p"]["hp"] < prev["p"]["hp"] and prev["e"]["animation_type"] in T.ATTACK_ANIM:
            mem.record(prev["e"]["r_weapon"], prev["e"]["animation_type"], prev["p"]["hp"] - r["p"]["hp"])
            d = ((prev["p"]["loc_x"] - prev["e"]["loc_x"]) ** 2 + (prev["p"]["loc_y"] - prev["e"]["loc_y"]) ** 2) ** 0.5
            if d < 6:
                mem.record_reach(prev["e"]["r_weapon"], d)
            aid, key = current_attack(prev["e"])
            if key and aid not in T.DODGE_TIMINGS:
                mem.record_hit_time(aid, prev["e"].get(key, 0.0))
            n += 1
        prev = r
mem.maybe_save(force=True)
print(f"{n} hits from recordings -> {len(mem.table)} attacks: {mem.table}")
