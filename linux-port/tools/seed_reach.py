"""Seed per-attack reach (models/hit_damage.json "reach0:<weapon class>:<anim type>") from the
recorded fights: for every hit on the bot, how far away that attack started.

  python tools/seed_reach.py

Same rule as the live brain (Brain.tick): the distance when the attacker's animation type
changed into the attack, recorded when that attack hits us. Only ever raises a value.
"""
from __future__ import annotations

import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai import tables as T                       # noqa: E402
from ptde_ai.brain import LUNGE_ANIM, PROJECTILE_ANIM  # noqa: E402
from ptde_ai.hitmemory import HitMemory                # noqa: E402
from ptde_ai.recorder import REC_DIR, load             # noqa: E402


def main():
    hits = HitMemory()
    found = defaultdict(float)
    for f in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
        last_anim, start_d, last_hp = 0, None, None
        for r in load(f):
            if not r.get("e") or not r.get("p"):
                continue
            p, e = r["p"], r["e"]
            d = math.hypot(p["loc_x"] - e["loc_x"], p["loc_y"] - e["loc_y"])
            at = e["animation_type"]
            if at != last_anim:
                if at in T.ATTACK_ANIM or at in LUNGE_ANIM:
                    start_d = d
                last_anim = at
            if (last_hp is not None and 0 < p["hp"] < last_hp and at in T.ATTACK_ANIM
                    and at not in PROJECTILE_ANIM and start_d is not None and 0.3 < start_d < 9.0
                    and last_hp - p["hp"] < 2000):
                k = (e["r_weapon"] // 1000, at, e.get("anim_id", 0))
                found[k] = max(found[k], start_d)
            last_hp = p["hp"]
    for (w, at, aid), dist in found.items():
        hits.record_reach_start(w * 1000, at, dist, aid)
    hits.maybe_save(force=True)
    names = {a.value: a.name for a in T.Anim}
    print(f"{len(found)} weapon/attack reaches from the recordings; farthest starts:")
    for (w, at, aid), dist in sorted(found.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  weapon class {w:5}  {names.get(at, at):22} (attack {aid}) {dist:.2f} m")


if __name__ == "__main__":
    main()
