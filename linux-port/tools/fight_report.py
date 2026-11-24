"""Per-fight report from the recordings: who hurt whom, with what, and what the bot tried.

  python tools/fight_report.py                 # every recording
  python tools/fight_report.py <file>...       # specific ones

For each fight (bot-driven ticks only):
  damage dealt / taken, host backstabs on the bot and bot backstabs landed,
  what hit the bot (enemy anim type -> total) and what the bot was doing 0.3 s before,
  how its punishes / kicks / backstab attempts went (enemy HP drop or stagger after),
  time spent idle (no routine running) while the host was mid-attack.
"""
from __future__ import annotations

import collections
import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptde_ai.recorder import REC_DIR, load    # noqa: E402

R1 = 0x20


def dist(r):
    return math.hypot(r["p"]["loc_x"] - r["e"]["loc_x"], r["p"]["loc_y"] - r["e"]["loc_y"])


def report(path: str) -> None:
    rows = [r for r in load(path) if "mode" in r and r.get("mode") == "bot" and r["e"]]
    events = [r for r in load(path) if "event" in r]
    if len(rows) < 120:
        return
    secs = (rows[-1]["t"] - rows[0]["t"]) / 1000
    dealt = taken = 0
    hit_by = collections.Counter()
    doing_when_hit = collections.Counter()
    host_bs = bot_bs = 0
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        de = a["e"]["hp"] - b["e"]["hp"]
        dp = a["p"]["hp"] - b["p"]["hp"]
        if 0 < de < 2000:
            dealt += de
        if 0 < dp < 2000:
            taken += dp
            hit_by[b["e"]["animation_type"]] += dp
            before = next((r for r in reversed(rows[max(0, i - 80):i]) if b["t"] - r["t"] >= 300), a)
            doing_when_hit[f"{before.get('attack')}/{before.get('dodge')}"] += dp
        if b["e"]["animation_type"] == 225 and a["e"]["animation_type"] != 225:
            host_bs += 1
        if b["p"]["animation_type"] == 225 and a["p"]["animation_type"] != 225:
            bot_bs += 1
    # what came of the bot's own attacks
    tries = collections.defaultdict(lambda: [0, 0])     # kind -> [started, worked]
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        kind = b.get("attack")
        if kind in ("PUNISH", "KICK", "BACKSTAB") and a.get("attack") != kind:
            after = [r for r in rows[i:i + 200] if r["t"] - b["t"] < 1500]
            ok = (any(r["e"]["hp"] < b["e"]["hp"] for r in after) or
                  any(r["e_sub"] == 14 for r in after) or
                  any(r["p"]["animation_type"] == 225 for r in after))
            tries[kind][0] += 1
            tries[kind][1] += ok
    idle_mid_attack = sum(1 for r in rows if r.get("attack") == "NONE" and r.get("dodge") == "NONE"
                          and r["e_sub"] in (0, 1, 2, 11))
    hz = len(rows) / max(secs, 1)
    result = next((e["event"] for e in events if "WON" in e["event"] or "lost to" in e["event"]), "")
    who = next((e["event"] for e in events if e["event"].startswith("opponent:")), "")
    print(f"\n== {Path(path).name}  {secs:.0f} s  {result or '(no result logged)'}  {who}")
    print(f"   dealt {dealt}, took {taken}; backstabs: bot {bot_bs}, host {host_bs}")
    print(f"   hit by: " + ", ".join(f"anim {k}: {v}" for k, v in hit_by.most_common(5)))
    print(f"   bot was doing (0.3 s before each hit): " +
          ", ".join(f"{k}: {v}" for k, v in doing_when_hit.most_common(5)))
    print(f"   its attacks (started/connected): " +
          ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in tries.items()))
    print(f"   idle while the host was mid-attack: {idle_mid_attack / hz:.1f} s")


if __name__ == "__main__":
    for f in sys.argv[1:] or sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
        report(f)
