"""Add round wins the fight log missed to records/fights.jsonl.

Until 2026-11-24 the log only counted a win when the opponent DIED, but in the arena a round
ends at 1 HP for most players. This scans the recordings for opponents dropping to 1 HP
while the bot was alive and clearly landed the hit: its backstab animation playing, or one
of its attack routines running within 2.5 m, in the 1.5 s before.

  python tools/recover_wins.py          # show what would be added
  python tools/recover_wins.py --write  # add them (once: already-added ones are skipped)
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai.fightlog import PATH                      # noqa: E402
from ptde_ai.recorder import REC_DIR, load             # noqa: E402

CLOCK_OFFSET = time.time() - time.perf_counter()   # valid while the machine hasn't rebooted
BOT_ATTACKS = {"BACKSTAB", "KICK", "PUNISH", "RUN_BS", "GHOST_HIT", "POKE", "LINE_UP"}


def scan(path: str) -> list[dict]:
    rows = list(load(path))
    who = sid = None
    prev = None
    found = []
    for i, r in enumerate(rows):
        if "event" in r:
            s = r["event"]
            m = re.search(r"profiles/(\d+)", s)
            if s.startswith("opponent:") and m:
                who, sid = s[len("opponent: "):].split(" (character")[0].split(",")[0], int(m.group(1))
            elif s.startswith("new target") or "summoned" in s or "back in our world" in s:
                who = sid = None
                prev = None
            continue
        if r.get("mode") != "bot" or not r.get("e"):
            prev = None
            continue
        if (prev is not None and who and r["e"]["hp"] == 1 and prev["e"]["hp"] > 1
                and r["p"]["hp"] > 1 and r["t"] - prev["t"] < 200):
            back = [x for x in rows[max(0, i - 400):i] if "p" in x and r["t"] - x["t"] <= 1500]
            dist = math.hypot(r["p"]["loc_x"] - r["e"]["loc_x"], r["p"]["loc_y"] - r["e"]["loc_y"])
            ours = (any(x["p"]["animation_type"] == 225 for x in back)
                    or (dist < 2.5 and any(x.get("attack") in BOT_ATTACKS for x in back)))
            if ours:
                # recordings use the monotonic clock (perf_counter, ms since boot)
                when = dt.datetime.fromtimestamp(r["t"] / 1000 + CLOCK_OFFSET)
                found.append({"time": when.strftime("%Y-%m-%d %H:%M:%S"), "result": "won",
                              "steam_id": sid, "steam_name": who, "char_name": who,
                              "profile": f"https://steamcommunity.com/profiles/{sid}",
                              "bot_hp": r["p"]["hp"], "their_hp": 1, "secs": None,
                              "round_end": "1hp", "recovered_from": Path(path).name,
                              "rec_t": round(r["t"], 1)})
        prev = r
    return found


def main():
    existing, logged_wins = set(), []
    if PATH.exists():
        for line in PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("recovered_from"):
                    existing.add((row["recovered_from"], row.get("rec_t")))
                elif row["result"] == "won":
                    logged_wins.append((row["steam_id"], dt.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")))
    new = []
    for f in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
        for row in scan(f):
            when = dt.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            dup = any(sid == row["steam_id"] and abs((w - when).total_seconds()) < 60
                      for sid, w in logged_wins)
            if (row["recovered_from"], row["rec_t"]) not in existing and not dup:
                new.append(row)
    for row in new:
        print(f"  {row['time']}  won vs {row['steam_name']} (took them to 1 HP; bot at {row['bot_hp']})")
    print(f"{len(new)} missed 1 HP wins")
    if "--write" in sys.argv and new:
        with PATH.open("a") as f:
            for row in new:
                f.write(json.dumps(row) + "\n")
        print(f"added to {PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
