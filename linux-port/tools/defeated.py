"""Everyone who has lost to [BOT]King, from records/fights.jsonl.

  python tools/defeated.py          # who lost to the bot (with Steam profile links)
  python tools/defeated.py --all    # every opponent, wins and losses
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "records" / "fights.jsonl"


def main() -> None:
    show_all = "--all" in sys.argv
    if not PATH.exists():
        print("no fights recorded yet (records/fights.jsonl)")
        return
    rows = [json.loads(line) for line in PATH.read_text().splitlines() if line.strip()]
    people: dict = defaultdict(lambda: {"won": 0, "lost": 0, "names": set(), "chars": set(), "last": ""})
    for r in rows:
        key = r.get("steam_id") or "unidentified"
        p = people[key]
        p[r["result"]] += 1
        if r.get("steam_name"):
            p["names"].add(r["steam_name"])
        if r.get("char_name"):
            p["chars"].add(r["char_name"])
        p["last"] = max(p["last"], r["time"])
        p["profile"] = r.get("profile") or ""
    wins = sum(r["result"] == "won" for r in rows)
    print(f"{len(rows)} fights: {wins} won, {len(rows) - wins} lost\n")
    order = sorted(people.items(), key=lambda kv: (-kv[1]["won"], kv[1]["last"]))
    title = "opponents" if show_all else "lost to [BOT]King"
    print(f"{title}:")
    for sid, p in order:
        if not show_all and not p["won"]:
            continue
        names = ", ".join(sorted(p["names"])) or "?"
        chars = f" (character: {', '.join(sorted(p['chars']))})" if p["chars"] else ""
        print(f"  {names}{chars}  beaten {p['won']}x, beat the bot {p['lost']}x, last {p['last']}")
        if sid != "unidentified":
            print(f"      Steam ID {sid}  {p['profile']}")


if __name__ == "__main__":
    main()
