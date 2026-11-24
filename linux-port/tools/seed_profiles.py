"""Seed records/profiles.json from the recorded fights.

  python tools/seed_profiles.py           # writes records/profiles.json (refuses to overwrite)
  python tools/seed_profiles.py --force   # rebuild it from scratch

Everything recorded before per-opponent styles existed was the bot's default play, so those
exchanges are scored for the "balanced" neutral style and the knockdowns for the "mix"
wake-up style. Opponents come from the "opponent: ... /profiles/<SteamID>" events.
"""
from __future__ import annotations

import glob
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai.adapt import ALL, PATH, Adapter, Profiles   # noqa: E402
from ptde_ai.recorder import REC_DIR, load              # noqa: E402



def seed(profiles: Profiles) -> dict:
    ad = Adapter(profiles)
    profiles.choose = lambda sid, dim, setup=None: "balanced" if dim == "neutral" else "mix"
    counts: dict = {}
    for f in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
        ad.abort()
        ad.sid, ad.name = None, "?"
        last_e, ev_new_target = None, False
        for r in load(f):
            if "event" in r:
                ev = r["event"]
                if ev.startswith("new target"):
                    ev_new_target = True
                m = re.search(r"profiles/(\d+)", ev)
                if ev.startswith("opponent:") and m:
                    name = ev[len("opponent: "):].split(" (character")[0].split(",")[0]
                    if ad.sid != m.group(1):
                        ad.set_opponent(m.group(1), name)
                elif "summoned" in ev or "back in our world" in ev:
                    ad.abort()
                    ad.sid, ad.name = None, "?"
                continue
            if r.get("mode") != "bot" or not r.get("e"):
                continue
            p, e = r["p"], r["e"]
            # a different character in the target slot: a 2 m+ jump between consecutive ticks
            # (recordings pause when nobody's fightable, so ignore jumps across gaps)
            jump = (last_e is not None and r["t"] - last_e[2] < 300
                    and math.hypot(e["loc_x"] - last_e[0], e["loc_y"] - last_e[1]) > 2.0)
            last_e = (e["loc_x"], e["loc_y"], r["t"])
            if jump or ev_new_target:
                ad.abort()                      # target switch: not this opponent any more
                ad.sid, ad.name = None, "?"     # until the next "opponent:" event names them
                ev_new_target = False
            down = (p.get("anim_id3", 0) > 0 and p["animation_type"] != 225) or r.get("p_sub") == 14
            dist = ((p["loc_x"] - e["loc_x"]) ** 2 + (p["loc_y"] - e["loc_y"]) ** 2) ** 0.5
            for line in ad.update(r["t"], SimpleNamespace(hp=p["hp"]), SimpleNamespace(hp=e["hp"], r_weapon=e.get("r_weapon", 0),
                                                                l_weapon=e.get("l_weapon", 0)),
                                  dist, down):
                key = ad.name if ad.sid else "(unidentified)"
                kind = "exchanges" if line.startswith("exchange") else "wakeups"
                counts.setdefault(key, {"exchanges": 0, "wakeups": 0})[kind] += 1
    return counts


def main():
    if PATH.exists() and "--force" not in sys.argv:
        print(f"{PATH} exists; use --force to rebuild it")
        return
    profiles = Profiles(path=None)
    counts = seed(profiles)
    profiles.path = PATH
    profiles._dirty_t = 0.0
    profiles.save(force=True)
    for who, c in sorted(counts.items(), key=lambda kv: -kv[1]["exchanges"]):
        print(f"  {who:24} {c['exchanges']:4} exchanges, {c['wakeups']:3} wake-ups")
    for sid, prof in profiles.data.items():
        if sid == ALL or "neutral" not in prof:
            continue
        w, l = prof["neutral"]["balanced"]
        ww, wl = prof.get("wakeup", {}).get("mix", [0, 0])
        if w + l >= 5:
            print(f"  {prof.get('name', sid):24} default play scored {w / (w + l):.0%} of exchanges' "
                  f"damage; wake-ups safe {ww / max(1e-9, ww + wl):.0%}")
    w, l = profiles.data[ALL]["neutral"]["balanced"]
    print(f"everyone: default play scored {w / (w + l):.0%}; wrote {PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
