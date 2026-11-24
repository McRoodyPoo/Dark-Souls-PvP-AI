"""Replay every recording through the current brain and flag regressions.

  python tools/replay_check.py

Checks: crashes, routine-restart bursts (same routine started 5+ times in 0.1 s),
R1 within 1.5 s of a wake-up. Run after every brain change, before it goes live.
"""
import glob, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from ptde_ai.recorder import REC_DIR, load               # noqa: E402
from ptde_ai.state import Snapshot                       # noqa: E402
from ptde_ai.brain import Brain, Config                  # noqa: E402
from ptde_ai.character import CharacterTracker           # noqa: E402
from ptde_ai.hitmemory import HitMemory                  # noqa: E402
from ptde_ai.steer import Steering                       # noqa: E402
from ptde_ai.nets import MODEL_DIR, MLP                  # noqa: E402
from ptde_ai.arena_map import ArenaMap                   # noqa: E402
from test_brain import Pad                               # noqa: E402

net = lambda n: MLP.load(MODEL_DIR / f"{n}.npz")
tot = {"wakeups": 0, "wake_r1": 0, "bursts": 0, "crashes": 0, "starts": 0, "secs": 0.0}
for f in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
    rows = [r for r in load(f) if "mode" in r and r["e"]]
    if len(rows) < 100:
        continue
    hits = HitMemory(path=None); hits.table = json.load(open(ROOT / "models/hit_damage.json"))
    b = Brain(Steering(0.0, 1.0), net("defense"), net("attack"), Config(), rng=random.Random(0), hits=hits)
    b.arena = ArenaMap.from_arena(json.load(open(ROOT / "arena.json")))
    me, them = CharacterTracker(True, 2.5), CharacterTracker(False, 6.0); them.timings = hits
    starts, wake_r1, wakes, last_woke = [], 0, 0, -1e9
    try:
        for r in rows:
            e = them.update(Snapshot(values=r["e"]), b.history, r["t"])
            p = me.update(Snapshot(values=r["p"]), b.history, r["t"])
            pad = Pad(); b.tick(p, e, 0.0, r["t"], pad)
            if b._woke_t != last_woke:
                last_woke, wakes = b._woke_t, wakes + 1
            if pad.buttons & 0x20 and r["t"] - b._woke_t < 1500:
                wake_r1 += 1
            for line in b.log:
                m = re.match(r"start (?:attack|dodge) ([A-Z_]+)", line)
                if m:
                    starts.append((r["t"], m.group(1)))
            b.log.clear()
    except Exception as ex:                                # noqa: BLE001
        print(Path(f).name, "CRASH", repr(ex)); tot["crashes"] += 1; continue
    bursts = sum(1 for i in range(len(starts))
                 if sum(1 for x in starts[i:i + 8] if x[1] == starts[i][1] and x[0] - starts[i][0] < 100) >= 5)
    tot["wakeups"] += wakes; tot["wake_r1"] += wake_r1; tot["bursts"] += bursts
    tot["starts"] += len(starts); tot["secs"] += (rows[-1]["t"] - rows[0]["t"]) / 1000
    if bursts or wake_r1:
        print(Path(f).name, "bursts", bursts, "wake-up R1 ticks", wake_r1)
print(f"{tot['crashes']} crashes | {tot['bursts']} restart bursts | {tot['wake_r1']} R1 ticks within 1.5 s "
      f"of {tot['wakeups']} wake-ups | {tot['starts'] / max(tot['secs'], 1):.2f} routine starts/s")
sys.exit(1 if tot["crashes"] or tot["bursts"] or tot["wake_r1"] else 0)
