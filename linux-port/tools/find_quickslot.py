"""Find where the game stores the selected quick-item slot.

Run it with two quick items equipped and the game window focused. It snapshots all
writable game memory, asks you to press d-pad down (cycle quick item) three times, and
keeps the addresses whose value flips in step with your presses (A, B, A, B).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptde_ai.memory import ProcMem      # noqa: E402
from ptde_ai.process import find_game_pid  # noqa: E402

pid = find_game_pid()
mem = ProcMem(pid)


def regions():
    out = []
    for line in open(f"/proc/{pid}/maps"):
        addr, perms = line.split()[:2]
        lo, hi = (int(x, 16) for x in addr.split("-"))
        if perms.startswith("rw") and hi <= 1 << 32:
            out.append((lo, hi))
    return out


def snap(regs):
    data = {}
    for lo, hi in regs:
        try:
            data[lo] = np.frombuffer(mem.read(lo, hi - lo), dtype=np.uint8)
        except OSError:
            pass
    return data


def countdown(msg, secs=3):
    for s in range(secs, 0, -1):
        print(f"  {msg} in {s}...", flush=True)
        time.sleep(1)


regs = regions()
PRESSES = 4
print(f"{len(regs)} regions. Keep the GAME focused; watch this tab.")
print("Press d-pad down ONLY when told; don't touch anything otherwise.\n")
countdown("snapshot A")
a = snap(regs)
cand = {}   # region -> (indices, last value)


def quiet_filter(secs=3.0):
    """Drop candidates that change while nobody is pressing anything."""
    t_end = time.time() + secs
    while time.time() < t_end:
        for lo in list(cand):
            idx, last = cand[lo]
            try:
                arr = np.frombuffer(mem.read(lo + int(idx[0]), int(idx[-1] - idx[0]) + 1), dtype=np.uint8)
            except OSError:
                del cand[lo]; continue
            keep = arr[idx - idx[0]] == last
            if keep.any():
                cand[lo] = (idx[keep], last[keep])
            else:
                del cand[lo]
        time.sleep(0.25)


for step in range(PRESSES):
    print(f"\n>>> PRESS D-PAD DOWN ONCE NOW ({step + 1}/{PRESSES}) <<<", flush=True)
    time.sleep(3)
    if step == 0:
        b = snap(regs)
        for lo, arr in a.items():
            arr2 = b.get(lo)
            if arr2 is None or len(arr2) != len(arr):
                continue
            idx = np.nonzero((arr != arr2) & (arr < 5) & (arr2 < 5))[0]
            if len(idx):
                cand[lo] = (idx, arr2[idx].copy())
        del a, b
    else:
        for lo in list(cand):
            idx, last = cand[lo]
            try:
                arr = np.frombuffer(mem.read(lo + int(idx[0]), int(idx[-1] - idx[0]) + 1), dtype=np.uint8)
            except OSError:
                del cand[lo]; continue
            v = arr[idx - idx[0]]
            keep = v != last                     # must change on every press
            if keep.any():
                cand[lo] = (idx[keep], v[keep].copy())
            else:
                del cand[lo]
    print("  hands off...", flush=True)
    quiet_filter()
    print(f"  candidates left: {sum(len(i) for i, _ in cand.values())}", flush=True)

print("\nAddresses that changed on every press and never on their own:")
out = []
for lo, (idx, last) in cand.items():
    for k, i in enumerate(idx):
        out.append(f"{lo + int(i):#010x} now={int(last[k])}")
print("\n".join(out[:80]))
Path(__file__).with_name("quickslot_candidates.txt").write_text("\n".join(out))
