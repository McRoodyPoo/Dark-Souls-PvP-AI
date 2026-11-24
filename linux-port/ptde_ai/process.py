"""Find the PTDE process running under Proton/Wine.

Under Wine the Windows .exe is an ordinary Linux process; its /proc/<pid>/comm is
the exe name (truncated to 15 chars) and its address space uses the same virtual
addresses the game sees, so Windows-style pointer chains work unchanged.
"""
from __future__ import annotations

import os

GAME_EXE = "DARKSOULS.exe"


def find_game_pid(exe: str = GAME_EXE) -> int:
    want = exe.lower()[:15]
    hits = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm") as f:
                comm = f.read().strip().lower()
        except OSError:
            continue
        if comm == want:
            hits.append(int(entry))
    if not hits:
        raise RuntimeError(f"{exe} is not running (start PTDE through Steam first)")
    if len(hits) > 1:
        # The real game process is the one with the most mapped memory.
        hits.sort(key=_rss, reverse=True)
    return hits[0]


def _rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1])
    except OSError:
        return 0
