"""Check each Linux building block against the live game.

  python scripts/probe.py                    # find game, read memory once
  python scripts/probe.py --watch            # live view of both characters (Ctrl+C)
  python scripts/probe.py --pad              # create the virtual pad and leave it plugged in
  python scripts/probe.py --calibrate-steering   # in a safe spot: walks you fwd/right, saves calibration.json

For the memory check, be in a duel (or summon a phantom) so the opponent exists.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptde_ai import offsets as O                       # noqa: E402
from ptde_ai.memory import ProcMem                     # noqa: E402
from ptde_ai.process import find_game_pid              # noqa: E402
from ptde_ai.state import StateReader, distance        # noqa: E402

KEY_FIELDS = ["hp", "stamina", "loc_x", "loc_y", "rotation", "animation_type",
              "anim_id", "anim_id2", "anim_timer", "r_weapon", "l_weapon"]


def open_game():
    pid = find_game_pid()
    try:
        mem = ProcMem(pid)
        mem.u32(O.MODULE_BASE)  # MZ header page should be readable
    except PermissionError:
        sys.exit(f"Found DARKSOULS.exe (pid {pid}) but cannot read its memory.\n"
                 "kernel.yama.ptrace_scope=1 is blocking it. Temporary fix:\n"
                 "  sudo sysctl kernel.yama.ptrace_scope=0")
    return pid, mem


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def dump_once(reader: StateReader) -> None:
    p, en, cam = reader.player(), reader.enemy(), reader.camera()
    for label, snap in (("PLAYER (you / the AI)", p), ("ENEMY (opponent)", en), ("CAMERA", cam)):
        print(f"\n== {label}")
        for k, v in snap.values.items():
            print(f"  {k:16} {fmt(v)}")
        for k, err in snap.errors.items():
            print(f"  {k:16} <unreadable: {err}>")
    if not p.errors and not en.errors:
        print(f"\ndistance: {distance(p, en):.2f}")


def watch(reader: StateReader) -> None:
    while True:
        p, en, cam = reader.player(), reader.enemy(), reader.camera()
        row = lambda s: " ".join(f"{k}={fmt(s.values.get(k, '-'))}" for k in KEY_FIELDS if k in s.values)
        d = f"{distance(p, en):.2f}" if {"loc_x", "loc_y"} <= p.values.keys() & en.values.keys() else "-"
        print(f"\033[2J\033[H"
              f"YOU   {row(p)}\n\nENEMY {row(en)}\n\n"
              f"cam yaw={fmt(cam.values.get('yaw', '-'))}  dist={d}", flush=True)
        time.sleep(0.1)


def make_pad():
    from ptde_ai.gamepad import VirtualPad
    pad = VirtualPad()
    print(f"virtual pad up at {pad.ui.device.path}; check that the game reacts. Ctrl+C to unplug.")
    return pad


def calibrate(reader: StateReader) -> None:
    from ptde_ai.gamepad import MIDDLE, XRIGHT, YTOP
    from ptde_ai.steer import Steering, heading, wrap

    print("Stand somewhere open with ~5m free in front and to the right. Starting in 3s...")
    with make_pad() as pad:
        time.sleep(3)

        def push(lx, ly, secs=0.8):
            p0, cam = reader.player(), reader.camera()
            pad.reset(); pad.lx, pad.ly = lx, ly; pad.send()
            time.sleep(secs)
            pad.reset(); pad.send()
            time.sleep(0.5)
            p1 = reader.player()
            dx, dy = p1.loc_x - p0.loc_x, p1.loc_y - p0.loc_y
            if math.hypot(dx, dy) < 0.3:
                sys.exit("Character barely moved; is the game reading the virtual pad?")
            return heading(dx, dy), cam.yaw

        h_fwd, yaw = push(MIDDLE, YTOP)
        h_right, _ = push(XRIGHT, MIDDLE)

    turn = wrap(h_right - h_fwd)
    s = Steering(offset=wrap(h_fwd - yaw), sign=1.0 if turn > 0 else -1.0)
    s.save()
    print(f"forward->right turned {math.degrees(turn):+.0f} deg (expect about +/-90)")
    print(f"saved calibration: offset={s.offset:.3f} rad, sign={s.sign:+.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--pad", action="store_true")
    ap.add_argument("--calibrate-steering", action="store_true")
    a = ap.parse_args()

    if a.pad:
        pad = make_pad()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pad.close()
        return

    pid, mem = open_game()
    lo, hi = mem.module_range("DARKSOULS.exe")
    print(f"DARKSOULS.exe pid {pid}, image {lo:#x}-{hi:#x}")
    reader = StateReader(mem)
    try:
        if a.calibrate_steering:
            calibrate(reader)
        elif a.watch:
            watch(reader)
        else:
            dump_once(reader)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
