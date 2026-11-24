"""Handoff tests: a simulated Bluetooth Xbox controller (uinput) and the mode switch.

  python tests/test_handoff.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evdev import AbsInfo, UInput, ecodes as e                      # noqa: E402

from ptde_ai.gamepad import CROSS, L2, L3, MIDDLE, R1, R3, SELECT, SQUARE, TRIANGLE, DLEFT  # noqa: E402
from ptde_ai.handoff import BOT, MANUAL, Handoff                     # noqa: E402
from ptde_ai.realpad import PadState, RealPad                        # noqa: E402


def fake_bt_xbox():
    """Layout of an Xbox Wireless controller over Bluetooth with hid-microsoft:
    right stick on Z/RZ, triggers on BRAKE/GAS (0..1023), View button as KEY_BACK."""
    stick = AbsInfo(32768, 0, 65535, 16, 128, 0)
    trig = AbsInfo(0, 0, 1023, 0, 0, 0)
    hat = AbsInfo(0, -1, 1, 0, 0, 0)
    caps = {
        e.EV_KEY: [e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST, e.BTN_TL, e.BTN_TR,
                   e.BTN_THUMBL, e.BTN_THUMBR, e.BTN_START, e.BTN_MODE, e.KEY_BACK],
        e.EV_ABS: [(e.ABS_X, stick), (e.ABS_Y, stick), (e.ABS_Z, stick), (e.ABS_RZ, stick),
                   (e.ABS_BRAKE, trig), (e.ABS_GAS, trig), (e.ABS_HAT0X, hat), (e.ABS_HAT0Y, hat)],
    }
    return UInput(caps, name="Xbox Wireless Controller", vendor=0x045E, product=0x0B13,
                  phys="test-real-controller")


def wait_for(cond, timeout=3.0):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_reads_bluetooth_layout():
    with fake_bt_xbox() as ui:
        rp = RealPad()
        try:
            assert wait_for(lambda: rp.connected), "real controller not found"
            time.sleep(0.2)
            ui.write(e.EV_ABS, e.ABS_X, 65535)        # left stick full right
            ui.write(e.EV_ABS, e.ABS_RZ, 0)           # right stick full up
            ui.write(e.EV_ABS, e.ABS_BRAKE, 1023)     # left trigger
            ui.write(e.EV_KEY, e.BTN_SOUTH, 1)
            ui.write(e.EV_KEY, e.BTN_NORTH, 1)        # physical X (left face)
            ui.write(e.EV_KEY, e.KEY_BACK, 1)
            ui.write(e.EV_ABS, e.ABS_HAT0X, -1)
            ui.syn()
            assert wait_for(lambda: rp.snapshot().lx > 0.9)
            s = rp.snapshot()
            assert s.ry < -0.9, s
            b = s.buttons()
            assert b & CROSS and b & SQUARE and b & SELECT and b & L2, hex(b)
            assert not b & TRIANGLE
            assert s.hat() == DLEFT

            class Pad: pass
            pad = Pad()
            s.apply_to(pad)
            assert pad.lx > 32000 and pad.ry < 100 and pad.buttons == b
        finally:
            rp.close()


def test_survives_unplug():
    ui = fake_bt_xbox()
    rp = RealPad()
    try:
        assert wait_for(lambda: rp.connected)
        ui.close()
        assert wait_for(lambda: not rp.connected), "didn't notice the unplug"
        with fake_bt_xbox():
            assert wait_for(lambda: rp.connected, 4.0), "didn't pick up the replug"
    finally:
        rp.close()


def state(buttons=0, lx=0.0):
    keymap = {L3: e.BTN_THUMBL, R3: e.BTN_THUMBR, CROSS: e.BTN_SOUTH, R1: e.BTN_TR}
    return PadState(lx=lx, keys={k for bit, k in keymap.items() if buttons & bit})


def test_combo_hands_to_bot_and_any_input_takes_back():
    h = Handoff()
    assert h.mode == MANUAL
    assert h.update(state(L3 | R3), 0) == BOT             # one click of both: no hold needed
    assert h.update(state(L3 | R3), 100) is None          # still holding
    assert h.update(state(lx=0.8), 200) is None           # stick nudge from the click: grace
    assert h.update(state(), 800) is None and h.mode == BOT
    assert h.update(state(lx=0.2), 900) is None           # small stick drift ignored
    assert h.update(state(CROSS), 1000) == MANUAL         # press anything -> you
    assert h.mode == MANUAL


def test_sloppy_combo_still_counts():
    """User 2026-11-23: the 0.5 s simultaneous hold was too tight."""
    h = Handoff()
    assert h.update(state(L3), 0) is None                 # one stick...
    assert h.update(state(), 100) is None                 # (let go: flicker)
    assert h.update(state(R3), 350) == BOT                # ...then the other, 0.35 s later
    h2 = Handoff()
    h2.update(state(R3), 0); h2.update(state(), 50)
    assert h2.update(state(L3), 900) is None and h2.mode == MANUAL   # too far apart


def test_stick_takes_back():
    h = Handoff()
    h.update(state(L3 | R3), 0); h.update(state(), 100)
    assert h.update(state(lx=0.8), 300) is None           # inside the grace period
    assert h.update(state(lx=0.8), 800) == MANUAL


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as ex:
            failed += 1
            print(f"FAIL  {name}: {type(ex).__name__}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
