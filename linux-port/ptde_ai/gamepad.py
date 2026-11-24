"""Virtual Xbox 360 pad over Linux uinput. Replaces metal-crow's vJoy driver.

Proton/SDL sees a device with Microsoft's 360 vendor/product ids and maps it to
XInput, so the game treats it as a normal controller.

The API mirrors metal-crow's vJoy report so the ported logic reads the same:
  pad.buttons |= R1; pad.lx = XRIGHT; pad.send()
Axes are 1..32768 with MIDDLE = 16384 like vJoy, converted on send().
PS-style names map to 360 buttons by position (square = X, cross = A, ...).
"""
from __future__ import annotations

import time

from evdev import AbsInfo, UInput, ecodes as e

# vJoy-compatible axis range (HelperUtil.h)
XLEFT = YTOP = 1
XRIGHT = YBOTTOM = 32768
MIDDLE = 16384

# Button bits, same values as metal-crow's HelperUtil.h
SQUARE, TRIANGLE, CROSS, CIRCLE = 0x1, 0x2, 0x4, 0x8
L1, R1, L2, R2 = 0x10, 0x20, 0x40, 0x80
L3, R3 = 0x100, 0x200        # stick clicks (R3 = lock-on in PTDE)
SELECT, START = 0x400, 0x800

# D-pad (vJoy discrete POV values)
DUP, DRIGHT, DDOWN, DLEFT, DCENTER = 0, 1, 2, 3, -1

_BTN = {  # bit -> evdev key (xpad layout: BTN_X=307 is the left face button)
    SQUARE: e.BTN_X, TRIANGLE: e.BTN_Y, CROSS: e.BTN_A, CIRCLE: e.BTN_B,
    L1: e.BTN_TL, R1: e.BTN_TR, L3: e.BTN_THUMBL, R3: e.BTN_THUMBR,
    SELECT: e.BTN_SELECT, START: e.BTN_START,
}
_HAT = {DCENTER: (0, 0), DUP: (0, -1), DDOWN: (0, 1), DLEFT: (-1, 0), DRIGHT: (1, 0)}


# Verified 2026-11-22 with an XInput probe inside the game's prefix: Wine exposes this
# pad as an XInput controller, but slots go in connection order and Dark Souls only
# reads slot 0. The pad must connect before any real controller (see realpad.py).
DEFAULT_NAME = "PTDE Bot Pad"
DEFAULT_VENDOR = 0x045E
DEFAULT_PRODUCT = 0x028E


def _stick(v: int) -> int:
    """vJoy 1..32768 -> evdev -32768..32767."""
    return max(-32768, min(32767, int((v - MIDDLE) * 2)))


class VirtualPad:
    def __init__(self, name: str = DEFAULT_NAME, vendor: int = DEFAULT_VENDOR,
                 product: int = DEFAULT_PRODUCT):
        stick = AbsInfo(0, -32768, 32767, 16, 128, 0)
        trig = AbsInfo(0, 0, 255, 0, 0, 0)
        hat = AbsInfo(0, -1, 1, 0, 0, 0)
        caps = {
            e.EV_KEY: sorted(set(_BTN.values()) | {e.BTN_MODE}),
            e.EV_ABS: [(e.ABS_X, stick), (e.ABS_Y, stick), (e.ABS_RX, stick), (e.ABS_RY, stick),
                       (e.ABS_Z, trig), (e.ABS_RZ, trig),
                       (e.ABS_HAT0X, hat), (e.ABS_HAT0Y, hat)],
        }
        self.ui = UInput(caps, name=name, vendor=vendor, product=product, version=0x110,
                         bustype=e.BUS_USB)
        self.reset()

    # --- report state (vJoy-style) --------------------------------------
    def reset(self) -> None:
        self.lx = self.ly = self.rx = self.ry = MIDDLE
        self.buttons = 0
        self.hat = DCENTER

    def send(self) -> None:
        w = self.ui.write
        w(e.EV_ABS, e.ABS_X, _stick(self.lx))
        w(e.EV_ABS, e.ABS_Y, _stick(self.ly))
        w(e.EV_ABS, e.ABS_RX, _stick(self.rx))
        w(e.EV_ABS, e.ABS_RY, _stick(self.ry))
        w(e.EV_ABS, e.ABS_Z, 255 if self.buttons & L2 else 0)
        w(e.EV_ABS, e.ABS_RZ, 255 if self.buttons & R2 else 0)
        for bit, key in _BTN.items():
            w(e.EV_KEY, key, 1 if self.buttons & bit else 0)
        hx, hy = _HAT[self.hat]
        w(e.EV_ABS, e.ABS_HAT0X, hx)
        w(e.EV_ABS, e.ABS_HAT0Y, hy)
        self.ui.syn()

    # --- helpers ---------------------------------------------------------
    def tap(self, buttons: int = 0, hat: int = DCENTER, hold: float = 0.1) -> None:
        self.reset()
        self.buttons, self.hat = buttons, hat
        self.send()
        time.sleep(hold)
        self.reset()
        self.send()

    def close(self) -> None:
        self.reset()
        self.send()
        self.ui.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
