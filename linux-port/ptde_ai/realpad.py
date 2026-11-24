"""Read the user's physical controller so its input can be passed through the bot's pad.

Why: Dark Souls only reads XInput slot 0 and Wine hands out slots in connection
order. If the bot's virtual pad owns slot 0, the real controller lands in slot 1
and the game ignores it, so the bot forwards it instead. Then the game only ever
listens to one controller, and switching between you and the bot is instant.

Handles the common Linux layouts for Xbox-style pads:
  * xpad / xpadneo: right stick ABS_RX/RY, triggers ABS_Z/RZ
  * hid-microsoft over Bluetooth: right stick ABS_Z/RZ, triggers ABS_BRAKE/GAS,
    View/Menu sometimes as KEY_BACK / KEY_MENU
"""
from __future__ import annotations

import select
import subprocess
import threading
import time
from dataclasses import dataclass, field

import evdev
from evdev import ecodes as e

from .gamepad import (CIRCLE, CROSS, DCENTER, DDOWN, DLEFT, DRIGHT, DUP, L1, L2, L3,
                      MIDDLE, R1, R2, R3, SELECT, SQUARE, START, TRIANGLE)

BOT_PAD_NAME = "PTDE Bot Pad"

_KEYS = {
    e.BTN_SOUTH: CROSS, e.BTN_EAST: CIRCLE, e.BTN_NORTH: SQUARE, e.BTN_WEST: TRIANGLE,
    e.BTN_TL: L1, e.BTN_TR: R1, e.BTN_THUMBL: L3, e.BTN_THUMBR: R3,
    e.BTN_SELECT: SELECT, e.KEY_BACK: SELECT, e.BTN_START: START, e.KEY_MENU: START,
    e.BTN_TL2: L2, e.BTN_TR2: R2,
}

STICK_ACTIVE = 0.35    # stick deflection that counts as "you're playing"
TRIGGER_ON = 0.5


@dataclass
class PadState:
    """Normalized controller state: sticks -1..1 (y down positive), triggers 0..1."""
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    keys: set = field(default_factory=set)

    def buttons(self) -> int:
        b = 0
        for k in self.keys:
            b |= _KEYS.get(k, 0)
        if self.lt > TRIGGER_ON:
            b |= L2
        if self.rt > TRIGGER_ON:
            b |= R2
        return b

    def hat(self) -> int:
        if self.hat_y < 0:
            return DUP
        if self.hat_y > 0:
            return DDOWN
        if self.hat_x < 0:
            return DLEFT
        if self.hat_x > 0:
            return DRIGHT
        return DCENTER

    def active(self, ignore: int = 0) -> bool:
        """True when the player is doing anything (used to take control from the bot)."""
        return bool(
            (self.buttons() & ~ignore) or self.hat() != DCENTER or
            max(abs(self.lx), abs(self.ly), abs(self.rx), abs(self.ry)) > STICK_ACTIVE
        )

    def apply_to(self, pad) -> None:
        """Copy this state onto the bot's virtual pad (vJoy-style units)."""
        to_v = lambda x: int(MIDDLE + max(-1.0, min(1.0, x)) * (MIDDLE - 1))
        pad.lx, pad.ly, pad.rx, pad.ry = to_v(self.lx), to_v(self.ly), to_v(self.rx), to_v(self.ry)
        pad.buttons = self.buttons()
        pad.hat = self.hat()


def find_real_controller() -> evdev.InputDevice | None:
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        caps = dev.capabilities()
        if (dev.name == BOT_PAD_NAME or dev.phys == "py-evdev-uinput"
                or e.BTN_SOUTH not in caps.get(e.EV_KEY, [])
                or e.EV_ABS not in caps):
            dev.close()
            continue
        return dev
    return None


class RealPad:
    """Background reader for the physical controller. Survives unplug/replug."""

    def __init__(self):
        self.state = PadState()
        self.device: evdev.InputDevice | None = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> PadState:
        with self._lock:
            s = self.state
            return PadState(s.lx, s.ly, s.rx, s.ry, s.lt, s.rt, s.hat_x, s.hat_y, set(s.keys))

    @property
    def connected(self) -> bool:
        return self.device is not None

    def close(self) -> None:
        self._stop = True

    # --------------------------------------------------------------- internals
    def _setup_axes(self, dev):
        absinfo = dict(dev.capabilities(absinfo=True).get(e.EV_ABS, []))
        has = absinfo.__contains__
        right = (e.ABS_RX, e.ABS_RY) if has(e.ABS_RX) else (e.ABS_Z, e.ABS_RZ)
        if has(e.ABS_BRAKE):
            trig = (e.ABS_BRAKE, e.ABS_GAS)
        elif right[0] == e.ABS_RX and has(e.ABS_Z):
            trig = (e.ABS_Z, e.ABS_RZ)
        else:
            trig = (None, None)
        self._axes = {e.ABS_X: "lx", e.ABS_Y: "ly", right[0]: "rx", right[1]: "ry"}
        self._trig = {trig[0]: "lt", trig[1]: "rt"}
        self._info = absinfo

    def _norm_stick(self, code, v):
        i = self._info[code]
        mid = (i.max + i.min) / 2
        return max(-1.0, min(1.0, (v - mid) / ((i.max - i.min) / 2)))

    def _norm_trig(self, code, v):
        i = self._info[code]
        return (v - i.min) / max(1, i.max - i.min)

    def _handle(self, ev):
        s = self.state
        if ev.type == e.EV_KEY:
            (s.keys.add if ev.value else s.keys.discard)(ev.code)
        elif ev.type == e.EV_ABS:
            if ev.code in self._axes:
                setattr(s, self._axes[ev.code], self._norm_stick(ev.code, ev.value))
            elif ev.code in self._trig and ev.code is not None:
                setattr(s, self._trig[ev.code], self._norm_trig(ev.code, ev.value))
            elif ev.code == e.ABS_HAT0X:
                s.hat_x = ev.value
            elif ev.code == e.ABS_HAT0Y:
                s.hat_y = ev.value

    def _run(self):
        while not self._stop:
            if self.device is None:
                dev = find_real_controller()
                if dev is None:
                    time.sleep(1.0)
                    continue
                self._setup_axes(dev)
                with self._lock:
                    self.state = PadState()
                self.device = dev
            try:
                r, _, _ = select.select([self.device.fd], [], [], 0.2)
                if r:
                    for ev in self.device.read():
                        with self._lock:
                            self._handle(ev)
            except OSError:           # unplugged / Bluetooth dropped
                with self._lock:
                    self.state = PadState()
                self.device = None


# ------------------------------------------------------------- Bluetooth help
def bluetooth_mac(dev: evdev.InputDevice) -> str | None:
    """Bluetooth controllers report their own address in `uniq`."""
    uniq = (dev.uniq or "").strip()
    return uniq.upper() if uniq.count(":") == 5 else None


def bluetooth_disconnect(mac: str) -> bool:
    r = subprocess.run(["bluetoothctl", "disconnect", mac], capture_output=True, text=True)
    return "Successful" in r.stdout or "successful" in r.stdout


def bluetooth_connect(mac: str, timeout: float = 8.0) -> bool:
    r = subprocess.run(["bluetoothctl", "--timeout", str(int(timeout)), "connect", mac],
                       capture_output=True, text=True)
    return "Connection successful" in r.stdout
