"""Camera-relative steering. Replaces metal-crow's camera lock.

metal-crow's angleToJoystick() assumed the camera was frozen at a known yaw
(HelperUtil.c: "MUST LOCK CAMERA for movement to work"), which is why the bot
needed the "lock camera rotation" Cheat Engine script plus a per-map compile flag
(OolicelMap). Here we read the live camera yaw every tick and rotate the desired
world-space heading into stick space instead, so nothing is written to the game.

The sign/offset between camera yaw and world heading is measured once by
`scripts/probe.py --calibrate-steering` and stored in calibration.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .gamepad import MIDDLE

CAL_PATH = Path(__file__).resolve().parent.parent / "calibration.json"


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def heading(dx: float, dy: float) -> float:
    """World heading of a displacement, same convention as metal-crow (atan2(dx, dy))."""
    return math.atan2(dx, dy)


@dataclass
class Steering:
    offset: float = 0.0   # world heading of "stick forward" minus camera yaw
    sign: float = 1.0     # +1 if "stick right" turns heading by +pi/2, else -1

    @classmethod
    def load(cls) -> "Steering":
        if CAL_PATH.exists():
            return cls(**json.loads(CAL_PATH.read_text()))
        return cls()

    def save(self) -> None:
        CAL_PATH.write_text(json.dumps({"offset": self.offset, "sign": self.sign}, indent=2))

    def stick_for_heading(self, world_heading: float, cam_yaw: float,
                          magnitude: float = 1.0) -> tuple[int, int]:
        """vJoy-style (lx, ly) that moves the character toward world_heading."""
        rel = self.sign * wrap(world_heading - cam_yaw - self.offset)
        x = math.sin(rel) * magnitude
        fwd = math.cos(rel) * magnitude
        lx = int(MIDDLE + x * (MIDDLE - 1))
        ly = int(MIDDLE - fwd * (MIDDLE - 1))   # vJoy: 1 = top = forward
        return lx, ly

    def stick_toward(self, me, target, cam_yaw: float, away: bool = False) -> tuple[int, int]:
        h = heading(target.loc_x - me.loc_x, target.loc_y - me.loc_y)
        return self.stick_for_heading(h + (math.pi if away else 0.0), cam_yaw)
