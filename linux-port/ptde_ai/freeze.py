"""Catch the bot's position freezing mid-fight, and find out whether it's really stuck.

Overnight 2026-11-24 (player A), in the three longest fights the bot's position stopped changing
for 2-15 minutes - twice right after being backstabbed - while its HP, facing and animations
kept updating and it kept pushing the stick. Both position reads (the per-frame one and the
slow PLAYER chain) froze together. Either the character was really stuck in the game or the
reads broke. When it happens again this:
  * saves a screenshot (records/freeze/), so we can see it,
  * logs the camera position over the freeze (the camera follows the character, so if the
    camera keeps moving the character is moving and the reads are what broke),
  * tries a roll every few seconds and logs whether that moved it.
"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from pathlib import Path

from .gamepad import CIRCLE, MIDDLE

FREEZE_DIR = Path(__file__).resolve().parent.parent / "records" / "freeze"
FROZEN_MS = 2000          # position unchanged this long while pushing the stick
MOVED_M = 0.02
PUSH = 8000               # stick deflection (of 16384) that should move us
TRY_EVERY_MS = 5000       # roll attempts while frozen
TRY_PRESS_MS = 100
TRY_CHECK_MS = 1200
CAM_CHECK_MS = 3000
ARRIVAL_QUIET_MS = 8000   # just summoned / back home: the arrival animation holds us in place
                          # (live 2026-11-24 the "freeze" 3 s after a summon was exactly that)


def save_screenshot(path: Path) -> None:
    """Background screenshot (spectacle), so the main loop doesn't stall."""
    def run():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["spectacle", "-b", "-n", "-f", "-o", str(path)], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    threading.Thread(target=run, daemon=True).start()


class FreezeWatch:
    def __init__(self, screenshots: bool = True):
        self.screenshots = screenshots
        self.pos = None
        self.t_moved = 0.0
        self.t_pushed = None       # stick held since
        self.frozen = False
        self.t_frozen = 0.0
        self.cam0 = None
        self.cam_said = False
        self.try_at = 0.0
        self.try_until = 0.0
        self.try_check = None      # (time, pos) to compare after a roll
        self.tries = 0
        self.quiet_until = 0.0

    def arrived(self, now: float) -> None:
        """Summoned or back in our world: no freeze checks for a few seconds."""
        self.reset()
        self.quiet_until = now + ARRIVAL_QUIET_MS

    def reset(self) -> None:
        self.pos, self.frozen, self.t_pushed, self.try_check = None, False, None, None

    def update(self, p, pad, cam: dict, now: float, engaged: bool) -> list[str]:
        """Call each bot tick after the brain set the pad. May add a roll press. Returns log lines."""
        out: list[str] = []
        if now < self.quiet_until:
            self.pos = None
            return out
        if p is None or p.hp <= 0 or not engaged:
            if self.frozen:
                out.append(f"position frozen {(now - self.t_frozen) / 1000 + FROZEN_MS / 1000:.0f} s "
                           f"(fight over / not fighting now)")
            self.reset()
            return out
        here = (p.loc_x, p.loc_y)
        if self.pos is None or math.hypot(here[0] - self.pos[0], here[1] - self.pos[1]) > MOVED_M:
            if self.frozen:
                out.append(f"position moving again after {(now - self.t_frozen + FROZEN_MS) / 1000:.1f} s "
                           f"frozen ({self.tries} roll tries)")
            self.pos, self.t_moved, self.frozen, self.tries, self.try_check = here, now, False, 0, None
            return out
        pushing = abs(pad.lx - MIDDLE) > PUSH or abs(pad.ly - MIDDLE) > PUSH
        if pushing:
            self.t_pushed = self.t_pushed or now
        else:
            self.t_pushed = None
        cam_xy = (cam.get("pos_x"), cam.get("pos_z"))
        if (not self.frozen and now - self.t_moved > FROZEN_MS and self.t_pushed is not None
                and now - self.t_pushed > FROZEN_MS * 0.5):
            self.frozen, self.t_frozen, self.tries = True, now, 0
            self.cam0, self.cam_said = cam_xy, False
            self.try_at = now + 1000
            shot = ""
            if self.screenshots:
                path = FREEZE_DIR / time.strftime("%Y%m%d-%H%M%S.png")
                save_screenshot(path)
                shot = f", screenshot records/freeze/{path.name}"
            slow = (p.raw.get("loc_x_slow"), p.raw.get("loc_y_slow"))
            out.append(f"POSITION FROZEN: {FROZEN_MS / 1000:.0f} s at ({here[0]:.2f}, {here[1]:.2f}) "
                       f"while pushing the stick (slow read {slow[0]}, {slow[1]}; anim "
                       f"{p.animation_type}; camera {cam_xy}){shot}")
        if not self.frozen:
            return out
        # does the camera (which follows the character) still move?
        if (not self.cam_said and now - self.t_frozen > CAM_CHECK_MS and None not in cam_xy
                and self.cam0 and None not in self.cam0):
            self.cam_said = True
            d = math.hypot(cam_xy[0] - self.cam0[0], cam_xy[1] - self.cam0[1])
            out.append(f"frozen check: camera moved {d:.2f} m in {CAM_CHECK_MS / 1000:.0f} s "
                       f"({'so the character IS moving: the position read is broken' if d > 0.5 else 'camera still too: the character looks really stuck'})")
        # roll to try to break free, then see if it moved us
        if now < self.try_until:
            pad.buttons |= CIRCLE
        elif self.try_check is None and now >= self.try_at:
            self.tries += 1
            self.try_until = now + TRY_PRESS_MS
            self.try_check = (now + TRY_CHECK_MS, here)
            pad.buttons |= CIRCLE
        elif self.try_check is not None and now >= self.try_check[0]:
            out.append(f"frozen: roll try {self.tries} didn't move us (anim {p.animation_type})")
            self.try_check = None
            self.try_at = now + TRY_EVERY_MS
        return out
