"""Which quick item is selected, read off the screen (the memory field couldn't be found).

The quick-item slot sits bottom-left of the game's HUD with the item name under it
(2026-11-24, game on the right-hand 1920x1080 monitor of a 3840x1080 desktop):
  * Red Sign Soapstone: a red crystal icon      -> many strongly red pixels
  * Traversal Trinket:  a grey/silver ring icon -> no red pixels, some silver
and the white label text is checked against saved masks of both names.
Anything else (menu open, loading screen, different monitor layout) reads as None.

A screenshot takes ~0.5 s, so it runs on a background thread: request(), then poll().
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "hud"
SCREEN = (3840, 1080)
ICON = (2200, 800, 2335, 950)         # x0, y0, x1, y1 of the quick-item icon
LABEL = (2340, 955, 2560, 1000)       # the item name under it
SOAPSTONE, TRINKET, KNIVES = "soapstone", "trinket", "knives"
RED_MIN = 0.03                        # soapstone measured 0.093, trinket 0.0
RED_MAX_TRINKET = 0.005
LABEL_MATCH = 0.45                    # IoU with the right label (the wrong one measured 0.17)
MAX_AGE_MS = 4000


def classify(img) -> str | None:
    """img: a PIL image of the whole desktop. Returns SOAPSTONE, TRINKET or None (can't tell)."""
    import numpy as np
    if img.size != SCREEN:
        return None
    a = np.asarray(img.convert("RGB")).astype(int)
    x0, y0, x1, y1 = ICON
    icon = a[y0:y1, x0:x1]
    r, g, b = icon[..., 0], icon[..., 1], icon[..., 2]
    red = ((r > 110) & (r > 2 * g) & (r > 2 * b)).mean()
    x0, y0, x1, y1 = LABEL
    text = a[y0:y1, x0:x1].min(axis=2) > 170
    iou = {}
    for name in (SOAPSTONE, TRINKET, KNIVES):
        try:
            m = np.load(ASSETS / f"label_{name}.npy")
        except OSError:
            if name == KNIVES:
                iou[name] = 0.0                    # no knife template captured yet
                continue
            return None
        union = (m | text).sum()
        iou[name] = (m & text).sum() / union if union else 0.0
    if iou[KNIVES] >= LABEL_MATCH and iou[KNIVES] > max(iou[SOAPSTONE], iou[TRINKET]):
        return KNIVES
    if red >= RED_MIN and iou[SOAPSTONE] >= LABEL_MATCH and iou[SOAPSTONE] > iou[TRINKET]:
        return SOAPSTONE
    if red <= RED_MAX_TRINKET and iou[TRINKET] >= LABEL_MATCH and iou[TRINKET] > iou[SOAPSTONE]:
        return TRINKET
    return None


PROMPT = (2690, 880, 3110, 922)       # "A: Check your Summon Sign" text
PROMPT_MATCH = 0.6                    # measured 1.0 with the sign down, 0.01 with a dialog up


def sign_prompt(img) -> bool | None:
    """Is "Check your Summon Sign" on screen (= our sign is down where we stand)? None: can't tell."""
    import numpy as np
    if img.size != SCREEN:
        return None
    try:
        m = np.load(ASSETS / "prompt_sign.npy")
    except OSError:
        return None
    a = np.asarray(img.convert("RGB")).astype(int)
    x0, y0, x1, y1 = PROMPT
    text = a[y0:y1, x0:x1].min(axis=2) > 170
    union = (m | text).sum()
    return bool(union and (m & text).sum() / union >= PROMPT_MATCH)


def _screenshot(fn):
    """Screenshot the desktop and run fn(img) on it (blocking, ~0.4 s). None on any failure."""
    try:
        from PIL import Image
    except ImportError:
        return None
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        subprocess.run(["spectacle", "-b", "-n", "-f", "-o", path], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        with Image.open(path) as img:
            return fn(img)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def grab() -> str | None:
    """Which quick item is selected, from a screenshot."""
    return _screenshot(classify)


def sign_visible() -> bool | None:
    """Our summon sign is down where we stand, from a screenshot (no buttons pressed)."""
    return _screenshot(sign_prompt)


class HudReader:
    """Background screenshots: request() starts one, poll() gives (item, taken_at_ms) when done."""

    def __init__(self, grabber=grab):
        self._grab = grabber
        self._thread: threading.Thread | None = None
        self._result: tuple[str | None, float] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request(self) -> None:
        if self.busy:
            return
        self._result = None

        def run():
            t = time.perf_counter() * 1000.0
            self._result = (self._grab(), t)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def poll(self) -> tuple[str | None, float] | None:
        if self.busy:
            return None
        r, self._result = self._result, None
        return r
