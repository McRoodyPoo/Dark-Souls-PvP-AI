"""Press keyboard keys through a virtual (uinput) keyboard, for menus the bot can't drive.

  python tools/keys.py ENTER              # one press
  python tools/keys.py DOWN ENTER         # several, 0.6 s apart

Goes to whichever window has focus (the game, when it's in front).
"""
from __future__ import annotations

import sys
import time

from evdev import UInput, ecodes as e

KEYS = {"ENTER": e.KEY_ENTER, "DOWN": e.KEY_DOWN, "UP": e.KEY_UP, "LEFT": e.KEY_LEFT,
        "RIGHT": e.KEY_RIGHT, "ESC": e.KEY_ESC, "E": e.KEY_E, "Q": e.KEY_Q}


def main(names: list[str]) -> None:
    ui = UInput({e.EV_KEY: list(KEYS.values())}, name="claude-keys")
    try:
        time.sleep(0.8)                      # let the compositor pick up the new keyboard
        for n in names:
            k = KEYS[n.upper()]
            ui.write(e.EV_KEY, k, 1); ui.syn()
            time.sleep(0.08)
            ui.write(e.EV_KEY, k, 0); ui.syn()
            time.sleep(0.6)
    finally:
        ui.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["ENTER"])
