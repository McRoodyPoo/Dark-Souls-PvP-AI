"""Who's driving: you or the bot.

  * Starts with you in control (your controller is passed straight through).
  * Click both sticks (L3 + R3) - together, or one then the other within 0.4 s, no need
    to hold: the bot takes over. (It used to need both held a full 0.5 s with no flicker,
    which the user found too tight, 2026-11-23.)
  * Touch anything while the bot drives (button, d-pad, stick past 35%): you're back in
    control instantly - except for a short grace period after the handover, so the stick
    nudge from clicking them doesn't snatch it straight back.
"""
from __future__ import annotations

from .gamepad import L3, R3
from .realpad import PadState

MANUAL, BOT = "you", "bot"
COMBO = L3 | R3
PAIR_WINDOW_MS = 400      # L3 and R3 clicked within this of each other
GRACE_MS = 700            # after the handover, ignore input this long (and until released)


class Handoff:
    def __init__(self, start_as_bot: bool = False):
        self.mode = BOT if start_as_bot else MANUAL
        self._l3_t = self._r3_t = -1e9
        self._quiet_until = -1e9
        self._wait_release = False     # ignore input until both sticks are let go

    def update(self, s: PadState, now_ms: float) -> str | None:
        """Feed the latest controller state; returns the new mode when it changes."""
        b = s.buttons()
        if b & L3:
            self._l3_t = now_ms
        if b & R3:
            self._r3_t = now_ms
        if self._wait_release:
            if not (b & COMBO):
                self._wait_release = False
            return None
        if now_ms < self._quiet_until:
            return None
        if (self.mode == MANUAL and now_ms - self._l3_t <= PAIR_WINDOW_MS
                and now_ms - self._r3_t <= PAIR_WINDOW_MS):
            self.mode = BOT
            self._wait_release = True
            self._quiet_until = now_ms + GRACE_MS
            self._l3_t = self._r3_t = -1e9
            return BOT
        if self.mode == BOT and s.active():
            self.mode = MANUAL
            return MANUAL
        return None
