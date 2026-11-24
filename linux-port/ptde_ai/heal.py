"""Heal between fights with a Divine Blessing (user, 2026-11-24: "he should know to heal between
fights even when other players are doing 1HP fights. Press start > spam A to use divine
blessing").

Sequence: Start (opens the menu), then A every PRESS_GAP_MS until the item-use animation
starts or our HP goes up (so a second Blessing is never used), then B x3 to back out of the
menu (B rather than Start: if using the item already closed the menu, Start would reopen it
and leave the bot stuck in it; a stray B outside the menu is only a backstep).
If nothing heals after MAX_A presses: close the menu and back off (out of Blessings?).
The caller decides when it's safe (no fight on); this only drives the buttons.
"""
from __future__ import annotations

from .gamepad import CIRCLE, CROSS, START

TAP_MS = 100
MENU_OPEN_MS = 700          # after Start, before the first A
PRESS_GAP_MS = 400          # between A presses (time to see the use animation start)
MAX_A = 6
SETTLE_MS = 2500            # after it worked: let the drink finish before closing the menu
CLOSE_PRESSES = 3
CLOSE_GAP_MS = 300
FAIL_BACKOFF_MS = 300_000   # nothing healed: don't try again for a while
ABORT_BACKOFF_MS = 8000     # a fight cut it short: not again straight away
FAR_M = 15.0                # an opponent further than this isn't a fight in progress


def between_fights(waiting_turn: bool, my_hp: int, opponent_fightable: bool, their_hp: int,
                   dist: float) -> bool:
    """Only heal BETWEEN fights (live 2026-11-24 it tried mid-fight at 6 m and at 212 HP):
    waiting our turn in the dugout, or the round is over (us or them at 1 HP), or there's no
    opponent at all / none within FAR_M. Distance alone inside a fight never counts."""
    if waiting_turn:
        return True
    if my_hp <= 1:
        return True
    if not opponent_fightable or dist > FAR_M:
        return True
    return their_hp <= 1
MIN_MISSING_FRAC = 0.10     # heal when this much HP is missing

IDLE, OPENING, PRESSING, DRINKING, CLOSING = "idle", "opening", "pressing", "drinking", "closing"


class Healer:
    def __init__(self):
        self.state = IDLE
        self.t = 0.0
        self.presses = 0
        self.hp0 = 0
        self.tap_until = 0.0
        self.tap_btn = 0
        self.retry_at = 0.0
        self.worked = False
        self.closes = 0

    @property
    def active(self) -> bool:
        return self.state != IDLE

    @staticmethod
    def wants(hp: int, max_hp: int) -> bool:
        return max_hp > 0 and 0 < hp and (max_hp - hp) >= MIN_MISSING_FRAC * max_hp

    def _tap(self, btn: int, now: float) -> None:
        self.tap_btn, self.tap_until = btn, now + TAP_MS

    def start(self, hp: int, now: float) -> str | None:
        if self.active or now < self.retry_at:
            return None
        self.state, self.t, self.presses, self.hp0, self.worked = OPENING, now, 0, hp, False
        self._tap(START, now)
        return "healing between fights: Start > A (Divine Blessing)"

    def _close(self, now: float) -> None:
        self.state, self.t, self.closes = CLOSING, now, 1
        self._tap(CIRCLE, now)

    def abort(self, now: float, pad) -> None:
        """A fight started mid-heal: back out of the menu and give up for now."""
        if self.state in (OPENING, PRESSING, DRINKING):
            self._close(now)
            self.retry_at = max(self.retry_at, now + ABORT_BACKOFF_MS)

    def update(self, hp: int, anim: int, now: float, pad) -> str | None:
        """Call every tick while active; drives the pad. Returns a log line on outcomes."""
        if now < self.tap_until:
            pad.buttons |= self.tap_btn
            return None
        healed = hp > self.hp0 + 20 or anim != 0
        if self.state == OPENING:
            if now - self.t >= MENU_OPEN_MS:
                self.state, self.t = PRESSING, now
                self._tap(CROSS, now)
                self.presses = 1
            return None
        if self.state == PRESSING:
            if healed:
                self.state, self.t, self.worked = DRINKING, now, True
                return None
            if now - self.t >= PRESS_GAP_MS:
                if self.presses >= MAX_A:
                    self._close(now)
                    self.retry_at = now + FAIL_BACKOFF_MS
                    return f"heal: {MAX_A} A presses and no heal (out of Divine Blessings?): closing the menu"
                self.t = now
                self._tap(CROSS, now)
                self.presses += 1
            return None
        if self.state == DRINKING:
            if now - self.t >= SETTLE_MS:
                self._close(now)
                return f"healed: {self.hp0} -> {hp} HP"
            return None
        if self.state == CLOSING:
            if now - self.t >= CLOSE_GAP_MS:
                if self.closes >= CLOSE_PRESSES:
                    self.state = IDLE
                else:
                    self.closes += 1
                    self.t = now
                    self._tap(CIRCLE, now)
            return None
        return None
