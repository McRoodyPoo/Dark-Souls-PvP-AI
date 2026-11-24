"""Auto red-sign: keep a Red Sign Soapstone sign down while the bot waits at home.

Port of metal-crow's AutoRedSign loop (Handler.c), with each blind sleep replaced by
a memory check:
  * home vs summoned comes from our own character type (0/8 = home, 2 = red phantom);
  * the soapstone must be the ONLY quick item, so "use item" can't use anything else
    (metal-crow assumed it was selected; his item-cycling code was disabled);
  * a press counts only if the use-item animation actually starts, else it retries;
    but if a sign is already down, the game ignores the press (no animation), so
    "nothing happened" then means "sign still there", not a failure;
  * coming home from a duel clears any earlier give-up (the sign was consumed);
  * A is pressed only as part of the periodic sign check (to clear a "Failed to join
    session" box), standing at the arena centre where there's nothing to talk to.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import offsets as O
from .character import Sub
from .gamepad import CIRCLE, CROSS, SQUARE

PRESS_MS = 120            # how long to hold "use item"
CONFIRM_MS = 1500         # use-item animation must start within this
RETRY_MS = 5000           # wait after an unconfirmed press
RESIGN_MS = 240_000       # re-place the sign this often while waiting
AFTER_DUEL_MS = 13000     # settle time after returning home (after a death the reload into
                          # the dugout came ~9.5 s later: an 8 s wait signed just before it)
CHECK_MS = 120_000        # with a sign down: check it's really still there this often
ACK_MS = 120              # A press that clears a message box ("Failed to join session")
AFTER_ACK_MS = 800
MAX_FAILS = 3


TRAVERSAL_TRINKET = 276          # arena mod: leaves the dugout (see escape.py)
POISON_KNIFE = 291               # thrown at corner campers (brain.Atk.KNIVES)


def soapstone_ready(quick_items: list[int]) -> tuple[bool, str]:
    equipped = [i for i in quick_items if i > 0]
    if equipped == [O.RED_SIGN_SOAPSTONE]:
        return True, "Red Sign Soapstone is the only quick item"
    if sorted(equipped) == sorted([O.RED_SIGN_SOAPSTONE, TRAVERSAL_TRINKET]):
        return True, ("Red Sign Soapstone + Traversal Trinket equipped "
                      "(have the SOAPSTONE selected when the bot starts)")
    if sorted(equipped) == sorted([O.RED_SIGN_SOAPSTONE, TRAVERSAL_TRINKET, POISON_KNIFE]):
        return True, ("Red Sign Soapstone + Traversal Trinket + Poison Throwing Knives equipped "
                      "(have the SOAPSTONE selected when the bot starts)")
    if O.RED_SIGN_SOAPSTONE not in equipped:
        return False, "equip the Red Sign Soapstone in a quick-item slot"
    return False, ("quick items must be just the soapstone (plus the Traversal Trinket), so "
                   f"use item can't use anything else (quick slots: {quick_items})")


@dataclass
class RedSign:
    last_sign_ms: float = -1e12
    press_until: float = 0.0
    confirm_until: float = 0.0
    retry_at: float = 0.0
    returned_at: float | None = None
    fails: int = 0
    was_away: bool = False
    gave_up: bool = False
    sign_down: bool = False   # a sign we placed is (probably) still on the ground
    dismiss_until: float = 0.0
    ack_until: float = 0.0    # holding A to clear a message box
    ack_wait_until: float = 0.0
    checking: bool = False    # this press is a check on a sign we think is down
    sign_visible: object = None   # () -> True/False/None: look at the screen for our sign

    def busy(self, now: float) -> bool:
        """Mid press / check / dialog dismissal: a restart now could leave a dialog open
        (live 2026-11-24 a hot reload between the soapstone press and its B left "Cancel summon
        sign?" up, which blocks every item)."""
        return bool(now < self.press_until + 300 or self.confirm_until or now < self.dismiss_until + 300
                    or now < self.ack_until or self.ack_wait_until)

    def update_location(self, self_chr_type: int | None, now: float) -> str | None:
        """Track home/summoned transitions; returns a message when one happens."""
        if self_chr_type == 2 and not self.was_away:
            self.was_away = True
            self.sign_down = False           # being summoned consumes the sign
            return "summoned as a red phantom: fighting"
        if self_chr_type in O.HOME_CHR_TYPES and self.was_away:
            self.was_away = False
            self.returned_at = now
            self.last_sign_ms = -1e12        # sign was consumed: place a new one
            self.sign_down = False
            self.fails, self.gave_up, self.confirm_until = 0, False, 0.0
            return "back in our world: re-signing shortly"
        return None

    def tick(self, p, now: float, pad) -> str | None:
        """Call while home with no opponent. Drives the pad; returns a log line or None."""
        if self.gave_up:
            return None
        if now < self.dismiss_until:
            pad.buttons |= CIRCLE                 # close a "Cancel summon sign?" box
            return None
        if now < self.ack_until:
            pad.buttons |= CROSS                  # close a "Failed to join session" box
            return None
        if now < self.ack_wait_until:
            return None
        if self.ack_wait_until:                   # message cleared: now try the soapstone
            self.ack_wait_until = 0.0
            self.press_until = now + PRESS_MS
            self.confirm_until = now + PRESS_MS + CONFIRM_MS
            pad.buttons |= SQUARE
            return None
        if self.sign_down and not self.confirm_until and now >= self.press_until:
            # A failed summon ("Failed to join session", live 2026-11-23) removes the sign
            # without us being summoned, and its message box blocks items. So every
            # CHECK_MS: A to clear any message, then the soapstone. If the sign is gone the
            # soapstone places a new one; if it's still there it opens "Cancel summon
            # sign?", which B closes (dismiss_until below).
            if (now - self.last_sign_ms >= CHECK_MS and now >= self.retry_at
                    and p.sub in (Sub.NEUTRAL, Sub.RECOVER) and p.animation_type == 0):
                # Look first (2026-11-24, reports of failed summons): standing on our own sign,
                # A opens "Check your Summon Sign" and the soapstone "Cancel summon sign?", and
                # a summon arriving while one is open may fail. The prompt on screen shows
                # the sign is down, so then press nothing.
                if self.sign_visible is not None and self.sign_visible():
                    self.last_sign_ms = now
                    return None
                self.checking = True
                self.ack_until = now + ACK_MS
                self.ack_wait_until = now + ACK_MS + AFTER_ACK_MS
                pad.buttons |= CROSS
            return None
        if now < self.press_until:
            pad.buttons |= SQUARE
            return None
        if self.confirm_until:
            if p.animation_type != 0:                     # use-item animation started
                self.confirm_until, self.fails = 0.0, 0
                self.last_sign_ms = now
                self.sign_down = True
                if self.checking:
                    self.checking = False
                    return "red sign placed (the old one was gone: failed summon?)"
                return "red sign placed"
            if now > self.confirm_until:
                self.confirm_until = 0.0
                self.dismiss_until = now + 120            # in case a dialog opened instead
                if self.sign_down:                        # the "Cancel summon sign?" box opened
                    self.last_sign_ms = now
                    quiet, self.checking = self.checking, False
                    return None if quiet else "sign still down (use item did nothing), checking again later"
                self.fails += 1
                self.retry_at = now + RETRY_MS
                if self.fails >= MAX_FAILS:
                    # Usually a sign is already down (placed by hand, or before a restart).
                    # Back off instead of giving up for good.
                    self.fails = 0
                    self.retry_at = now + RESIGN_MS
                    return (f"use item didn't register {MAX_FAILS}x: a sign is probably already "
                            f"down (or a menu is open); trying again in {RESIGN_MS // 60000} min")
                return f"use item didn't register, retrying ({self.fails}/{MAX_FAILS})"
            return None
        if now < self.retry_at:
            return None
        if self.returned_at is not None and now - self.returned_at < AFTER_DUEL_MS:
            return None
        if now - self.last_sign_ms < RESIGN_MS:
            return None
        if p.sub not in (Sub.NEUTRAL, Sub.RECOVER) or p.animation_type != 0:
            return None                                   # busy; wait until idle
        self.press_until = now + PRESS_MS
        self.confirm_until = now + PRESS_MS + CONFIRM_MS
        pad.buttons |= SQUARE
        return "placing red sign"
