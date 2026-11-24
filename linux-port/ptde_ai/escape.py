"""Dugout escape for the arena mod (home side).

When the bot dies it reloads into a "dugout" in its own world. Then it should:
  1. switch to the Traversal Trinket (quick item 276) and use it,
  2. walk to the centre of the arena,
  3. switch back to the Red Sign Soapstone, and place a sign (redsign.py does that).
Positions come from arena.json ("dugout" and "center", recorded in game).

Which quick item is selected can't be read from memory on this build, so the bot keeps
count of its own d-pad-down presses (`cycles`, 0 = soapstone) and corrects the count
from what "use item" does (live 2026-11-22 a lost press threw the count off):
  * we teleported out of the dugout   -> the trinket is selected
  * we teleported back INTO the dugout right after signing -> the trinket was selected
  * an item animation but no teleport -> the soapstone was selected: cycle and retry
  * nothing happened                  -> the press was lost: retry without cycling
Before every "use item" it also LOOKS at the quick-item slot on screen (hud.py) and fixes
the count from that. In the home dugout it never presses "use" unless the screen shows the
trinket: a soapstone press there would put the bot's sign in the dugout (user, 2026-11-24:
"it needs to not put its sign in the dugout"). Elsewhere an unreadable screen falls back
to the count. The user's own d-pad presses are counted too.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .character import Sub
from .gamepad import DDOWN, SQUARE
from .hud import KNIVES, SOAPSTONE, TRINKET
from .steer import heading

TRAVERSAL_TRINKET = 276
ARENA_PATH = Path(__file__).resolve().parent.parent / "arena.json"

DUGOUT_RADIUS = 6.0          # metres, horizontal
DUGOUT_HEIGHT_TOL = 2.0
SETTLE_MS = 2500             # after reloading, before pressing anything
PRESS_MS = 120
CYCLE_MS = 100               # d-pad down tap
AFTER_CYCLE_MS = 700
LEAVE_MS = 8000              # trinket used: should be out of the dugout by then
NO_ANIM_RETRY_MS = 1200      # no item animation at all by then: the press was lost, press again
                             # (live 2026-11-24 a press right after a summon was ignored and the
                             # bot waited the full 8 s before retrying: 13 s to reach the dugout)
ARRIVE_RADIUS = 1.0
WALK_TIMEOUT_MS = 40000
MAX_TRIES = 3
VERIFY_TRIES = 3             # unreadable screenshots before falling back to the count
VERIFY_RETRY_MS = 3000       # dugout, count not trusted, screen unreadable: look again this often

IDLE, SETTLE, SELECT_TRINKET, USE, LEAVING, WALK, SELECT_SOAP, DONE = (
    "idle", "settle", "select-trinket", "use", "leaving", "walk", "select-soapstone", "done")
# Arena etiquette (user, 2026-11-23): lost the round (1 HP) -> trinket out to the dugout; in the
# dugout, wait while two are fighting in the arena and go back in when one of them leaves.
OUT_SETTLE, OUT_SELECT, OUT_USE, OUT_LEAVING, WAIT_TURN = (
    "out-settle", "out-select", "out-use", "out-leaving", "wait-turn")
VERIFY = "verify"             # looking at the quick-item slot before pressing "use"
OUT_SETTLE_MS = 800          # at 1 HP: let the hit / knockdown play out first
TURN_CLEAR_MS = 3000         # the arena must stay down to <= 1 player this long before we go
ARRIVAL_LOOK_MS = 1500       # summoned: look at who's in the arena this long after arriving
ARRIVAL_QUEUE_MS = 30000     # (never later than this)


def should_queue_on_arrival(summoned: bool, since_arrival_ms: float,
                            others_in_arena: int | None, hp: int) -> bool:
    """Summoned and two players are already in the arena: straight to the dugout to wait our
    turn, no bow (user 2026-11-24: "if there are already two players in the arena, teleport to
    the dugout. Do not bow first. Just get out"). Only on arrival: never leave mid-round."""
    return (summoned and ARRIVAL_LOOK_MS <= since_arrival_ms < ARRIVAL_QUEUE_MS
            and (others_in_arena or 0) >= 2 and hp > 1)


def load_arena() -> dict:
    try:
        return json.loads(ARENA_PATH.read_text())
    except (OSError, ValueError):
        return {}


@dataclass
class DugoutEscape:
    arena: dict = field(default_factory=load_arena)
    n_items: int = 2
    cycles: int = 0                  # d-pad downs since the soapstone was selected
    trinket_at: int = 1              # presses from the soapstone to the trinket (quick-slot order)
    knife_at: int | None = None      # ...to the poison throwing knives, if equipped
    state: str = IDLE
    t_state: float = 0.0
    tries: int = 0
    press_until: float = 0.0
    cycle_until: float = 0.0
    wait_until: float = 0.0
    anim_seen: bool = False
    done_at: float | None = None
    seen_two: bool = False
    said_wait: bool = False
    clear_since: float | None = None
    hud: object = None               # hud.HudReader, or None (no screen check)
    known: bool = True               # False: the count may be off (the user pressed d-pad down)
    want: str = TRINKET              # VERIFY: the item we're about to use
    after_verify: str = IDLE         # VERIFY: the state that presses "use" next
    verify_fails: int = 0
    verify_cycles: int = 0
    strict: bool = False             # in our own world's dugout: never press on a guess
    said_unsure: bool = False
    verified: bool = False           # VERIFY passed (or fell back): go ahead and press
    summoned_now: bool = False       # in someone else's world (no sign possible, HUD item greyed)

    def queue(self, now: float) -> str | None:
        """Go to the dugout and wait our turn (the same trip as the 1 HP exit)."""
        if self.state not in (IDLE, DONE):
            return None
        self._go(OUT_SETTLE, now)
        return "two players already in the arena: straight to the dugout to wait my turn"

    # ---------------------------------------------------------------- helpers
    @property
    def ready(self) -> bool:
        return "dugout" in self.arena and "center" in self.arena

    @property
    def active(self) -> bool:
        return self.state not in (IDLE, DONE)

    def in_dugout(self, p) -> bool:
        if p is None or not self.ready:
            return False
        d = self.arena["dugout"]
        h = p.raw.get("loc_h")
        return (math.hypot(p.loc_x - d["x"], p.loc_y - d["y"]) < DUGOUT_RADIUS
                and h is not None and abs(h - d["h"]) < DUGOUT_HEIGHT_TOL)

    def _go(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    def _cycle(self, now: float, pad) -> None:
        self.cycles = (self.cycles + 1) % self.n_items
        self.cycle_until = now + CYCLE_MS
        self.wait_until = now + CYCLE_MS + AFTER_CYCLE_MS
        pad.hat = DDOWN

    def restore_soapstone(self, now: float, pad) -> bool:
        """Something left another item selected (the knives): one press toward the soapstone.
        True while still switching (the caller drives nothing else that tick)."""
        if now < self.cycle_until:
            pad.hat = DDOWN
            return True
        if now < self.wait_until:
            return True
        if self.cycles % self.n_items != 0:
            self._cycle(now, pad)
            return True
        return False

    def user_pressed_ddown(self) -> None:
        """The user (or a test hook) cycled quick items: count it, but stop trusting the count."""
        self.cycles = (self.cycles + 1) % self.n_items
        self.known = False

    def _verify(self, want: str, then: str, now: float) -> None:
        """Look at the screen before pressing "use" (falls through to `then` without a hud)."""
        if self.hud is None or self.summoned_now:
            # summoned: the item icon is greyed out (unreadable) and a soapstone press can't
            # place a sign in their world, so don't spend a second looking
            self.verified = True
            self._go(then, now)
            return
        self.want, self.after_verify = want, then
        self.verify_fails = self.verify_cycles = 0
        self.said_unsure, self.verified = False, False
        self.hud.request()
        self._go(VERIFY, now)

    def _verify_step(self, now: float, pad) -> str | None:
        r = self.hud.poll()
        if r is None:
            if not self.hud.busy:              # nothing pending (e.g. after a cycle): look again
                self.hud.request()
            return None
        item, taken = r
        if taken < self.t_state:               # a screenshot from before the last d-pad press
            self.hud.request()
            return None
        if item is not None:
            self.known = True
            self.cycles = {TRINKET: self.trinket_at, KNIVES: self.knife_at or 0}.get(item, 0) % self.n_items
            if item == self.want:
                self.verified = True
                self._go(self.after_verify, now)
                return None
            self.verify_cycles += 1
            if self.verify_cycles > 4:
                self._go(IDLE, now)
                return f"screen keeps showing the {item}, not the {self.want}: giving up"
            self._cycle(now, pad)
            self.t_state = now + CYCLE_MS + AFTER_CYCLE_MS
            return f"screen shows the {item} selected, not the {self.want}: switching"
        self.verify_fails += 1
        if self.verify_fails < VERIFY_TRIES:
            self.hud.request()
            return None
        if not self.strict:                    # elsewhere a wrong press costs nothing much
            self.verified = True
            self._go(self.after_verify, now)
            return None
        # our own dugout and the screen is unreadable: don't risk a sign in here, ever
        self.wait_until = now + VERIFY_RETRY_MS
        self.t_state = self.wait_until
        if not self.said_unsure:
            self.said_unsure = True
            return ("can't see which quick item is selected (is the game on screen?): not "
                    "pressing use in the dugout until I can")
        return None

    @staticmethod
    def _idle(p) -> bool:
        return p.sub in (Sub.NEUTRAL, Sub.RECOVER) and p.animation_type == 0

    # ------------------------------------------------------------------ main
    def update(self, p, at_home: bool, now: float, pad, steer, cam_yaw,
               others_in_arena: int | None = None, others_in_world: int | None = None,
               summoned: bool = False) -> str | None:
        """Call every tick the bot drives. Presses buttons only while active.
        at_home: allowed to run (home, or summoned into an arena world with the same dugout).
        Returns a log line on state changes. state == DONE means: place a sign now (at home)."""
        self.summoned_now = summoned
        if not at_home or p is None or not self.ready:
            if self.active:
                self._go(IDLE, now)
            return None
        if self.state in (IDLE, DONE):
            # 1 HP = round lost: off to the dugout - but only if someone else is waiting for a
            # turn. One-on-one (one other person in the world) stay put (user, 2026-11-23).
            if 0 < p.hp <= 1 and not self.in_dugout(p) and (others_in_world or 0) >= 2:
                self._go(OUT_SETTLE, now)
                return "1 HP (round lost): leaving the arena with the Traversal Trinket"
            if self.in_dugout(p):
                msg = "reloaded in the dugout: heading back to the arena"
                if self.state == DONE and self.done_at is not None and now - self.done_at < 8000:
                    # "placing a sign" sent us back here: that was the trinket, not the soapstone
                    self.cycles = self.trinket_at % self.n_items
                    msg = "the sign press used the trinket (teleported back): fixing the item count"
                self.tries = 0
                self._go(SETTLE, now)
                return msg
            return None

        # finish any button press in progress
        if now < self.cycle_until:
            pad.hat = DDOWN
            return None
        if now < self.press_until:
            pad.buttons |= SQUARE
            return None
        if now < self.wait_until:
            return None

        if self.state == VERIFY:
            self.strict = not summoned and self.in_dugout(p)
            return self._verify_step(now, pad)

        if self.state == SETTLE:
            if p.hp > 0 and now - self.t_state > SETTLE_MS and self._idle(p):
                self.seen_two, self.said_wait, self.clear_since = False, False, None
                self._go(WAIT_TURN, now)
            return None

        if self.state == WAIT_TURN:
            n = others_in_arena or 0
            if n >= 2:
                self.seen_two = True
                self.clear_since = None
                if not self.said_wait:
                    self.said_wait = True
                    return "two in the arena: waiting my turn"
                return None
            if self.seen_two:
                # someone left: make sure they stay gone (not a blip in the count)
                self.clear_since = self.clear_since or now
                if now - self.clear_since < TURN_CLEAR_MS:
                    return None
            # one or nobody in the arena: our turn now (user, 2026-11-23: it sat in the dugout
            # "giving a challenger" 12 s while the host waited alone in the arena)
            why = ("one of them left: my turn" if self.seen_two else
                   "arena's empty: going in" if n == 0 else "their turn's over: going in")
            self._go(SELECT_TRINKET, now)
            return why
            return None

        # ---- arena -> dugout at 1 HP
        if self.state == OUT_SETTLE:
            if self.in_dugout(p):
                self._go(SETTLE, now)
                return None
            if now - self.t_state > OUT_SETTLE_MS and self._idle(p):
                self._go(OUT_SELECT, now)
            return None
        if self.state == OUT_SELECT:
            if self.cycles != self.trinket_at % self.n_items:
                self._cycle(now, pad)
                return None
            self.tries = 0
            self._verify(TRINKET, OUT_USE, now)
            return None
        if self.state == OUT_USE:
            if not self._idle(p):
                return None
            self.press_until = now + PRESS_MS
            self.anim_seen = False
            pad.buttons |= SQUARE
            self._go(OUT_LEAVING, now)
            return None
        if self.state == OUT_LEAVING:
            if p.animation_type != 0:
                self.anim_seen = True
            if not self.anim_seen and not self.in_dugout(p) and now - self.t_state > NO_ANIM_RETRY_MS:
                self.tries += 1
                if self.tries >= MAX_TRIES + 2:
                    self._go(IDLE, now)
                    return "use item keeps not registering: giving up on the dugout"
                self._go(OUT_USE, now)
                return f"use item didn't register: pressing again ({self.tries})"
            if self.in_dugout(p):
                self.cycles = self.trinket_at % self.n_items     # it worked: the trinket is selected
                self.known = True
                self._go(SETTLE, now)
                return "in the dugout"
            if now - self.t_state > LEAVE_MS:
                self.tries += 1
                if self.tries >= MAX_TRIES:
                    self._go(IDLE, now)
                    return f"trinket didn't take us to the dugout after {MAX_TRIES} tries: giving up"
                if self.anim_seen:                 # used an item, didn't move: the soapstone
                    self.cycles = 0
                    self._cycle(now, pad)
                    self._verify(TRINKET, OUT_USE, now + CYCLE_MS + AFTER_CYCLE_MS)
                    return f"used the soapstone, not the trinket: switching ({self.tries}/{MAX_TRIES})"
                self._verify(TRINKET, OUT_USE, now)
                return f"use item didn't register: retrying ({self.tries}/{MAX_TRIES})"
            return None

        if self.state == SELECT_TRINKET:
            if self.cycles != self.trinket_at % self.n_items:
                self._cycle(now, pad)
                return None
            self._verify(TRINKET, USE, now)
            return "Traversal Trinket selected (checking the screen)" if self.hud else "Traversal Trinket selected"

        if self.state == USE:
            if not self._idle(p):
                return None
            self.press_until = now + PRESS_MS
            self.anim_seen = False
            pad.buttons |= SQUARE
            self._go(LEAVING, now)
            return "using the Traversal Trinket"

        if self.state == LEAVING:
            if p.animation_type != 0:
                self.anim_seen = True
            if not self.anim_seen and self.in_dugout(p) and now - self.t_state > NO_ANIM_RETRY_MS:
                self.tries += 1
                if self.tries >= MAX_TRIES + 2:
                    self._go(IDLE, now)
                    return f"use item keeps not registering: giving up"
                self._go(USE, now)
                return f"use item didn't register: retrying ({self.tries}/{MAX_TRIES})"
            if not self.in_dugout(p):
                self.cycles = self.trinket_at % self.n_items       # it teleported: the trinket is selected
                self.known = True
                self._go(WALK, now)
                return "out of the dugout: walking to the arena centre"
            if now - self.t_state > LEAVE_MS:
                self.tries += 1
                if self.tries >= MAX_TRIES:
                    self._go(IDLE, now)
                    return f"trinket didn't get us out after {MAX_TRIES} tries: giving up"
                if self.anim_seen:                   # used an item, didn't move: the soapstone
                    self.cycles = 0
                    self._cycle(now, pad)
                    self._verify(TRINKET, USE, now + CYCLE_MS + AFTER_CYCLE_MS)
                    return f"used the soapstone, not the trinket: switching and retrying ({self.tries}/{MAX_TRIES})"
                self._verify(TRINKET, USE, now)
                return f"use item didn't register: retrying ({self.tries}/{MAX_TRIES})"
            return None

        if self.state == WALK:
            c = self.arena["center"]
            dx, dy = c["x"] - p.loc_x, c["y"] - p.loc_y
            if math.hypot(dx, dy) < ARRIVE_RADIUS:
                self._go(SELECT_SOAP, now)
                return "at the arena centre"
            if now - self.t_state > WALK_TIMEOUT_MS:
                self._go(SELECT_SOAP, now)
                return "couldn't reach the arena centre in time: signing here"
            if cam_yaw is not None:
                pad.lx, pad.ly = steer.stick_for_heading(heading(dx, dy), cam_yaw)
            return None

        if self.state == SELECT_SOAP:
            if self.cycles != 0:
                self._cycle(now, pad)
                return None
            if self.hud is not None and not self.verified:
                self._verify(SOAPSTONE, SELECT_SOAP, now)   # look first: a trinket press would undo the walk
                return None
            self.verified = False
            self._go(DONE, now)
            self.done_at = now
            return "back in the arena: my turn" if summoned else "soapstone selected: placing a sign"
        return None
