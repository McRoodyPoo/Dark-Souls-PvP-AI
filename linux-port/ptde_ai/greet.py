"""Bow to the host after being summoned, before fighting.

The bow is the first gesture in the user's gesture menu, so it's Back (gesture menu)
then A. The greeting is dropped the moment the host swings at us within reach.

Live 2026-11-23 the bot logged "bowed" six times but never actually bowed: it started the
moment our character type flipped to red phantom, while the summon was still loading /
fading in (input ignored, and the "enemy" slot still held the NPC from our own world).
Now it waits for a real host player plus a minimum arrival time, and checks that the bow
animation actually started, retrying if it didn't.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import geometry as G
from .character import Sub
from .gamepad import CIRCLE, CROSS, R3, SELECT

MIN_ARRIVE_MS = 4500      # after the summon starts: loading + materialising
START_DELAY_MS = 1000     # then this long idle with the host in view
FACE_GIVE_UP_MS = 4000    # can't face them in this long: skip the bow (never bow facing away)
FACE_HOLD_MS = 150        # facing them this long (stick centred) before the bow
FACE_DEG = 20.0
MENU_WAIT_MS = 600        # gesture menu opening
VERIFY_MS = 1200          # bow animation must start within this after A
BOW_MS = 2600             # let the bow play out
CLOSE_MS = 400            # B to close a menu that may be open, before retrying
MAX_TRIES = 3
TIMEOUT_MS = 25000        # from the summon starting
ATTACK_CONE_DEG = 45.0    # a swing counts as aimed at us only if they face us this squarely


def _faces(attacker, target, cone_deg: float) -> bool:
    """attacker is facing target (within cone_deg of straight at them)."""
    fx, fy = G.facing_vector(attacker)
    rx, ry = target.loc_x - attacker.loc_x, target.loc_y - attacker.loc_y
    n = math.hypot(rx, ry) or 1e-6
    return (rx * fx + ry * fy) / n > math.cos(math.radians(cone_deg))

IDLE, WAIT, FACE, OPEN, PICK, BOWING, CLOSE, DONE = (
    "idle", "wait", "face", "open", "pick", "bowing", "close", "done")


@dataclass
class Greeter:
    state: str = IDLE
    t_state: float = 0.0
    t_start: float = 0.0
    press_until: float = 0.0
    press_btn: int = 0
    tries: int = 0
    force: bool = False           # test hook: don't wait for a host player
    anim_before: int = -1
    face_ok_since: float | None = None

    @property
    def active(self) -> bool:
        return self.state not in (IDLE, DONE)

    def start(self, now: float, force: bool = False) -> None:
        self.state, self.t_state, self.t_start = WAIT, now, now
        self.tries, self.force = 0, force

    def _go(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    @staticmethod
    def _gesturing(p, anim_before: int) -> bool:
        return p.animation_type != 0 or p.raw.get("anim_id", -1) not in (-1, anim_before)

    def tick(self, p, e, now: float, pad, steer, cam_yaw, host_ok: bool = True) -> str | None:
        """Drive the pad while greeting. Returns a log line on changes.
        host_ok: e is a real player we can fight (not stale data from our own world)."""
        if not self.active:
            return None
        if now - self.t_start > TIMEOUT_MS:
            self._go(DONE, now)
            return "greeting timed out: fighting"
        if p is None or e is None or not (host_ok or self.force):
            self.t_state = now if self.state == WAIT else self.t_state
            return None
        dist = G.distance(p, e)
        # A swing only counts as an attack on US if they're facing us. Live 2026-11-24 the bot
        # was summoned into the middle of the host's fight with another player: the host's
        # swings at them read as "host attacked as we arrived" in 12 of 19 summons, so the bot
        # skipped the bow and joined their fight.
        swinging = e.sub in (Sub.WINDUP, Sub.WINDUP_CLOSING, Sub.WINDUP_GHOST_HIT,
                             Sub.ACTIVE_DURING_HURTBOX) and _faces(e, p, ATTACK_CONE_DEG)
        if swinging and dist <= e.weapon_range + 0.5 and self.state != WAIT:
            self._go(DONE, now)
            return "host attacked during the greeting: fighting"
        if now < self.press_until:
            pad.buttons |= self.press_btn
            return None

        if self.state == WAIT:
            if swinging and dist <= e.weapon_range + 0.5:
                self._go(DONE, now)
                return "host attacked as we arrived: fighting"
            if (p.animation_type == 0 and p.sub in (Sub.NEUTRAL, Sub.RECOVER)
                    and (self.force or now - self.t_start > MIN_ARRIVE_MS)):
                if now - self.t_state > START_DELAY_MS:
                    self._go(FACE, now)
            else:
                self.t_state = now                 # still arriving / busy
            return None

        if self.state == FACE:
            # Face them with the stick, unlocked (user 2026-11-24: face the opponent before
            # bowing). Lock-on grabs whoever's nearest the screen centre, and the old version
            # bowed anyway after 1.2 s whether it was facing them or not.
            if p.locked_on and now - self.t_state < 40:
                pad.buttons |= R3
            h = G.heading_to(p, e)
            facing = math.degrees(abs((G.facing_heading(p) - h + math.pi) % (2 * math.pi) - math.pi))
            if facing < FACE_DEG:
                self.face_ok_since = self.face_ok_since or now
                if now - self.face_ok_since >= FACE_HOLD_MS:
                    self.press_btn, self.press_until = SELECT, now + 120
                    self.tries += 1
                    self.face_ok_since = None
                    self._go(OPEN, now)
                    return "facing them: bowing" + (f" (try {self.tries})" if self.tries > 1 else "")
                return None                        # stick centred: settle facing them
            self.face_ok_since = None
            if now - self.t_state > FACE_GIVE_UP_MS:
                self._go(DONE, now)
                return "couldn't face them: skipping the bow"
            if cam_yaw is not None:
                pad.lx, pad.ly = steer.stick_for_heading(h, cam_yaw, magnitude=0.35)
            return None

        if self.state == OPEN:
            if now - self.t_state > 120 + MENU_WAIT_MS:
                self.anim_before = p.raw.get("anim_id", -1)
                self.press_btn, self.press_until = CROSS, now + 120
                self._go(PICK, now)
            return None

        if self.state == PICK:
            if self._gesturing(p, self.anim_before):
                self._go(BOWING, now)
                return "bow started"
            if now - self.t_state > 120 + VERIFY_MS:
                if self.tries >= MAX_TRIES:
                    self.press_btn, self.press_until = CIRCLE, now + 120
                    self._go(DONE, now)
                    return f"bow never started after {MAX_TRIES} tries: fighting"
                self.press_btn, self.press_until = CIRCLE, now + 120   # close the menu if open
                self._go(CLOSE, now)
                return "bow didn't start: retrying"
            return None

        if self.state == CLOSE:
            if now - self.t_state > 120 + CLOSE_MS:
                self._go(FACE, now)
            return None

        if self.state == BOWING:
            if now - self.t_state > BOW_MS:
                self._go(DONE, now)
                return "bowed: fighting"
            return None
        return None
