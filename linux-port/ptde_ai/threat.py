"""Backstab threat: how likely is the opponent to backstab us in the next second?

Replaces metal-crow's backstab net (2026-11-24). His 8 inputs took absolute values of world
coordinates (the same situation looked different depending on where in the map it was) and
were trained on his own 2015 fights. These inputs are all relative to the bot, and the net is
trained on the bot's own recorded fights (tools/train_threat.py), so it learns how the people
who actually fight it set up backstabs. The same code computes the inputs for training and
live, so they can't drift apart.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

NAMES = ["dist", "their_fwd", "their_side", "bearing", "aim", "align", "closing", "circling",
         "their_speed", "our_speed", "our_turn", "their_turn", "we_busy", "they_busy"]
N = len(NAMES)
SPAN_MS = 150              # velocities over this long
MAX_SPEED = 12.0           # m/s; above this it's a teleport / target switch
MAX_TURN = 1440.0          # deg/s


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ang(ax, ay, bx, by) -> float:
    """Unsigned angle between two vectors, 0..1 (x 180 degrees)."""
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.5
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.acos(c) / math.pi


class ThreatFeatures:
    """Feed every tick with positions/rotations; features() gives the net's inputs."""

    def __init__(self):
        self.hist: deque = deque()          # (t, px, py, prot, ex, ey, erot)

    def reset(self) -> None:
        self.hist.clear()

    def push(self, t, px, py, prot, ex, ey, erot) -> None:
        if self.hist and math.hypot(ex - self.hist[-1][4], ey - self.hist[-1][5]) > 1.5:
            self.hist.clear()                # target switch / teleport: start over
        self.hist.append((t, px, py, prot, ex, ey, erot))
        while len(self.hist) > 2 and t - self.hist[1][0] >= SPAN_MS:
            self.hist.popleft()

    def features(self, we_busy: bool, they_busy: bool) -> np.ndarray | None:
        if len(self.hist) < 2:
            return None
        t, px, py, pr, ex, ey, er = self.hist[-1]
        t0, px0, py0, pr0, ex0, ey0, er0 = self.hist[0]
        dt = (t - t0) / 1000.0
        if dt < 0.05:
            return None
        dx, dy = ex - px, ey - py
        d = math.hypot(dx, dy)
        fx, fy = -math.sin(pr), -math.cos(pr)          # our facing
        gx, gy = -math.sin(er), -math.cos(er)          # theirs
        fwd = dx * fx + dy * fy                          # + in front of us, - behind
        side = abs(dx * fy - dy * fx)
        bearing = _ang(fx, fy, dx, dy)                   # 0 in front .. 1 behind us
        aim = _ang(gx, gy, -dx, -dy)                     # 0 = they face us
        align = _ang(fx, fy, gx, gy)                     # 0 = facing the same way
        d0 = math.hypot(ex0 - px0, ey0 - py0)
        closing = max(-MAX_SPEED, min(MAX_SPEED, (d - d0) / dt))
        vex, vey = (ex - ex0) / dt, (ey - ey0) / dt
        vpx, vpy = (px - px0) / dt, (py - py0) / dt
        rx, ry = vex - vpx, vey - vpy
        circling = abs(rx * dy - ry * dx) / d if d > 1e-3 else 0.0
        clamp = lambda v: min(MAX_SPEED, v)
        our_turn = min(MAX_TURN, abs(math.degrees(_wrap(pr - pr0))) / dt)
        their_turn = min(MAX_TURN, abs(math.degrees(_wrap(er - er0))) / dt)
        return np.array([d, fwd, side, bearing, aim, align, closing, clamp(circling),
                         clamp(math.hypot(vex, vey)), clamp(math.hypot(vpx, vpy)),
                         our_turn, their_turn, float(we_busy), float(they_busy)])
