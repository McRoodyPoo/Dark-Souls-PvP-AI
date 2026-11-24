"""Per-opponent adaptation: remember who we're fighting and what works against them.

User 2026-11-24: "so he can remember names and play accordingly?" -> yes. Opponents are keyed
by SteamID64 (names change, the ID doesn't); records/profiles.json holds, per opponent and
pooled over everyone ("_all"), a score for each style the bot can play.

Two choices adapt, both inside RULES.md (styles only move the brain's existing dials):
  * neutral style, picked per EXCHANGE: from both alive in range until 1.5 s after the first
    real damage (so a poise trade resolves), or 25 s with nothing. Score = our share of the
    damage dealt in it (1 = all theirs, 0 = all ours, 0.5 = nothing happened).
  * wake-up style, picked per KNOCKDOWN: 1 if we got up and stayed un-hit for 2.5 s, else 0.
Picking is Thompson sampling: each style's record is a Beta(wins, losses); sample each, play
the best sample. Styles that work get played more, but every style keeps some chance, so the
bot stays hard to read and notices when someone adapts. Every result also goes into the record
against everyone ("_all"), which remembers far longer: what's learned on one player becomes
the starting point for the next. New opponents start from that record (capped, so their own
results take over quickly).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "records" / "profiles.json"
ALL = "_all"

# Each style is a set of dial values for the brain (brain.DEFAULT_STYLE has the defaults).
NEUTRAL_STYLES = {
    "balanced": {},                                                    # the hand-tuned defaults
    "pressure": {"race_static_ms": 600, "run_bs_closing": 2.0, "footsie_in_m": 0.0, "footsie_out_m": 1.0},
    "bait":     {"race_static_ms": 2000, "run_bs_closing": 1.8, "footsie_in_m": 0.5, "footsie_out_m": 1.8},
    "racer":    {"race_static_ms": 300, "run_bs_closing": 2.8},
}
WAKEUP_STYLES = {
    "quick": {"wait_out_chance": 0.1},
    "mix":   {"wait_out_chance": 0.4},
    "stay":  {"wait_out_chance": 0.8},
}
DIMS = {"neutral": NEUTRAL_STYLES, "wakeup": WAKEUP_STYLES}

PRIOR_CAP = 6.0          # pooled results count as at most this many fights' worth for a newcomer
FADE = 0.985             # per opponent: each new result shrinks the old ones a little
                         # (people change how they play: ~70 exchanges of memory)
FADE_ALL = 0.999         # the record against everyone absorbs every player's results and
                         # keeps ~1000 exchanges (user 2026-11-24: profiles feed the general model)
EXCHANGE_RANGE_M = 6.0
DAMAGE_MIN = 120         # HP lost that counts as "something happened"
RESOLVE_MS = 1500        # after the first damage, keep counting this long (trades)
EXCHANGE_MAX_MS = 25000
REST_MS = 1500           # gap between exchanges
WAKE_SAFE_MS = 2500
BS_REWARD = 0.9         # an exchange where we land a backstab scores at least this
SETUP_STABLE_MS = 5000   # a weapon held this long counts as their setup (quick swaps don't)


class Profiles:
    def __init__(self, path: Path | None = PATH, rng: random.Random | None = None):
        self.path = path
        self.rng = rng or random.Random()
        self.data: dict = {}
        self._dirty_t = None
        if path is not None and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (OSError, ValueError):
                self.data = {}

    def _rec(self, key: str, dim: str) -> dict:
        p = self.data.setdefault(key, {})
        d = p.setdefault(dim, {})
        for arm in DIMS[dim]:
            d.setdefault(arm, [0.0, 0.0])
        return d

    def note_name(self, sid: str, name: str) -> None:
        p = self.data.setdefault(sid, {})
        p["name"] = name
        p["seen"] = time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def levels(sid: str | None, setup: str | None) -> list[str]:
        """General -> specific: everyone, everyone with this setup, this player, this player
        with this setup (user 2026-11-24: players switch setups)."""
        out = [ALL]
        if setup:
            out.append(f"w:{setup}")
        if sid:
            out.append(sid)
            if setup:
                out.append(f"{sid}|{setup}")
        return out

    def posterior(self, sid: str | None, dim: str, setup: str | None = None) -> dict[str, tuple[float, float]]:
        """Each more general level is a prior for the next, worth at most PRIOR_CAP results, so
        the most specific record takes over quickly once it has data."""
        lv = self.levels(sid, setup)
        recs = [self._rec(k, dim) for k in lv]
        out = {}
        for arm in DIMS[dim]:
            a = b = 1.0
            for i, rec in enumerate(recs):
                w, l = rec[arm]
                n = w + l
                k = 1.0 if i == len(recs) - 1 else (min(1.0, PRIOR_CAP / n) if n else 0.0)
                a, b = a + w * k, b + l * k
            out[arm] = (a, b)
        return out

    def choose(self, sid: str | None, dim: str, setup: str | None = None) -> str:
        post = self.posterior(sid, dim, setup)
        return max(post, key=lambda arm: self.rng.betavariate(*post[arm]))

    def best(self, sid: str | None, dim: str, setup: str | None = None) -> tuple[str, float]:
        post = self.posterior(sid, dim, setup)
        arm = max(post, key=lambda a: post[a][0] / sum(post[a]))
        return arm, post[arm][0] / sum(post[arm])

    def record(self, sid: str | None, dim: str, arm: str, score: float, now: float = 0.0,
               setup: str | None = None) -> None:
        score = max(0.0, min(1.0, score))
        for key in self.levels(sid, setup):
            d = self._rec(key, dim)
            fade = FADE_ALL if key == ALL or key.startswith("w:") else FADE
            for a in d:
                d[a] = [d[a][0] * fade, d[a][1] * fade]
            d[arm][0] += score
            d[arm][1] += 1.0 - score
        self._dirty_t = self._dirty_t or now

    def save(self, now: float = 0.0, force: bool = False) -> None:
        if self.path is None or self._dirty_t is None:
            return
        if not force and now - self._dirty_t < 10000:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)
        self._dirty_t = None


class Exchanges:
    """Cuts a fight into exchanges and scores each (shared by the live bot and the seeding tool)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.t0 = self.rest_until = 0.0
        self.hp0 = None
        self.first_damage_t = None
        self.bs_us = self.bs_them = False

    def backstab(self, by_us: bool) -> None:
        """A backstab landed in this exchange (user 2026-11-24: backstabs deserve a bigger reward)."""
        if self.active:
            if by_us:
                self.bs_us = True
            else:
                self.bs_them = True

    def update(self, t: float, p_hp: int, e_hp: int, dist: float) -> tuple[str, float | None]:
        """Returns ("start", None), ("end", score) or ("", None)."""
        if p_hp <= 1 or e_hp <= 1:                  # 1 HP = round over (arena rules)
            if self.active:
                score = self._score(p_hp, e_hp)
                self.reset()
                self.rest_until = t + REST_MS
                return "end", score
            return "", None
        if not self.active:
            if t >= self.rest_until and dist <= EXCHANGE_RANGE_M:
                self.active, self.t0, self.hp0, self.first_damage_t = True, t, (p_hp, e_hp), None
                return "start", None
            return "", None
        if e_hp > self.hp0[1] + 50 or p_hp > self.hp0[0] + 50:
            self.hp0 = (max(self.hp0[0], p_hp), max(self.hp0[1], e_hp))   # healed: rebase
        lost_p, lost_e = self.hp0[0] - p_hp, self.hp0[1] - e_hp
        if self.first_damage_t is None and max(lost_p, lost_e) >= DAMAGE_MIN:
            self.first_damage_t = t
        done = ((self.first_damage_t is not None and t - self.first_damage_t >= RESOLVE_MS)
                or t - self.t0 >= EXCHANGE_MAX_MS)
        if done:
            score = self._score(p_hp, e_hp)
            self.reset()
            self.rest_until = t + REST_MS
            return "end", score
        return "", None

    def _score(self, p_hp: int, e_hp: int) -> float:
        if e_hp <= 1 < p_hp:
            return 1.0                               # we ended their round
        if p_hp <= 1 < e_hp:
            return 0.0                               # they ended ours
        share = self._share(p_hp, e_hp)
        if self.bs_us and not self.bs_them:
            return max(share, BS_REWARD)             # our backstab: a good exchange, whatever else
        if self.bs_them and not self.bs_us:
            return min(share, 1.0 - BS_REWARD)
        return share

    def _share(self, p_hp: int, e_hp: int) -> float:
        lost_p = max(0, self.hp0[0] - max(p_hp, 0))
        lost_e = max(0, self.hp0[1] - max(e_hp, 0))
        if lost_p + lost_e < DAMAGE_MIN:
            return 0.5
        return lost_e / (lost_p + lost_e)


class Adapter:
    """Live: picks styles for the brain and scores them as the fight goes."""

    def __init__(self, profiles: Profiles):
        self.profiles = profiles
        self.sid: str | None = None
        self.name = "?"
        self.ex = Exchanges()
        self.neutral = "balanced"
        self.wakeup = "mix"
        self._down_t = None
        self._up_t = None
        self._wake_hp = None
        self._anims = (0, 0)
        self.setup: str | None = None
        self._seen_setup: str | None = None
        self._seen_since = 0.0

    def set_opponent(self, sid, name: str) -> str:
        self.abort()
        self.sid = str(sid) if sid else None
        self.name = name or "?"
        if self.sid:
            self.profiles.note_name(self.sid, self.name)
        return f"profile: {self.name}: " + self.summary()

    def summary(self, setup: str | None = None) -> str:
        """Styles tried against this opponent (with this setup, if given) and their scores."""
        if not self.sid and not setup:
            return "unidentified: using the record against everyone"
        key = self.sid if not setup else (f"{self.sid}|{setup}" if self.sid else f"w:{setup}")
        rec = self.profiles._rec(key, "neutral")
        if setup and sum(sum(v) for v in rec.values()) < 3:
            w = self.profiles._rec(f"w:{setup}", "neutral")
            n = sum(sum(v) for v in w.values())
            return (f"new setup for them: starting from their usual record"
                    + (f" and {n:.0f} exchanges against anyone with these weapons" if n >= 3 else ""))
        tried = {a: w / (w + l) for a, (w, l) in rec.items() if w + l >= 3}
        untried = [a for a in rec if a not in tried]
        parts = [f"{a} {v:.0%}" for a, v in sorted(tried.items(), key=lambda kv: -kv[1])]
        txt = ("share of damage won by style: " + ", ".join(parts)) if parts else "no exchanges yet"
        return txt + (f"; untried: {', '.join(untried)}" if untried else "")

    def abort(self) -> None:
        """Target switch / fight over without a result: drop the open exchange unscored."""
        self.ex.reset()
        self._down_t = self._up_t = None
        self.setup = self._seen_setup = None

    def style(self) -> dict:
        s = dict(NEUTRAL_STYLES[self.neutral])
        s.update(WAKEUP_STYLES[self.wakeup])
        return s

    @staticmethod
    def setup_of(e) -> str | None:
        """Their main weapon (right hand, by weapon class: upgrades/infusions don't count).
        The exact pair in hand changes every few seconds as people swap, which split player A
        into 35 'setups' with a few exchanges each."""
        r = getattr(e, "r_weapon", 0) or 0
        return str(r // 1000) if r > 0 else None

    def update(self, now: float, p, e, dist: float, down: bool) -> list[str]:
        out = []
        setup = self.setup_of(e)
        if setup != self._seen_setup:
            self._seen_setup, self._seen_since = setup, now
        elif setup and setup != self.setup and now - self._seen_since >= SETUP_STABLE_MS:
            if self.setup is not None:
                out.append(f"{self.name} switched to weapon class {setup}: " + self.summary(setup))
            self.setup = setup
        if self.setup is None and setup:
            self.setup = setup
        pa, ea = getattr(p, "animation_type", 0), getattr(e, "animation_type", 0)
        if pa == 225 and self._anims[0] != 225 and ea != 225:
            self.ex.backstab(by_us=True)
        elif ea == 225 and self._anims[1] != 225 and pa != 225:
            self.ex.backstab(by_us=False)
        self._anims = (pa, ea)
        ev, score = self.ex.update(now, p.hp, e.hp, dist)
        if ev == "start":
            self.neutral = self.profiles.choose(self.sid, "neutral", self.setup)
        elif ev == "end":
            self.profiles.record(self.sid, "neutral", self.neutral, score, now, self.setup)
            out.append(f"exchange vs {self.name}: style '{self.neutral}' scored {score:.2f}")
        # wake-up: one result per knockdown
        if down and self._down_t is None and p.hp > 0:
            self._down_t, self._up_t = now, None
            self.wakeup = self.profiles.choose(self.sid, "wakeup", self.setup)
        elif self._down_t is not None and not down and self._up_t is None:
            self._up_t, self._wake_hp = now, p.hp
        if self._up_t is not None:
            hit = p.hp < self._wake_hp - 50 or p.hp <= 0
            if hit or now - self._up_t >= WAKE_SAFE_MS:
                self.profiles.record(self.sid, "wakeup", self.wakeup, 0.0 if hit else 1.0, now,
                                     self.setup)
                out.append(f"wake-up vs {self.name}: '{self.wakeup}' "
                           f"{'got punished' if hit else 'got up safely'}")
                self._down_t = self._up_t = None
        self.profiles.save(now)
        return out
