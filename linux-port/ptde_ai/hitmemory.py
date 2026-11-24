"""How much each enemy attack actually hurts, learned from the hits the bot takes.

Keyed by weapon class (r_weapon // 1000, same as the poise table) and animation type.
Saved to models/hit_damage.json so it carries over between runs. Used to decide whether
tanking a hit (poise through it) is worth it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "models" / "hit_damage.json"


class HitMemory:
    def __init__(self, path: Path | None = PATH):
        self.path = path
        self.table: dict[str, float] = {}
        if path is not None and path.exists():
            try:
                self.table = json.loads(path.read_text())
            except (OSError, ValueError):
                self.table = {}
        self._dirty_since = None

    @staticmethod
    def key(weapon_id: int, anim_type: int) -> str:
        return f"{weapon_id // 1000}:{anim_type}"

    def get(self, weapon_id: int, anim_type: int) -> float | None:
        return self.table.get(self.key(weapon_id, anim_type))

    def record(self, weapon_id: int, anim_type: int, damage: float) -> None:
        k = self.key(weapon_id, anim_type)
        old = self.table.get(k)
        # keep the worst case seen, smoothed a little so one counter-hit doesn't dominate
        self.table[k] = damage if old is None else max(damage, 0.7 * old + 0.3 * damage)
        self._dirty_since = self._dirty_since or time.monotonic()

    def record_reach(self, weapon_id: int, dist: float) -> None:
        """Farthest distance this weapon has hit us from (its reach)."""
        k = f"reach:{weapon_id // 1000}"
        if dist > self.table.get(k, 0.0):
            self.table[k] = round(dist, 2)
            self._dirty_since = self._dirty_since or time.monotonic()

    def record_reach_start(self, weapon_id: int, anim_type: int, dist: float, anim_id: int = 0) -> None:
        """How far away this attack STARTED and still hit us (lunges close distance while they
        wind up, so the distance at the hit undersells them). 2026-11-24: rolling attacks,
        R2s and backstep attacks hit from 2.5-7 m, outside a flat 3 m guess."""
        if not 0.3 < dist < 9.0:
            return
        keys = [f"reach0:{weapon_id // 1000}:{anim_type}"]
        if anim_id > 0:              # exact attack: an R1 and a running attack share a type
            keys.append(f"reach0id:{weapon_id // 1000}:{anim_id}")
        for k in keys:
            if dist > self.table.get(k, 0.0):
                self.table[k] = round(dist, 2)
                self._dirty_since = self._dirty_since or time.monotonic()

    def reach_start(self, weapon_id: int, anim_type: int, anim_id: int = 0) -> float | None:
        """The exact attack's reach if we've been hit by it, else its type's."""
        if anim_id > 0:
            r = self.table.get(f"reach0id:{weapon_id // 1000}:{anim_id}")
            if r is not None:
                return r
        return self.table.get(f"reach0:{weapon_id // 1000}:{anim_type}")

    def record_hit_time(self, anim_id: int, timer: float) -> None:
        """How far into attack `anim_id` its hit landed on us (earliest seen = when to dodge)."""
        if anim_id <= 1000 or not 0.15 < timer < 3.0:   # faster = timer not reset yet
            return
        k = f"hit_t:{anim_id}"
        if timer < self.table.get(k, 99.0):
            self.table[k] = round(timer, 3)
            self._dirty_since = self._dirty_since or time.monotonic()

    def hit_time(self, anim_id: int) -> float | None:
        return self.table.get(f"hit_t:{anim_id}")

    def record_value(self, key: str, value: float, keep: float = 0.8) -> None:
        """A smoothed running value (e.g. how long sorceries take to land)."""
        old = self.table.get(key)
        self.table[key] = round(value if old is None else keep * old + (1 - keep) * value, 3)
        self._dirty_since = self._dirty_since or time.monotonic()

    def value(self, key: str, default: float) -> float:
        return self.table.get(key, default)

    def reach(self, weapon_id: int) -> float | None:
        return self.table.get(f"reach:{weapon_id // 1000}")

    def maybe_save(self, force: bool = False) -> None:
        if self.path is None or self._dirty_since is None:
            return
        if force or time.monotonic() - self._dirty_since > 5:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # merge with what's on disk (seeding tools / another run may have added entries)
            try:
                disk = json.loads(self.path.read_text()) if self.path.exists() else {}
            except (OSError, ValueError):
                disk = {}
            for k, v in disk.items():
                if k.startswith(("reach", "reach0")) and k in self.table:
                    self.table[k] = max(self.table[k], v)      # reach: farthest wins
                else:
                    self.table.setdefault(k, v)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.table, indent=1, sort_keys=True))
            tmp.replace(self.path)
            self._dirty_since = None
