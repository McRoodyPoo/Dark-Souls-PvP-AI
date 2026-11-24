"""Who the bot fought and how it went: records/fights.jsonl, one line per finished fight.

  {"time": "...", "result": "won" | "lost", "steam_id": ..., "steam_name": ..., "char_name": ...,
   "profile": "https://steamcommunity.com/profiles/...", "bot_hp": ..., "their_hp": ..., "secs": ...}

"won" = the opponent's round ended while the bot was still in it (they lost to the bot).
A round ends at 1 HP for most players in the arena (user 2026-11-24: "1HP means round over
for most people"), or at death. Until then only deaths counted, so 0-15 was really ~5-15.
tools/defeated.py lists everyone who has lost to the bot.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "records" / "fights.jsonl"
RETRY_MS = 3000          # look the opponent up again this often until it works
ROUND_OVER_HP = 1        # 1 HP (or dead) = that player's round is over


class FightLog:
    def __init__(self, path: Path | None = PATH):
        self.path = path
        self.opponent = None          # players.PlayerId
        self.started = None           # ms
        self.done = False
        self._last_try = -1e9
        self._last = (None, None)     # (bot hp, their hp) last tick

    def new_fight(self) -> None:
        """Summoned (or back home): forget the last opponent."""
        self.opponent, self.started, self.done = None, None, False
        self._last = (None, None)

    def retarget(self) -> None:
        """Fighting a different character now: look them up again."""
        self.opponent = None
        self._last = (None, None)
        self._last_try = -1e9

    def want_lookup(self, now_ms: float) -> bool:
        if self.opponent is not None or now_ms - self._last_try < RETRY_MS:
            return False
        self._last_try = now_ms
        return True

    def set_opponent(self, pid) -> str | None:
        if pid is None or (self.opponent and self.opponent.steam_id == pid.steam_id):
            return None
        self.opponent = pid
        name = pid.steam_name + (f" (character {pid.char_name})" if pid.char_name else "")
        return f"opponent: {name}, {pid.profile}"

    def tick(self, bot_hp: int, their_hp: int, now_ms: float) -> str | None:
        """Call every engaged tick. Returns "won"/"lost" once when the fight ends."""
        if self.started is None:
            self.started = now_ms
        last_bot, last_them = self._last
        self._last = (bot_hp, their_hp)
        if self.done or last_bot is None:
            return None
        result = None
        if their_hp <= ROUND_OVER_HP < last_them and bot_hp > ROUND_OVER_HP:
            result = "won"
        elif bot_hp <= ROUND_OVER_HP < last_bot:
            result = "lost"
        if result:
            self.done = True
            self._write(result, last_bot if bot_hp <= ROUND_OVER_HP else bot_hp, their_hp, now_ms)
        return result

    def _write(self, result: str, bot_hp: int, their_hp: int, now_ms: float) -> None:
        o = self.opponent
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "result": result,
               "steam_id": o.steam_id if o else None, "steam_name": o.steam_name if o else None,
               "char_name": o.char_name if o else None, "profile": o.profile if o else None,
               "bot_hp": bot_hp, "their_hp": their_hp,
               "secs": round((now_ms - (self.started or now_ms)) / 1000, 1)}
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
