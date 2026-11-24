"""Record every fight tick for later training.

One gzip'd JSON-lines file per run in recordings/. Each line is one tick while a real
opponent is present (NPCs are skipped), whoever is driving:
  {"t": ms, "mode": "bot"|"you", "self_type": 2,
   "p": {raw player memory}, "e": {raw opponent memory},
   "p_sub": 30, "e_sub": 11, "e_dodge_left": 0.12,
   "pad": [lx, ly, rx, ry, buttons, hat],
   "dodge": "STANDARD_ROLL", "attack": "NONE"}
Events (summoned, back home, deaths) are lines with an "event" key.
"""
from __future__ import annotations

import gzip
import json
import time
import zlib
from pathlib import Path

REC_DIR = Path(__file__).resolve().parent.parent / "recordings"


def _clean(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


class Recorder:
    def __init__(self, directory: Path = REC_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime("%Y%m%d-%H%M%S.jsonl.gz")
        self._f = gzip.open(self.path, "at", compresslevel=6)
        self.ticks = 0
        self._last_hp = {}
        self._last_flush = time.monotonic()

    def tick(self, now_ms: float, mode: str, self_type, p, e, pad, brain=None) -> None:
        row = {
            "t": round(now_ms, 1), "mode": mode, "self_type": self_type,
            "p": _clean(p.raw), "e": _clean(e.raw),
            "p_sub": int(p.sub), "e_sub": int(e.sub),
            "e_dodge_left": round(e.dodge_time_remaining, 4),
            "pad": [pad.lx, pad.ly, pad.rx, pad.ry, pad.buttons, pad.hat],
        }
        if brain is not None:
            row["dodge"] = brain.dodge_type.name if brain.in_dodge() else "NONE"
            row["attack"] = brain.attack_type.name if brain.in_attack() else "NONE"
        self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.ticks += 1
        self._maybe_flush()
        for who, c in (("bot", p), ("opponent", e)):
            if c.hp <= 0 < self._last_hp.get(who, 1):
                self.event(now_ms, f"{who} died")
            self._last_hp[who] = c.hp

    def event(self, now_ms: float, what: str) -> None:
        self._f.write(json.dumps({"t": round(now_ms, 1), "event": what}) + "\n")
        self._maybe_flush(force=True)

    def _maybe_flush(self, force: bool = False) -> None:
        """Sync-flush every couple of seconds so a crash loses at most that much."""
        now = time.monotonic()
        if force or now - self._last_flush > 2.0:
            self._f.flush()
            self._f.buffer.flush(zlib.Z_SYNC_FLUSH) if hasattr(self._f, "buffer") else None
            self._last_flush = now

    def close(self) -> None:
        self._f.close()


def load(path) -> list[dict]:
    """Read a recording, including one still being written or cut off by a crash."""
    d = zlib.decompressobj(31)                  # 31: gzip header
    raw = d.decompress(Path(path).read_bytes())
    lines = raw.decode("utf-8", "replace").split("\n")
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass                            # last line cut mid-write
    return out
