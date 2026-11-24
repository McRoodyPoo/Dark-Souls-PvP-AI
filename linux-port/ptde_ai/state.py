"""Snapshot both characters (and the camera) from game memory each tick."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import offsets as O
from .memory import ProcMem

_READERS = {"f32": ProcMem.f32, "i32": ProcMem.i32, "u32": ProcMem.u32,
            "u16": ProcMem.u16, "u8": ProcMem.u8}


@dataclass
class Snapshot:
    values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None


class StateReader:
    """Resolves pointer chains every read.

    metal-crow cached the final addresses and re-resolved after loading screens;
    re-resolving each tick costs ~5 preads per field, cheap enough at 60 Hz and
    immune to pointers moving when a phantom joins or leaves.
    """

    def __init__(self, mem: ProcMem):
        self.mem = mem

    def _read_group(self, base: int, table: dict) -> Snapshot:
        snap = Snapshot()
        for name, (chain, typ) in table.items():
            try:
                addr = self.mem.chain(base, chain)
                snap.values[name] = _READERS[typ](self.mem, addr)
            except (OSError, ValueError, Exception) as e:  # unmapped/null link
                snap.errors[name] = str(e)
        return snap

    def enemy(self, slot: int = 1) -> Snapshot:
        """The character in `slot` of the character table (0 = us). Live 2026-11-23 a host
        had a second red phantom in slot 1 and the bot, reading only slot 1, stood idle next
        to the host (slot 2) or walked into a wall toward the other red."""
        if slot == 1:
            return self._read_group(O.ENEMY_BASE, O.ENEMY)
        table = {k: ([c[0], 4 * slot] + c[2:] if len(c) > 1 and c[1] == 0x4 else c, t)
                 for k, (c, t) in O.ENEMY.items()}
        return self._read_group(O.ENEMY_BASE, table)

    def slots(self, n: int = 10) -> list[dict]:
        """Cheap look at character slots 1..n-1: chr type, team, hp, position."""
        out = []
        m = self.mem
        try:
            arr = m.u32(m.u32(O.ENEMY_BASE) + 0x4)
        except (OSError, ValueError):
            return out
        for i in range(1, n):
            try:
                c = m.u32(arr + 4 * i)
                if not c:
                    continue
                posp = m.u32(m.u32(c + 0x28) + 0x1C)
                motp = m.u32(m.u32(c + 0x28) + 0x54)          # same struct as ENEMY rotation/anim
                try:                                           # same struct as ENEMY r/l_weapon
                    eq = m.u32(m.u32(c + 0x34C) + 0x654)
                    r_w, l_w = m.u32(eq + 0x1F8), m.u32(eq + 0x1B8)
                except (OSError, ValueError):
                    r_w = l_w = 0
                out.append({"slot": i, "chr_type": m.i32(c + 0x70), "team_type": m.i32(c + 0x74),
                            "hp": m.u32(c + 0x2D4), "x": m.f32(posp + 0x10), "y": m.f32(posp + 0x18),
                            "rot": m.f32(motp + 0x34), "anim": m.u16(motp + 0x1EC),
                            "r_weapon": r_w, "l_weapon": l_w,
                            "entity_id": m.u32(c + O.ENTITY_ID_OFFSET)})
            except (OSError, ValueError):
                continue
        return out

    def player(self) -> Snapshot:
        snap = self._read_group(O.PLAYER_BASE, O.PLAYER)
        fast = self._read_group(O.ENEMY_BASE, O.SELF_POS)     # per-frame position
        for k, v in fast.values.items():
            snap.values[k + "_slow"] = snap.values.get(k)
            snap.values[k] = v
        return snap

    def self_types(self) -> Snapshot:
        return self._read_group(O.ENEMY_BASE, O.SELF)

    def quick_items(self) -> list[int]:
        snap = self._read_group(O.CHAR_DATA2_BASE, O.QUICK_ITEMS)
        return [snap.values.get(f"item{i + 1}", -1) for i in range(5)]

    def camera(self) -> Snapshot:
        return self._read_group(O.CAMERA_BASE, O.CAMERA)


def distance(a: Snapshot, b: Snapshot) -> float:
    return math.hypot(a.loc_x - b.loc_x, a.loc_y - b.loc_y)
