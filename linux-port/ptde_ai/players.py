"""Who is that? Steam name, SteamID64 and character name of a player character (read-only).

Found live 2026-11-23 (offsets from the user's DARKSOULS_Ashley.CT "[Steam]" section, the
character link by searching our own character for our own entry):
  * the game's connected-player list is an MSVC std::list: head node at +0xF62DCC, count at
    +0xF62DD0; each node's +0xC points at a 0x80-byte player entry:
        +0x30  Steam name (UTF-16)
        [+0xC] +0x28  SteamID64 (u64)   (+0x30 holds it as a hex string)
  * a player character links to its entry: [[ChrIns + 0x170] + 0x420]
  * character name: [ChrIns + 0x414] + 0xA0 (UTF-16)
The list also holds DSCM-connected players who aren't in our world, so a character is only
identified when its link lands on an entry that's in the list.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from . import offsets as O

NODE_LIST_HEAD = O.MODULE_BASE + 0xF62DCC
NODE_COUNT = O.MODULE_BASE + 0xF62DD0
MAX_NODES = 64
STEAMID_HIGH = 0x01100001       # individual account, public universe


@dataclass
class PlayerId:
    steam_id: int
    steam_name: str
    char_name: str = ""

    @property
    def profile(self) -> str:
        return f"https://steamcommunity.com/profiles/{self.steam_id}"


def _wstr(mem, addr: int, chars: int = 32) -> str:
    raw = mem.read(addr, chars * 2)
    return raw.decode("utf-16-le", errors="replace").split("\0")[0]


def _entry(mem, d: int) -> PlayerId | None:
    sid = struct.unpack("<Q", mem.read(mem.u32(d + 0xC) + 0x28, 8))[0]
    if sid >> 32 != STEAMID_HIGH:
        return None
    return PlayerId(sid, _wstr(mem, d + 0x30))


def session_players(mem) -> dict[int, PlayerId]:
    """Player entry address -> PlayerId, for everyone in the connected-player list."""
    out: dict[int, PlayerId] = {}
    try:
        head = mem.u32(NODE_LIST_HEAD)
        node = mem.u32(head)
        for _ in range(MAX_NODES):
            if node in (0, head):
                break
            d = mem.u32(node + 0xC)
            if d:
                pid = _entry(mem, d)
                if pid:
                    out[d] = pid
            node = mem.u32(node)
    except (OSError, ValueError, struct.error):
        pass
    return out


def identify(mem, chr_ins: int) -> PlayerId | None:
    """SteamID / names of the player character at chr_ins, or None (NPC, not connected)."""
    try:
        d = mem.u32(mem.u32(chr_ins + 0x170) + 0x420)
        known = session_players(mem)
        pid = known.get(d)
        if pid is None:
            return None
        try:
            pid.char_name = _wstr(mem, mem.u32(chr_ins + 0x414) + 0xA0, 16)
        except (OSError, ValueError, struct.error):
            pass
        return pid
    except (OSError, ValueError, struct.error):
        return None


def enemy_chr_ins(mem, slot: int = 1) -> int:
    """ChrIns address of the character in `slot` of the character table (the bot's target)."""
    return mem.u32(mem.chain(O.ENEMY_BASE, [0x4, 4 * slot]))


def self_chr_ins(mem) -> int:
    return mem.u32(mem.chain(O.ENEMY_BASE, [0x4, 0x0]))
