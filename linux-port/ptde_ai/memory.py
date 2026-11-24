"""Read-only access to the game's memory via /proc/<pid>/mem.

Nothing here writes to the game. Wine marks its processes ptrace-able by any
process of the same user (PR_SET_PTRACER_ANY), which is what lets this work with
kernel.yama.ptrace_scope=1; if opening fails, probe.py says so.
"""
from __future__ import annotations

import os
import re
import struct


class ProcMem:
    def __init__(self, pid: int):
        self.pid = pid
        self.fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)

    def close(self) -> None:
        os.close(self.fd)

    # --- primitive reads -------------------------------------------------
    def read(self, addr: int, size: int) -> bytes:
        return os.pread(self.fd, size, addr)

    def u8(self, addr: int) -> int:
        return self.read(addr, 1)[0]

    def u16(self, addr: int) -> int:
        return struct.unpack("<H", self.read(addr, 2))[0]

    def i32(self, addr: int) -> int:
        return struct.unpack("<i", self.read(addr, 4))[0]

    def u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def f32(self, addr: int) -> float:
        return struct.unpack("<f", self.read(addr, 4))[0]

    # --- pointer chains --------------------------------------------------
    def chain(self, base: int, offsets: list[int]) -> int:
        """Cheat Engine semantics: addr=[base]+o1; addr=[addr]+o2; ... return addr.

        The final address is *not* dereferenced, so read the field you want from it.
        Raises OSError/ValueError if any link is null or unmapped.
        """
        addr = base
        for off in offsets:
            ptr = self.u32(addr)
            if ptr == 0:
                raise ValueError(f"null pointer in chain at {addr:#x}")
            addr = ptr + off
        return addr

    # --- module + AOB scan ----------------------------------------------
    def module_range(self, name: str) -> tuple[int, int]:
        """Address range of the mapped exe (DARKSOULS.exe is non-ASLR, base 0x400000)."""
        lo = hi = None
        with open(f"/proc/{self.pid}/maps") as f:
            for line in f:
                if name.lower() in line.lower():
                    a, b = (int(x, 16) for x in line.split()[0].split("-"))
                    lo = a if lo is None else min(lo, a)
                    hi = b if hi is None else max(hi, b)
        if lo is None:
            # Wine sometimes maps PE images anonymously; fall back to the fixed base.
            return 0x400000, 0x400000 + 0x1400000
        return lo, hi

    def aob_scan(self, pattern: str, start: int, end: int) -> int | None:
        """Find a byte pattern like "8B 15 ? ? ? ? F3 0F" in [start, end)."""
        regex = b"".join(
            b"." if tok == "?" else re.escape(bytes([int(tok, 16)]))
            for tok in pattern.split()
        )
        rx = re.compile(regex, re.DOTALL)
        chunk = 1 << 20
        overlap = len(pattern.split())
        addr = start
        while addr < end:
            n = min(chunk + overlap, end - addr)
            try:
                buf = self.read(addr, n)
            except OSError:
                addr += chunk
                continue
            m = rx.search(buf)
            if m:
                return addr + m.start()
            addr += chunk
        return None
