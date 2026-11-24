"""Is the Dark Souls window focused?

Dark Souls ignores controller input while its window is in the background, so the bot's
presses do nothing then (verified 2026-11-22). On KDE Plasma (Wayland) a small KWin
script prints the caption of each newly focused window to the journal, and a
`journalctl -f` reader picks it up. Elsewhere `focused` stays None (unknown).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

TAG = "PTDEFOCUS "
SCRIPT_NAME = "ptde_focus_watch"
GAME_CAPTION = "DARK SOULS"

_SCRIPT = f"""
function report(w) {{ print("{TAG}" + (w ? w.caption : "")); }}
workspace.windowActivated.connect(report);
report(workspace.activeWindow);
"""


def _qdbus(*args: str) -> str:
    r = subprocess.run(["qdbus6", "org.kde.KWin", "/Scripting", *args],
                       capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


class FocusWatch:
    def __init__(self):
        self.caption: str | None = None
        self._proc = None
        self._file = None
        if not (shutil.which("qdbus6") and shutil.which("journalctl")):
            return
        try:
            self._proc = subprocess.Popen(
                ["journalctl", "--user", "-f", "-n", "0", "-o", "cat"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
            time.sleep(0.3)                          # let journalctl attach first
            self._file = Path(tempfile.mkstemp(suffix=".js", prefix="ptde_focus_")[1])
            self._file.write_text(_SCRIPT)
            _qdbus("org.kde.kwin.Scripting.unloadScript", SCRIPT_NAME)
            _qdbus("org.kde.kwin.Scripting.loadScript", str(self._file), SCRIPT_NAME)
            _qdbus("org.kde.kwin.Scripting.start")
        except (OSError, subprocess.SubprocessError):
            self.close()

    def _read(self):
        for line in self._proc.stdout:
            i = line.find(TAG)
            if i >= 0:
                self.caption = line[i + len(TAG):].rstrip("\n")

    @property
    def focused(self) -> bool | None:
        """True/False once KWin has reported a window; None if unknown."""
        if self.caption is None:
            return None
        return GAME_CAPTION in self.caption.upper()

    def close(self):
        try:
            if shutil.which("qdbus6"):
                _qdbus("org.kde.kwin.Scripting.unloadScript", SCRIPT_NAME)
        except (OSError, subprocess.SubprocessError):
            pass
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        if self._file is not None:
            self._file.unlink(missing_ok=True)
            self._file = None
