"""Bring the Dark Souls window to the front (KDE Plasma / KWin), and report what has focus.

  python tools/focus_game.py
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

NAME = "ptde_focus_game"
SCRIPT = """
var found = false;
var list = workspace.windowList();
for (var i = 0; i < list.length; i++) {
    var w = list[i];
    if (w.caption.toUpperCase() == "DARK SOULS") { workspace.activeWindow = w; found = true; }
}
print("PTDEACT " + (found ? "activated" : "not found") + " | active: " +
      (workspace.activeWindow ? workspace.activeWindow.caption : ""));
"""


def q(*a: str) -> str:
    return subprocess.run(["qdbus6", "org.kde.KWin", "/Scripting", *a],
                          capture_output=True, text=True, timeout=5).stdout.strip()


def main() -> None:
    f = Path(tempfile.mkstemp(suffix=".js", prefix="ptde_act_")[1])
    f.write_text(SCRIPT)
    j = subprocess.Popen(["journalctl", "--user", "-f", "-n", "0", "-o", "cat"],
                         stdout=subprocess.PIPE, text=True)
    try:
        time.sleep(0.3)
        q("org.kde.kwin.Scripting.unloadScript", NAME)
        q("org.kde.kwin.Scripting.loadScript", str(f), NAME)
        q("org.kde.kwin.Scripting.start")
        t0 = time.time()
        while time.time() - t0 < 3:
            line = j.stdout.readline()
            if "PTDEACT" in line:
                print(line[line.find("PTDEACT") + 8:].strip())
                break
    finally:
        q("org.kde.kwin.Scripting.unloadScript", NAME)
        j.terminate()
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
