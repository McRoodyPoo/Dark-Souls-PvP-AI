"""Retrain the backstab-threat net in the background while the bot waits for a summon.

User 2026-11-24: what's learned on each player should be absorbed by the general model. The
threat net learns from every recorded fight, so new fights get folded in here:
  * only while idle at home (bot driving, nobody to fight), at most every RETRAIN_GAP_H hours,
    and only if there are new fight recordings since the last training;
  * runs `tools/train_threat.py --auto` at the lowest CPU priority with few threads; that
    script skips if there are too few new backstabs, and only installs a net that scores well
    on the newest fights it wasn't trained on;
  * the job is frozen (SIGSTOP) the moment a fight starts or the user takes over, and resumed
    when idle again, so it never costs frames in a fight;
  * a new models/threat.npz is loaded while idle.
Survives a hot reload: the pid is kept in records/retrain.pid (exec keeps it our child).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "threat.npz"
META = ROOT / "models" / "threat.json"
LOG = ROOT / "records" / "retrain.log"
PIDFILE = ROOT / "records" / "retrain.pid"
REC_DIR = ROOT / "recordings"
RETRAIN_GAP_H = 6.0
CHECK_EVERY_MS = 60_000
MIN_RECORDING_BYTES = 200_000     # a recording with real fighting in it
THREADS = "4"


def _alive(pid: int) -> bool:
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)       # reaps it if it's our finished child
        return done == 0
    except ChildProcessError:
        pass                                          # not our child: look at /proc
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] not in ("Z", "X")
    except OSError:
        return False


class Retrainer:
    def __init__(self, root: Path = ROOT, now_s=time.time):
        self.now_s = now_s
        self.pid: int | None = None
        self.paused = False
        self.log_pos = 0
        self.last_check = -1e18
        self.model_mtime = MODEL.stat().st_mtime if MODEL.exists() else 0.0
        try:                                          # adopt a job from before a hot reload
            pid = int(PIDFILE.read_text())
            if _alive(pid):
                self.pid = pid
        except (OSError, ValueError):
            pass

    @property
    def running(self) -> bool:
        return self.pid is not None

    def due(self) -> bool:
        last = META.stat().st_mtime if META.exists() else 0.0
        if self.now_s() - last < RETRAIN_GAP_H * 3600:
            return False
        return any(f.stat().st_mtime > last and f.stat().st_size >= MIN_RECORDING_BYTES
                   for f in REC_DIR.glob("*.jsonl.gz"))

    def _start(self) -> None:
        env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS,
                   MKL_NUM_THREADS=THREADS)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self.log_pos = LOG.stat().st_size if LOG.exists() else 0
        with LOG.open("a") as log:
            log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} retrain started\n")
            log.flush()
            proc = subprocess.Popen(["nice", "-n", "19", sys.executable,
                                     str(ROOT / "tools" / "train_threat.py"), "--auto"],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.pid, self.paused = proc.pid, False
        PIDFILE.write_text(str(proc.pid))

    def _signal(self, sig) -> None:
        try:
            os.kill(self.pid, sig)
        except OSError:
            pass

    def tick(self, now_ms: float, idle: bool) -> list[str]:
        """Call every loop. idle = at home, bot driving, nobody to fight."""
        out: list[str] = []
        if self.pid is not None:
            if not _alive(self.pid):
                self.pid = None
                PIDFILE.unlink(missing_ok=True)
                out.append("retrain finished: " + self._last_log_line())
            elif not idle and not self.paused:
                self._signal(signal.SIGSTOP)
                self.paused = True
                out.append("retrain paused (fighting / you're playing)")
            elif idle and self.paused:
                self._signal(signal.SIGCONT)
                self.paused = False
                out.append("retrain resumed")
            return out
        if idle and now_ms - self.last_check >= CHECK_EVERY_MS:
            self.last_check = now_ms
            if self.due():
                self._start()
                out.append("retraining the backstab net on the newest fights in the background "
                           "(lowest priority; paused during fights; log: records/retrain.log)")
        return out

    def model_updated(self) -> bool:
        """A new threat net was installed since we last loaded one."""
        m = MODEL.stat().st_mtime if MODEL.exists() else 0.0
        if m != self.model_mtime:
            self.model_mtime = m
            return True
        return False

    def release(self) -> None:
        """Bot stopping / reloading: never leave the job frozen."""
        if self.pid is not None and self.paused:
            self._signal(signal.SIGCONT)
            self.paused = False

    def _last_log_line(self) -> str:
        try:
            with LOG.open() as f:
                f.seek(self.log_pos)
                lines = f.read().splitlines()
        except OSError:
            return "(no log)"
        result = [l for l in lines if l.startswith("retrain:")]
        return result[-1][len("retrain: "):] if result else "FAILED (see records/retrain.log)"
