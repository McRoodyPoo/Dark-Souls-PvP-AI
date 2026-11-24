"""Tiny numpy MLPs that replace metal-crow's FANN networks.

Why retrain: the repo's Attack_dark_souls_ai.net and Defense_dark_souls_ai.net are
byte-identical, take 5 inputs, and were never paired with their input scaling, while
the code feeds them 8 and 177 inputs. So the shipped "brains" did essentially nothing
meaningful. The raw recordings (Neural Nets/*.train, *.test) are still there, so we
train on those, store the per-column scaling with the weights, and apply the same
scaling at runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def read_fann_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """FANN text format: 'N n_in n_out' then alternating input / output lines.

    attack_training_data.train has 6 lines where two input rows were written back to
    back with no label (354 fields), which shifts every later sample if the file is
    read as one flat stream. Parse line by line and keep only the labelled row.
    """
    lines = [l.split() for l in path.read_text().replace("\r", "").split("\n")[1:] if l.strip()]
    header = path.read_text().split("\n", 1)[0].split()
    n_in, n_out = int(header[1]), int(header[2])
    xs, ys, pending = [], [], None
    for toks in lines:
        if len(toks) == n_out and pending is not None:
            xs.append(pending)
            ys.append([float(t) for t in toks])
            pending = None
        elif len(toks) % n_in == 0 and len(toks) >= n_in:
            pending = [float(t) for t in toks[-n_in:]]   # unlabelled earlier rows dropped
    return np.array(xs), np.array(ys)


class MLP:
    def __init__(self, n_in: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)
        self.lo = np.zeros(n_in)
        self.hi = np.ones(n_in)

    # --- scaling (same idea as fann_scale_train_data, but saved) -----------
    def fit_scaling(self, x: np.ndarray) -> None:
        self.lo, self.hi = x.min(0), x.max(0)

    def scale(self, x: np.ndarray) -> np.ndarray:
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return np.clip(2 * (x - self.lo) / span - 1, -1.5, 1.5)

    # --- inference ---------------------------------------------------------
    def __call__(self, x) -> float:
        x = self.scale(np.asarray(x, dtype=np.float64)[None, :])
        return float(np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)[0, 0])

    # --- training (full-batch Adam on MSE, targets in [-1, 1]) --------------
    def train(self, x, y, epochs=3000, lr=0.01, l2=1e-3) -> None:
        self.fit_scaling(x)
        xs = self.scale(x)
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for t in range(1, epochs + 1):
            h = np.tanh(xs @ self.w1 + self.b1)
            out = np.tanh(h @ self.w2 + self.b2)
            d_out = 2 * (out - y) / len(x) * (1 - out ** 2)
            g_w2 = h.T @ d_out + l2 * self.w2
            g_b2 = d_out.sum(0)
            d_h = d_out @ self.w2.T * (1 - h ** 2)
            g_w1 = xs.T @ d_h + l2 * self.w1
            g_b1 = d_h.sum(0)
            for i, (p, g) in enumerate(zip(params, [g_w1, g_b1, g_w2, g_b2])):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.tanh(self.scale(x) @ self.w1 + self.b1) @ self.w2 + self.b2)[:, 0]

    # --- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, lo=self.lo, hi=self.hi)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        d = np.load(path)
        net = cls(d["w1"].shape[0], d["w1"].shape[1])
        for k in ("w1", "b1", "w2", "b2", "lo", "hi"):
            setattr(net, k, d[k])
        return net
