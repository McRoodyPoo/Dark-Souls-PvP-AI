"""Retrain metal-crow's two networks from his recorded data.

  python tools/train_nets.py     # writes models/defense.npz and models/attack.npz

defense (backstab detector): 8 inputs, label +1 = backstab attempt coming, -1 = not.
attack: 177 inputs, label +1 = attacking now worked out, -1 = it didn't, 0 = neutral.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai.nets import MODEL_DIR, MLP, read_fann_data  # noqa: E402

# metal-crow's original repo: vendored here, or (in the GitHub fork) the folder above this one
UPSTREAM = next((d for d in (ROOT / "vendor" / "Dark-Souls-PvP-AI", ROOT.parent)
                 if (d / "Neural Nets").exists()), ROOT / "vendor" / "Dark-Souls-PvP-AI")
DATA = UPSTREAM / "Neural Nets"


def report(name, net, x, y):
    pred = net.predict(x)
    nz = y[:, 0] != 0
    acc = np.mean(np.sign(pred[nz]) == np.sign(y[nz, 0])) if nz.any() else float("nan")
    base = max(np.mean(y[nz, 0] > 0), np.mean(y[nz, 0] < 0)) if nz.any() else float("nan")
    print(f"  {name:5} n={len(x):3}  sign accuracy {acc:.1%}  (always-guess-majority {base:.1%})")


def train(stem: str, out: str, hidden: int, l2: float):
    xtr, ytr = read_fann_data(DATA / f"{stem}.train")
    xte, yte = read_fann_data(DATA / f"{stem}.test")
    net = MLP(xtr.shape[1], hidden=hidden)
    net.train(xtr, ytr, l2=l2)
    print(f"{out}: {xtr.shape[1]} inputs")
    report("train", net, xtr, ytr)
    report("test", net, xte, yte)
    net.save(MODEL_DIR / f"{out}.npz")


if __name__ == "__main__":
    train("backstab_training_data", "defense", hidden=8, l2=1e-3)
    train("attack_training_data", "attack", hidden=16, l2=1e-2)
