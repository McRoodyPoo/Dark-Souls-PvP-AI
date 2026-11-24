"""Train the backstab-threat net on the bot's own recorded fights.

  python tools/train_threat.py            # evaluate (train on older fights, test on newer), then
                                          # train on everything -> models/threat.npz
  python tools/train_threat.py --eval     # evaluate only, write nothing
  python tools/train_threat.py --auto     # what the bot runs while idle: retrain only if there
                                          # are enough new backstabs, and install the new net only
                                          # if it scores well on the newest fights

A "backstab on the bot" = the opponent's backstab animation (225) starts while the bot isn't
in one itself, and the bot loses >100 HP within 2.5 s. Each tick 150-900 ms before one is a
positive (early enough to react); ticks with none coming in the next 2 s (and not just after
one) are negatives. Only ticks with both alive and within 6 m count.
"""
from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptde_ai import geometry as G                   # noqa: E402
from ptde_ai.history import History                 # noqa: E402
from ptde_ai.nets import MODEL_DIR, MLP             # noqa: E402
from ptde_ai.recorder import REC_DIR, load          # noqa: E402
from ptde_ai.threat import ThreatFeatures           # noqa: E402

BS_ANIM = 225
POS_FROM, POS_TO = 900, 150          # ms before the backstab
NEG_CLEAR_BEFORE, NEG_CLEAR_AFTER = 2000, 2500
MAX_DIST = 6.0
TEST_FROM = "20261123-231052"        # files from here on are the test set (newest ~100 backstabs)


class _C:                            # just enough of a Character for the legacy features
    def __init__(self, r):
        self.loc_x, self.loc_y, self.rotation_raw = r["loc_x"], r["loc_y"], r["rotation"]
        self.rotation_deg = math.degrees(r["rotation"] + math.pi)


def backstabs(rows) -> list[float]:
    out = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["e"]["animation_type"] == BS_ANIM and a["e"]["animation_type"] != BS_ANIM \
                and b["p"]["animation_type"] != BS_ANIM:
            later = [r["p"]["hp"] for r in rows[i:i + 400] if r["t"] - b["t"] <= 2500]
            if b["p"]["hp"] - min(later) > 100:
                out.append(b["t"])
    return out


def build(path: str):
    rows = [r for r in load(path) if "mode" in r and r.get("e") and r.get("p")]
    events = backstabs(rows)
    tf, hist = ThreatFeatures(), History()
    X, L, Y, EV = [], [], [], []     # new features, legacy features, label, event index (or -1)
    j = 0
    for r in rows:
        p, e, t = r["p"], r["e"], r["t"]
        tf.push(t, p["loc_x"], p["loc_y"], p["rotation"], e["loc_x"], e["loc_y"], e["rotation"])
        pc, ec = _C(p), _C(e)
        hist.push_distance(G.legacy_distance(pc, ec), t)
        while j < len(events) and events[j] < t - NEG_CLEAR_AFTER:
            j += 1
        if p["hp"] <= 0 or e["hp"] <= 0 or math.hypot(p["loc_x"] - e["loc_x"], p["loc_y"] - e["loc_y"]) > MAX_DIST:
            continue
        x = tf.features(p["animation_type"] != 0, e["animation_type"] != 0)
        if x is None:
            continue
        label, ev = None, -1
        soon = events[j:j + 5]
        for k, T in enumerate(soon, start=j):
            if T - POS_FROM <= t <= T - POS_TO:
                label, ev = 1, k
                break
        if label is None:
            if any(T - NEG_CLEAR_BEFORE <= t <= T + NEG_CLEAR_AFTER for T in soon):
                continue              # too close to a backstab to call it safe
            label = 0
        h = hist.distances
        X.append(x)
        L.append([G.legacy_distance(pc, ec), h[0], h[1], h[2], h[3],
                  G.legacy_angle_delta_from_front(pc, ec), e.get("velocity", 0.0),
                  G.legacy_rotation_difference(pc, ec)])
        Y.append(label)
        EV.append(ev)
    return np.array(X), np.array(L), np.array(Y), np.array(EV), len(events)


def load_all():
    data = []
    for f in sorted(glob.glob(str(REC_DIR / "*.jsonl.gz"))):
        X, L, Y, EV, n = build(f)
        if len(X):
            data.append((Path(f).name, X, L, Y, EV, n))
    return data


def auc(score, y) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def evaluate(name, score, y, ev_ids, thr, hz=30.0):
    caught = {k for k, s in zip(ev_ids, score) if k >= 0 and s > thr}
    events = {k for k in ev_ids if k >= 0}
    false = np.mean(score[y == 0] > thr)
    print(f"  {name:28} AUC {auc(score, y):.3f} | backstabs warned {len(caught)}/{len(events)} "
          f"({len(caught) / max(1, len(events)):.0%}) | false alarms on {false:.1%} of safe ticks "
          f"(~{false * hz * 60:.0f}/min)")


def stack(parts, key):
    Xs, Ls, Ys, EVs, off = [], [], [], [], 0
    for name, X, L, Y, EV, n in parts:
        Xs.append(X); Ls.append(L); Ys.append(Y)
        EVs.append(np.where(EV >= 0, EV + off, -1)); off += n
    return np.vstack(Xs), np.vstack(Ls), np.concatenate(Ys), np.concatenate(EVs)


def fit(X, Y, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
    neg = rng.choice(neg, min(len(neg), 6 * len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    net = MLP(X.shape[1], hidden=16, seed=seed)
    net.train(X[idx], np.where(Y[idx] == 1, 1.0, -1.0)[:, None], epochs=2500, lr=0.01, l2=1e-3)
    return net


MIN_NEW_BACKSTABS = 10       # --auto: not worth retraining for fewer
AUC_FLOOR = 0.80             # --auto: never install a net that scores below this...
AUC_SLACK = 0.03             # ...or more than this below the last one


def newest_split(data, frac=0.25):
    """Test set = the newest recordings holding at least `frac` of all backstabs."""
    total = sum(d[5] for d in data)
    test, n = [], 0
    for d in reversed(data):
        if n >= frac * total:
            break
        test.append(d); n += d[5]
    names = {d[0] for d in test}
    return [d for d in data if d[0] not in names], sorted(test)


def install(data, X, Y, held_out_auc: float, n_backstabs: int) -> None:
    final = fit(X, Y)
    tmp = MODEL_DIR / "threat.tmp.npz"
    final.save(tmp)
    tmp.replace(MODEL_DIR / "threat.npz")          # atomic: the bot never loads half a file
    (MODEL_DIR / "threat.json").write_text(json.dumps(
        {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "trained_on_ticks": int(len(Y)),
         "positives": int(Y.sum()), "backstabs": int(n_backstabs),
         "held_out_auc": round(held_out_auc, 4), "files": [d[0] for d in data]}, indent=1))


def auto() -> int:
    meta_path = MODEL_DIR / "threat.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    known = set(meta.get("files", []))
    data = load_all()
    new = sum(d[5] for d in data if d[0] not in known)
    if new < MIN_NEW_BACKSTABS and "--force" not in sys.argv:
        print(f"retrain: only {new} new backstabs on the bot since the last training: skipped")
        return 0
    train, test = newest_split(data)
    Xtr, _, Ytr, _ = stack(train, 0)
    Xte, _, Yte, EVte = stack(test, 0)
    score = auc(fit(Xtr, Ytr).predict(Xte), Yte)
    last = meta.get("held_out_auc", 0.0)
    if score < AUC_FLOOR or score < last - AUC_SLACK:
        print(f"retrain: new net scored {score:.3f} on the newest fights (last {last:.3f}): "
              f"kept the old one")
        return 2
    X = np.vstack([Xtr, Xte]); Y = np.concatenate([Ytr, Yte])
    install(data, X, Y, score, sum(d[5] for d in data))
    print(f"retrain: {new} new backstabs absorbed; held-out score {score:.3f} (last {last:.3f}); "
          f"installed models/threat.npz")
    return 0


def main():
    if "--auto" in sys.argv:
        sys.exit(auto())
    data = load_all()
    train = [d for d in data if d[0] < TEST_FROM]
    test = [d for d in data if d[0] >= TEST_FROM]
    Xtr, Ltr, Ytr, EVtr = stack(train, 0)
    Xte, Lte, Yte, EVte = stack(test, 0)
    print(f"train: {len(Ytr)} ticks, {int(Ytr.sum())} positive, {len(set(EVtr[EVtr >= 0]))} backstabs")
    print(f"test:  {len(Yte)} ticks, {int(Yte.sum())} positive, {len(set(EVte[EVte >= 0]))} backstabs")
    net = fit(Xtr, Ytr)
    s = net.predict(Xte)
    old = MLP.load(MODEL_DIR / "defense.npz")
    so = old.predict(Lte)
    print("\non the newer fights (not trained on):")
    evaluate("old net (metal-crow, >0.5)", so, Yte, EVte, 0.5)
    for thr in (0.0, 0.3, 0.5, 0.7):
        evaluate(f"new net (>{thr})", s, Yte, EVte, thr)
    if "--eval" in sys.argv:
        return
    X = np.vstack([Xtr, Xte]); Y = np.concatenate([Ytr, Yte])
    install(data, X, Y, auc(s, Yte), sum(d[5] for d in data))
    print(f"\nwrote models/threat.npz (trained on all {len(Y)} ticks)")


if __name__ == "__main__":
    main()
