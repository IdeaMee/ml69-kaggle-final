"""Shared helpers: paths, metrics, threshold search, submission validation."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # repo root (data/raw/ and outputs/ live here, not tracked by git)
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs"


def f1_binary(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def f1_curve(y, prob):
    """F1 for every distinct threshold (predict 1 if prob >= thr). Returns thr, f1 arrays (descending thr)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-prob, kind="stable")
    ys = y[order]
    ps = prob[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + y.sum())  # 2TP / (TP+FP + TP+FN) = 2TP/(k + P)
    last = np.r_[ps[1:] != ps[:-1], True]  # only cut between distinct scores
    return ps[last], f1[last], k[last]


def best_threshold(y, prob, smooth=0.0):
    """Threshold maximizing F1. smooth>0: choose the centre of the thresholds whose F1 >= max-smooth."""
    thr, f1, _ = f1_curve(y, prob)
    best = f1.max()
    if smooth <= 0:
        i = int(np.argmax(f1))
        return float(thr[i]), float(f1[i])
    ok = np.where(f1 >= best - smooth)[0]
    i = int(ok[len(ok) // 2])
    return float(thr[i]), float(f1[i])


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, kind: str):
    """Raise AssertionError if the file does not match the competition format."""
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"]) == set(sample["id"]), "id set differs from sample_submission"
    assert not sub.isna().any().any(), "NaN present"
    col = sample.columns[1]
    if kind == "c1":
        assert set(sub[col].unique()) <= {0, 1}, "is_fraud must be 0/1"
        assert pd.api.types.is_integer_dtype(sub[col]), "is_fraud must be integer"
    elif kind == "r3":
        v = sub[col].to_numpy(float)
        assert np.isfinite(v).all() and (v >= 0).all(), "energy must be finite and >= 0"
    return True
