"""Discrimination metrics. All weighted, because most of this repo is."""
from __future__ import annotations

import numpy as np


def _prep(y, s, w):
    y = np.asarray(y, dtype="float64")
    s = np.asarray(s, dtype="float64")
    w = np.ones_like(y) if w is None else np.asarray(w, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(s) & np.isfinite(w) & (w > 0)
    return y[ok], s[ok], w[ok]


def roc_curve(y, score, w=None):
    """Weighted ROC. `score` is oriented so higher = more likely bad."""
    y, s, w = _prep(y, score, w)
    order = np.argsort(-s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    P, N = tp[-1], fp[-1]
    if P <= 0 or N <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # collapse ties so the curve is a proper step function
    keep = np.append(np.diff(s) != 0, True)
    tpr = np.append(0.0, tp[keep] / P)
    fpr = np.append(0.0, fp[keep] / N)
    return fpr, tpr


def auc(y, score, w=None) -> float:
    fpr, tpr = roc_curve(y, score, w)
    return float(np.trapezoid(tpr, fpr))


def gini(y, score, w=None) -> float:
    return 2.0 * auc(y, score, w) - 1.0


def ks(y, score, w=None) -> float:
    fpr, tpr = roc_curve(y, score, w)
    return float(np.max(np.abs(tpr - fpr)))


def weighted_mean(x, w=None) -> float:
    x = np.asarray(x, dtype="float64")
    w = np.ones_like(x) if w is None else np.asarray(w, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def effective_sample_size(w) -> float:
    """Kish ESS. A weighting scheme that collapses this has thrown away data."""
    w = np.asarray(w, dtype="float64")
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def all_metrics(y, score, w=None) -> dict:
    """`score` here is a PD-like quantity: higher = worse."""
    return {"auc": auc(y, score, w), "gini": gini(y, score, w),
            "ks": ks(y, score, w), "bad_rate": weighted_mean(y, w),
            "n": float(len(np.asarray(y)))}
