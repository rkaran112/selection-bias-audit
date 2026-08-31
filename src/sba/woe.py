"""Weight-of-evidence binning.

Everything here is weight-aware. That is not decoration: fuzzy augmentation
puts each rejected applicant into the data twice with fractional weights, and
inverse-propensity reweighting multiplies every accepted applicant by 1/p. If
the binning ignored weights, three of the four inference methods would be
silently applying their correction to the coefficients only, not to the bins,
and the comparison in Part 3 would be measuring the wrong thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import WOEConfig

EPS = 1e-12


@dataclass
class Bin:
    label: str
    woe: float
    n: float
    bad: float
    good: float
    bad_rate: float
    iv: float
    lo: float = np.nan          # numeric bins only
    hi: float = np.nan
    categories: tuple = ()      # categorical bins only
    is_missing: bool = False


@dataclass
class FeatureBinning:
    name: str
    kind: str                   # "numeric" | "categorical"
    bins: list = field(default_factory=list)
    edges: np.ndarray | None = None
    cat_map: dict = field(default_factory=dict)
    missing_woe: float = 0.0
    iv: float = 0.0

    def transform(self, s: pd.Series) -> np.ndarray:
        out = np.full(len(s), self.missing_woe, dtype="float64")
        if self.kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
            ok = ~np.isnan(v)
            if ok.any():
                # edges are interior cut points; digitize -> bin index
                idx = np.digitize(v[ok], self.edges, right=True)
                woes = np.array([b.woe for b in self.bins if not b.is_missing])
                idx = np.clip(idx, 0, len(woes) - 1)
                out[ok] = woes[idx]
        else:
            key = s.astype("string").fillna("__MISSING__")
            out = key.map(self.cat_map).fillna(self.missing_woe).to_numpy("float64")
        return out

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": self.name, "bin": b.label, "n": b.n,
            "bad": b.bad, "good": b.good, "bad_rate": b.bad_rate,
            "woe": b.woe, "iv": b.iv,
        } for b in self.bins])


def _woe_iv(bad: np.ndarray, good: np.ndarray, cfg: WOEConfig
            ) -> tuple[np.ndarray, np.ndarray]:
    """Laplace-smoothed WOE so an all-good or all-bad bin cannot go infinite."""
    b = bad + cfg.smoothing
    g = good + cfg.smoothing
    pb = b / b.sum()
    pg = g / g.sum()
    woe = np.log(pg / pb)
    iv = (pg - pb) * woe
    return woe, iv


def _merge_small(edges: list[float], stats: list[tuple], min_n: float) -> list[float]:
    """Merge adjacent bins until each carries at least min_n weight."""
    while len(stats) > 2:
        ns = np.array([s[0] for s in stats])
        j = int(np.argmin(ns))
        if ns[j] >= min_n:
            break
        k = j - 1 if (j == len(stats) - 1 or
                      (j > 0 and ns[j - 1] <= ns[j + 1])) else j + 1
        lo, hi = min(j, k), max(j, k)
        stats[lo] = tuple(a + b for a, b in zip(stats[lo], stats[hi]))
        del stats[hi]
        # bin k is bounded above by edges[k], so the cut separating lo from
        # lo+1 is edges[lo]; deleting edges[hi] would merge the wrong pair
        del edges[lo]
    return edges


def _enforce_monotonic(edges: list[float], stats: list[tuple]) -> list[float]:
    """Merge adjacent bins whose bad rate breaks the dominant direction."""
    if len(stats) < 3:
        return edges
    rates = np.array([s[1] / max(s[0], EPS) for s in stats])
    # direction from the rank correlation of bin index vs bad rate
    direction = np.sign(np.corrcoef(np.arange(len(rates)), rates)[0, 1])
    if not np.isfinite(direction) or direction == 0:
        return edges
    guard = 0
    while len(stats) > 2 and guard < 100:
        guard += 1
        rates = np.array([s[1] / max(s[0], EPS) for s in stats])
        d = np.diff(rates) * direction
        viol = np.where(d < 0)[0]
        if len(viol) == 0:
            break
        j = int(viol[np.argmin(d[viol])])      # worst violation
        stats[j] = tuple(a + b for a, b in zip(stats[j], stats[j + 1]))
        del stats[j + 1]
        del edges[j]
    return edges


def fit_numeric(x: pd.Series, y: np.ndarray, w: np.ndarray,
                cfg: WOEConfig, name: str) -> FeatureBinning:
    v = pd.to_numeric(x, errors="coerce").to_numpy("float64")
    miss = np.isnan(v)
    tot_w = w.sum()
    min_n = cfg.min_bin_frac * tot_w

    fb = FeatureBinning(name=name, kind="numeric")

    obs = ~miss
    if obs.sum() == 0:
        fb.edges = np.array([np.inf])
        fb.bins = [Bin("MISSING", 0.0, tot_w, 0, 0, 0, 0, is_missing=True)]
        return fb

    vo, yo, wo = v[obs], y[obs], w[obs]
    # fine quantile pre-bins, then merge down
    qs = np.linspace(0, 1, cfg.max_bins * 3 + 1)[1:-1]
    cuts = np.unique(np.quantile(vo, qs))
    edges = list(cuts) + [np.inf]

    def _stats(edges_):
        idx = np.digitize(vo, np.array(edges_), right=True)
        idx = np.clip(idx, 0, len(edges_) - 1)
        out = []
        for k in range(len(edges_)):
            m = idx == k
            out.append((wo[m].sum(), (wo[m] * yo[m]).sum()))
        return out

    edges = _merge_small(edges, _stats(edges), min_n)
    if cfg.monotonic:
        edges = _enforce_monotonic(edges, _stats(edges))
    # cap bin count
    while len(edges) > cfg.max_bins:
        st = _stats(edges)
        ns = np.array([s[0] for s in st])
        j = int(np.argmin(ns[:-1] + ns[1:]))
        del edges[j]

    st = _stats(edges)
    n_arr = np.array([s[0] for s in st])
    bad_arr = np.array([s[1] for s in st])
    good_arr = n_arr - bad_arr
    if miss.any():
        n_arr = np.append(n_arr, w[miss].sum())
        bad_arr = np.append(bad_arr, (w[miss] * y[miss]).sum())
        good_arr = np.append(good_arr, n_arr[-1] - bad_arr[-1])

    woe, iv = _woe_iv(bad_arr, good_arr, cfg)
    n_real = len(edges)
    lows = [-np.inf] + list(edges[:-1])
    for k in range(n_real):
        fb.bins.append(Bin(
            label=f"({_fmt(lows[k])}, {_fmt(edges[k])}]",
            woe=float(woe[k]), n=float(n_arr[k]), bad=float(bad_arr[k]),
            good=float(good_arr[k]),
            bad_rate=float(bad_arr[k] / max(n_arr[k], EPS)),
            iv=float(iv[k]), lo=lows[k], hi=edges[k]))
    if miss.any():
        fb.bins.append(Bin("MISSING", float(woe[-1]), float(n_arr[-1]),
                           float(bad_arr[-1]), float(good_arr[-1]),
                           float(bad_arr[-1] / max(n_arr[-1], EPS)),
                           float(iv[-1]), is_missing=True))
        fb.missing_woe = float(woe[-1])
    fb.edges = np.array(edges, dtype="float64")
    fb.iv = float(iv.sum())
    return fb


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:,.4g}"


def fit_categorical(x: pd.Series, y: np.ndarray, w: np.ndarray,
                    cfg: WOEConfig, name: str) -> FeatureBinning:
    key = x.astype("string").fillna("__MISSING__")
    df = pd.DataFrame({"k": key, "y": y, "w": w})
    agg = df.groupby("k", observed=True).apply(
        lambda g: pd.Series({"n": g["w"].sum(), "bad": (g["w"] * g["y"]).sum()}),
        include_groups=False)
    tot = agg["n"].sum()
    rare = agg.index[agg["n"] / tot < cfg.rare_cat_frac]
    mapping = {k: ("__OTHER__" if k in set(rare) else k) for k in agg.index}
    df["k2"] = df["k"].map(mapping)
    agg2 = df.groupby("k2", observed=True).apply(
        lambda g: pd.Series({"n": g["w"].sum(), "bad": (g["w"] * g["y"]).sum()}),
        include_groups=False).sort_index()

    bad_arr = agg2["bad"].to_numpy("float64")
    n_arr = agg2["n"].to_numpy("float64")
    good_arr = n_arr - bad_arr
    woe, iv = _woe_iv(bad_arr, good_arr, cfg)

    fb = FeatureBinning(name=name, kind="categorical")
    level_woe = dict(zip(agg2.index.tolist(), woe.tolist()))
    fb.cat_map = {k: level_woe[mapping[k]] for k in agg.index}
    fb.missing_woe = level_woe.get("__OTHER__", 0.0)
    for i, k in enumerate(agg2.index):
        fb.bins.append(Bin(label=str(k), woe=float(woe[i]), n=float(n_arr[i]),
                           bad=float(bad_arr[i]), good=float(good_arr[i]),
                           bad_rate=float(bad_arr[i] / max(n_arr[i], EPS)),
                           iv=float(iv[i]),
                           categories=tuple(k2 for k2, v in mapping.items()
                                            if v == k)))
    fb.iv = float(iv.sum())
    return fb


class WOEBinner:
    """Fit once on a training frame, then transform anything."""

    def __init__(self, numeric: list[str], categorical: list[str],
                 cfg: WOEConfig):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.cfg = cfg
        self.features_: dict[str, FeatureBinning] = {}

    def fit(self, X: pd.DataFrame, y, w=None) -> "WOEBinner":
        y = np.asarray(y, dtype="float64")
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
        for c in self.numeric:
            self.features_[c] = fit_numeric(X[c], y, w, self.cfg, c)
        for c in self.categorical:
            self.features_[c] = fit_categorical(X[c], y, w, self.cfg, c)
        return self

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [self.features_[c].transform(X[c]) for c in self.columns])

    def fit_transform(self, X, y, w=None) -> np.ndarray:
        return self.fit(X, y, w).transform(X)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "iv": self.features_[c].iv,
                 "n_bins": len(self.features_[c].bins)} for c in self.columns]
        return pd.DataFrame(rows).sort_values("iv", ascending=False)

    def bin_table(self) -> pd.DataFrame:
        return pd.concat([self.features_[c].table() for c in self.columns],
                         ignore_index=True)
