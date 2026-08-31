"""WOE + logistic application scorecard, scaled to points.

This is deliberately a plain, conventional scorecard rather than a gradient
boosting machine. The question this repo asks is about the sample a model is
fitted on, not about the model class; using the same textbook scorecard
everywhere keeps the four inference methods comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .config import ScorecardConfig, WOEConfig
from .woe import WOEBinner


@dataclass
class Scorecard:
    binner: WOEBinner
    model: LogisticRegression
    cfg: ScorecardConfig
    features: list

    # ---------------------------------------------------------------- scaling
    @property
    def factor(self) -> float:
        return self.cfg.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.cfg.base_points - self.factor * np.log(self.cfg.base_odds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(bad)."""
        return self.model.predict_proba(self.binner.transform(X))[:, 1]

    def log_odds_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self.binner.transform(X))

    def points(self, X: pd.DataFrame) -> np.ndarray:
        """Higher points = better applicant, the usual industry orientation."""
        return self.offset + self.factor * (-self.log_odds_bad(X))

    # ---------------------------------------------------------------- reports
    def points_table(self) -> pd.DataFrame:
        """Per-bin point allocation: the artefact a credit officer signs off."""
        n = len(self.features)
        b0 = float(self.model.intercept_[0])
        coefs = dict(zip(self.features, self.model.coef_[0]))
        rows = []
        for f in self.features:
            fb = self.binner.features_[f]
            for b in fb.bins:
                pts = (self.offset / n) + self.factor * (
                    -b0 / n - coefs[f] * b.woe)
                rows.append({"feature": f, "bin": b.label, "n": b.n,
                             "bad_rate": b.bad_rate, "woe": b.woe,
                             "coef": coefs[f], "points": pts})
        return pd.DataFrame(rows)

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "coef": self.model.coef_[0],
            "iv": [self.binner.features_[f].iv for f in self.features],
        }).sort_values("iv", ascending=False)

    def evaluate(self, X: pd.DataFrame, y, w=None) -> dict:
        return M.all_metrics(y, self.predict_proba(X), w)


def fit_scorecard(X: pd.DataFrame, y, w=None, *,
                  numeric: list, categorical: list,
                  woe_cfg: WOEConfig, sc_cfg: ScorecardConfig) -> Scorecard:
    y = np.asarray(y, dtype="float64")
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
    keep = np.isfinite(y) & np.isfinite(w) & (w > 0)
    X, y, w = X.loc[keep].reset_index(drop=True), y[keep], w[keep]

    binner = WOEBinner(numeric, categorical, woe_cfg).fit(X, y, w)
    Z = binner.transform(X)
    model = LogisticRegression(C=sc_cfg.C, max_iter=2000, solver="lbfgs")
    model.fit(Z, y, sample_weight=w)
    return Scorecard(binner=binner, model=model, cfg=sc_cfg,
                     features=binner.columns)


def train_test_split_seeded(df: pd.DataFrame, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < frac
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)
