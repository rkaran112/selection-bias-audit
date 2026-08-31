"""Common contract for the four reject-inference methods.

Every method must turn (accepted-with-outcomes, rejected-without-outcomes)
into two things:

  1. a scoring function applicable to ANY applicant, accepted or not, and
  2. an estimate of the through-the-door bad rate.

Part 3 then checks both against a truth the method was not shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import ScorecardConfig, WOEConfig
from ..scorecard import Scorecard, fit_scorecard


@dataclass
class InferenceContext:
    accepted: pd.DataFrame          # 'bad' observed
    rejected: pd.DataFrame          # 'bad' is NaN by construction
    baseline: Scorecard             # accepts-only model, the biased benchmark
    numeric: list
    categorical: list
    woe_cfg: WOEConfig
    sc_cfg: ScorecardConfig
    seed: int = 0
    w_accepted: float = 1.0         # design weights from the sampling frame
    w_rejected: float = 1.0
    # Multiplier on the accepted bad rate used by the two methods that need an
    # exogenous assumption about how much worse rejects are. There is no way to
    # estimate this from the data; that is the point of Part 3.
    reject_bad_scale: float = 2.0
    # Variables entering the SELECTION equation only. Heckman is identified off
    # the probit functional form alone when this is empty, which is weak; the
    # pipeline reports rho both ways so the reader can see how much of the
    # answer is assumption. Everything else ignores this field.
    exclusion: list = field(default_factory=list)

    @property
    def features(self) -> list:
        return self.numeric + self.categorical


@dataclass
class InferenceResult:
    name: str
    score_fn: Callable[[pd.DataFrame], np.ndarray]   # returns PD, higher=worse
    est_reject_bad_rate: float
    est_ttd_bad_rate: float
    diagnostics: dict = field(default_factory=dict)
    scorecard: Scorecard | None = None
    # The method's OWN view of the through-the-door population: feature
    # columns plus 'bad' and 'w'. This is what a practitioner would compute
    # their corrected Gini from, so Part 3 measures the bias in exactly that
    # number rather than in some quantity the method never claimed.
    train_frame: pd.DataFrame | None = None

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.score_fn(X)


def refit(frame: pd.DataFrame, ctx: InferenceContext) -> Scorecard:
    """Refit the same textbook scorecard on an augmented, weighted frame."""
    return fit_scorecard(
        frame[ctx.features], frame["bad"].to_numpy(), frame["w"].to_numpy(),
        numeric=ctx.numeric, categorical=ctx.categorical,
        woe_cfg=ctx.woe_cfg, sc_cfg=ctx.sc_cfg)


REGISTRY: dict[str, Callable[[InferenceContext], InferenceResult]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        fn.method_name = name
        return fn
    return deco


def run_all(ctx: InferenceContext, methods: list | None = None
            ) -> dict[str, InferenceResult]:
    names = methods or list(REGISTRY)
    out = {}
    for n in names:
        out[n] = REGISTRY[n](ctx)
    return out
