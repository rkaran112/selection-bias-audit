"""(b) Fuzzy augmentation.

Every rejected applicant enters the training data twice: once as a good with
weight (1 - p), once as a bad with weight p, where p is the biased model's
predicted PD scaled up by the same exogenous factor parcelling uses.

Fuzzy is parcelling with the sampling noise integrated out. It is therefore
perfectly stable across seeds - which looks like an advantage until Part 3
shows that a stable estimator of the wrong quantity is still wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import InferenceContext, InferenceResult, refit, register


@register("fuzzy")
def fuzzy(ctx: InferenceContext) -> InferenceResult:
    acc, rej = ctx.accepted, ctx.rejected
    p = ctx.baseline.predict_proba(rej[ctx.features])
    p_scaled = np.clip(ctx.reject_bad_scale * p, 0.0, 1.0)

    a = acc[ctx.features].copy()
    a["bad"] = acc["bad"].to_numpy()
    a["w"] = ctx.w_accepted

    r_bad = rej[ctx.features].copy()
    r_bad["bad"] = 1.0
    r_bad["w"] = ctx.w_rejected * p_scaled

    r_good = rej[ctx.features].copy()
    r_good["bad"] = 0.0
    r_good["w"] = ctx.w_rejected * (1.0 - p_scaled)

    frame = pd.concat([a, r_bad, r_good], ignore_index=True)
    frame = frame[frame["w"] > 1e-9].reset_index(drop=True)

    sc = refit(frame, ctx)
    overall = float(np.mean(acc["bad"].to_numpy()))
    est_rej = float(p_scaled.mean())
    wa, wr = ctx.w_accepted * len(acc), ctx.w_rejected * len(rej)
    est_ttd = (wa * overall + wr * est_rej) / (wa + wr)

    return InferenceResult(
        name="fuzzy",
        score_fn=lambda X: sc.predict_proba(X),
        est_reject_bad_rate=est_rej, est_ttd_bad_rate=est_ttd,
        diagnostics={"scale_factor": ctx.reject_bad_scale,
                     "mean_raw_pd_rejects": float(p.mean()),
                     "accepted_bad_rate": overall,
                     "frac_pd_clipped_at_1": float((ctx.reject_bad_scale * p
                                                    > 1.0).mean())},
        scorecard=sc, train_frame=frame)
