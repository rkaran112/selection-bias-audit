"""(a) Parcelling.

Score the rejects on the biased model, cut them into score bands, and within
each band assign bad labels at the accepted bad rate multiplied by a fixed
scaling factor. Labels are assigned by a seeded draw, not fractionally, which
is what makes parcelling distinct from fuzzy augmentation - and what makes it
the only one of the four with genuine seed-to-seed sampling noise.

The scaling factor is an assumption, not an estimate. Nothing in the data
identifies it. Part 3 exists largely to show what that assumption costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import InferenceContext, InferenceResult, refit, register

N_BANDS = 10


@register("parcelling")
def parcelling(ctx: InferenceContext) -> InferenceResult:
    rng = np.random.default_rng(ctx.seed + 101)
    acc, rej = ctx.accepted, ctx.rejected

    pd_acc = ctx.baseline.predict_proba(acc[ctx.features])
    pd_rej = ctx.baseline.predict_proba(rej[ctx.features])

    # Bands are cut on the accepted distribution so band k means the same
    # thing on both sides of the decision.
    edges = np.unique(np.quantile(pd_acc, np.linspace(0, 1, N_BANDS + 1)[1:-1]))
    b_acc = np.digitize(pd_acc, edges, right=True)
    b_rej = np.digitize(pd_rej, edges, right=True)

    overall = float(np.mean(acc["bad"].to_numpy()))
    band_rate = {}
    for k in range(len(edges) + 1):
        m = b_acc == k
        band_rate[k] = float(acc["bad"].to_numpy()[m].mean()) if m.sum() >= 50 \
            else overall

    target = np.array([min(1.0, ctx.reject_bad_scale * band_rate[k])
                       for k in b_rej])
    y_rej = (rng.random(len(rej)) < target).astype("float64")

    a = acc[ctx.features].copy()
    a["bad"] = acc["bad"].to_numpy()
    a["w"] = ctx.w_accepted
    r = rej[ctx.features].copy()
    r["bad"] = y_rej
    r["w"] = ctx.w_rejected
    frame = pd.concat([a, r], ignore_index=True)

    sc = refit(frame, ctx)
    est_rej = float(y_rej.mean())
    wa, wr = ctx.w_accepted * len(acc), ctx.w_rejected * len(rej)
    est_ttd = (wa * overall + wr * est_rej) / (wa + wr)

    return InferenceResult(
        name="parcelling",
        score_fn=lambda X: sc.predict_proba(X),
        est_reject_bad_rate=est_rej, est_ttd_bad_rate=est_ttd,
        diagnostics={"scale_factor": ctx.reject_bad_scale,
                     "n_bands": int(len(edges) + 1),
                     "accepted_bad_rate": overall,
                     "band_rates": {str(k): v for k, v in band_rate.items()}},
        scorecard=sc, train_frame=frame)
