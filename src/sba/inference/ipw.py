"""(c) Inverse-propensity reweighting.

Model P(accepted | X), then reweight each accepted applicant by 1/p so the
accepted sample is made to look like the through-the-door population. No
rejected outcome is ever invented; the rejects only inform the propensity
model. That is IPW's real virtue.

Its assumption is missingness-at-random: conditional on the four observable
fields, whether an applicant was approved carries no further information about
whether they would have defaulted. That is a strong claim about LendingClub,
whose underwriters saw a full credit bureau file this model cannot see. Where
IPW breaks, it breaks because of that, and Part 3 measures the damage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import metrics as M
from ..woe import WOEBinner
from .base import InferenceContext, InferenceResult, refit, register

TRIM_PCTL = 99.0        # cap weights here to stop one applicant owning the fit
MIN_P = 1e-4


@register("ipw")
def ipw(ctx: InferenceContext) -> InferenceResult:
    acc, rej = ctx.accepted, ctx.rejected
    feats = ctx.features

    stacked = pd.concat([acc[feats], rej[feats]], ignore_index=True)
    s = np.r_[np.ones(len(acc)), np.zeros(len(rej))]
    dw = np.r_[np.full(len(acc), ctx.w_accepted),
               np.full(len(rej), ctx.w_rejected)]

    # Same WOE representation as the scorecard, so the propensity model is not
    # quietly better specified than the thing it is correcting.
    binner = WOEBinner(ctx.numeric, ctx.categorical, ctx.woe_cfg).fit(
        stacked, s, dw)
    Z = binner.transform(stacked)
    ps_model = LogisticRegression(C=ctx.sc_cfg.C, max_iter=2000, solver="lbfgs")
    ps_model.fit(Z, s, sample_weight=dw)

    p_acc = np.clip(ps_model.predict_proba(Z[:len(acc)])[:, 1], MIN_P, 1.0)
    raw_w = 1.0 / p_acc
    cap = float(np.percentile(raw_w, TRIM_PCTL))
    w = np.minimum(raw_w, cap) * ctx.w_accepted

    frame = acc[feats].copy()
    frame["bad"] = acc["bad"].to_numpy()
    frame["w"] = w
    sc = refit(frame, ctx)

    # IPW's estimate of the through-the-door bad rate is the reweighted mean of
    # the observed one. It infers no reject labels, so its implied reject bad
    # rate has to be backed out of the identity.
    est_ttd = M.weighted_mean(acc["bad"].to_numpy(), w)
    obs = float(np.mean(acc["bad"].to_numpy()))
    wa, wr = ctx.w_accepted * len(acc), ctx.w_rejected * len(rej)
    a_share = wa / (wa + wr)
    est_rej = (est_ttd - a_share * obs) / max(1.0 - a_share, 1e-9)

    return InferenceResult(
        name="ipw",
        score_fn=lambda X: sc.predict_proba(X),
        est_reject_bad_rate=float(np.clip(est_rej, 0.0, 1.0)),
        est_ttd_bad_rate=float(est_ttd),
        diagnostics={
            "propensity_auc": M.auc(s, ps_model.predict_proba(Z)[:, 1], dw),
            "mean_propensity_accepted": float(p_acc.mean()),
            "min_propensity_accepted": float(p_acc.min()),
            "weight_cap_pctl": TRIM_PCTL,
            "frac_weights_capped": float((raw_w > cap).mean()),
            "ess_ratio": float(M.effective_sample_size(w) / len(w)),
        },
        scorecard=sc, train_frame=frame)
