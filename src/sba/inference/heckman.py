"""(d) Heckman-type selection correction: bivariate probit with selection.

The textbook Heckman two-step assumes a continuous outcome. Default is binary,
so the correct member of that family is the Van de Ven & Van Praag (1981)
bivariate probit with sample selection, estimated by full maximum likelihood:

    selection   S* = z'g + u ,   S = 1[S* > 0]        (approved)
    outcome     Y* = x'b + e ,   Y = 1[Y* > 0]        (defaulted)
                (u, e) ~ BVN(0, 0, 1, 1, rho)
    Y observed only where S = 1.

rho is the whole ballgame. It is the correlation between whatever unobserved
factors drove the approval decision and whatever unobserved factors drove
default. rho = 0 means approval carried no information beyond x, and the
accepts-only scorecard is unbiased. This module estimates rho, puts a standard
error and a likelihood-ratio test on it, and reports what it finds - including,
if that is the answer, that it is indistinguishable from zero.

Identification. With no exclusion restriction the model is identified only off
the nonlinearity of the normal CDF, which is weak and widely criticised. This
implementation therefore supports an exclusion restriction (variables in the
selection equation only) and the pipeline reports rho with and without one, so
the reader can see how much of the answer is functional form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import norm

from ..woe import WOEBinner
from .base import InferenceContext, InferenceResult, register

# Gauss-Legendre nodes for the BVN integral. 24 gives ~1e-10 absolute error
# for |rho| < 0.9, which is far tighter than anything else in this pipeline.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_LOG_EPS = 1e-12
_MAX_ATANH = 4.0        # |rho| <= 0.9993
HECKMAN_MAX_N = 200_000


# --------------------------------------------------------------------------
def bvn_cdf(h, k, rho):
    """P(H <= h, K <= k) for standard bivariate normal with correlation rho.

    Uses  Phi2(h,k,r) = Phi(h)Phi(k) + (1/2pi) * int_0^r
          (1-t^2)^-0.5 exp( -(h^2 - 2thk + k^2) / (2(1-t^2)) ) dt
    which is exact and, on Gauss-Legendre nodes, fully vectorised.
    """
    h = np.asarray(h, dtype="float64")
    k = np.asarray(k, dtype="float64")
    base = norm.cdf(h) * norm.cdf(k)
    if abs(rho) < 1e-14:
        return np.clip(base, _LOG_EPS, 1.0)
    t = 0.5 * rho * (_GL_X + 1.0)
    wq = 0.5 * rho * _GL_W
    om = 1.0 - t * t
    hh, kk = h[:, None], k[:, None]
    integ = np.exp(-(hh * hh - 2.0 * t * hh * kk + kk * kk) / (2.0 * om)) \
        / np.sqrt(om)
    return np.clip(base + (integ @ wq) / (2.0 * np.pi), _LOG_EPS, 1.0)


def bvn_pdf(h, k, rho):
    om = 1.0 - rho * rho
    return np.exp(-(h * h - 2.0 * rho * h * k + k * k) / (2.0 * om)) \
        / (2.0 * np.pi * np.sqrt(om))


# --------------------------------------------------------------------------
def _negll_grad(theta, Z, X, s, y, w, kz, kx):
    """Negative weighted log-likelihood and its analytic gradient.

    theta = [gamma (kz), beta (kx), atanh(rho)]
    """
    g = theta[:kz]
    b = theta[kz:kz + kx]
    a = np.clip(theta[-1], -_MAX_ATANH, _MAX_ATANH)
    rho = np.tanh(a)
    root = np.sqrt(max(1.0 - rho * rho, 1e-14))

    wI = Z @ g                       # selection index, all rows
    sel = s > 0.5
    v = X[sel] @ b                   # outcome index, accepted rows only
    wsel = wI[sel]
    ysel = y[sel]
    wt_sel, wt_rej = w[sel], w[~sel]

    grad = np.zeros_like(theta)
    ll = 0.0

    # ---- rejected: log Phi(-wI)
    if (~sel).any():
        wr = wI[~sel]
        P = np.clip(norm.cdf(-wr), _LOG_EPS, 1.0)
        ll += np.sum(wt_rej * np.log(P))
        lam = norm.pdf(wr) / P
        grad[:kz] += -(Z[~sel].T @ (wt_rej * lam))

    # ---- accepted: log Phi2(wI, +-v, +-rho)
    if sel.any():
        sgn = np.where(ysel > 0.5, 1.0, -1.0)        # +1 bad, -1 good
        k2 = sgn * v
        r2 = sgn * rho
        # bvn_cdf takes a scalar rho, so split by outcome sign
        L = np.empty_like(wsel)
        for sv in (1.0, -1.0):
            m = sgn == sv
            if m.any():
                L[m] = bvn_cdf(wsel[m], k2[m], sv * rho)
        L = np.clip(L, _LOG_EPS, 1.0)
        ll += np.sum(wt_sel * np.log(L))

        dh = norm.pdf(wsel) * norm.cdf((k2 - r2 * wsel) / root)
        dk = norm.pdf(k2) * norm.cdf((wsel - r2 * k2) / root)
        dr = bvn_pdf(wsel, k2, r2)

        grad[:kz] += Z[sel].T @ (wt_sel * dh / L)
        grad[kz:kz + kx] += X[sel].T @ (wt_sel * sgn * dk / L)
        grad[-1] += np.sum(wt_sel * sgn * dr / L)

    grad[-1] *= (1.0 - rho * rho)    # chain rule for atanh parameterisation
    return -ll, -grad


def _probit_start(A, t, wt):
    """Cheap starting values: an IRLS-free probit via logistic then rescale."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=1000, fit_intercept=False)
    m.fit(A, t, sample_weight=wt)
    return m.coef_[0] * 0.5875        # logit -> probit scale


# --------------------------------------------------------------------------
def fit_bivariate_probit(Z, X, s, y, w, *, verbose=False) -> dict:
    """Full-information ML. Returns coefficients, rho, SE and an LR test."""
    kz, kx = Z.shape[1], X.shape[1]
    sel = s > 0.5

    g0 = _probit_start(Z, s, w)
    b0 = _probit_start(X[sel], y[sel], w[sel])
    theta0 = np.r_[g0, b0, 0.0]

    args = (Z, X, s, y, w, kz, kx)
    res = optimize.minimize(_negll_grad, theta0, args=args, jac=True,
                            method="L-BFGS-B",
                            bounds=[(None, None)] * (kz + kx)
                                   + [(-_MAX_ATANH, _MAX_ATANH)],
                            options={"maxiter": 500, "ftol": 1e-10})
    theta = res.x
    ll_full = -res.fun

    # restricted model: rho = 0, i.e. two independent probits
    def _neg_restricted(th):
        f, gr = _negll_grad(np.r_[th, 0.0], *args)
        return f, gr[:-1]

    res0 = optimize.minimize(_neg_restricted, theta0[:-1], jac=True,
                             method="L-BFGS-B",
                             options={"maxiter": 500, "ftol": 1e-10})
    ll_restricted = -res0.fun

    lr_stat = max(2.0 * (ll_full - ll_restricted), 0.0)
    lr_p = float(stats.chi2.sf(lr_stat, df=1))

    # SE from a finite-difference Hessian of the analytic gradient
    def _grad_only(th):
        return _negll_grad(th, *args)[1]

    p = len(theta)
    H = np.zeros((p, p))
    step = 1e-5 * np.maximum(np.abs(theta), 1.0)
    for i in range(p):
        e = np.zeros(p)
        e[i] = step[i]
        H[:, i] = (_grad_only(theta + e) - _grad_only(theta - e)) / (2 * step[i])
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    a = float(np.clip(theta[-1], -_MAX_ATANH, _MAX_ATANH))
    rho = float(np.tanh(a))
    se_a = float(se[-1])
    se_rho = se_a * (1.0 - rho ** 2)          # delta method
    z_a = a / se_a if np.isfinite(se_a) and se_a > 0 else np.nan
    p_a = float(2 * norm.sf(abs(z_a))) if np.isfinite(z_a) else np.nan
    lo, hi = (np.tanh(a - 1.96 * se_a), np.tanh(a + 1.96 * se_a)) \
        if np.isfinite(se_a) else (np.nan, np.nan)

    return {
        "gamma": theta[:kz], "beta": theta[kz:kz + kx],
        "atanh_rho": a, "rho": rho, "se_rho": se_rho, "se_atanh_rho": se_a,
        "rho_ci95": (float(lo), float(hi)),
        "wald_z": float(z_a), "wald_p": p_a,
        "lr_stat": float(lr_stat), "lr_p": lr_p,
        "loglik": float(ll_full), "loglik_restricted": float(ll_restricted),
        "converged": bool(res.success), "n": int(len(s)),
        "n_selected": int(sel.sum()),
    }


# --------------------------------------------------------------------------
def _design(binner: WOEBinner, df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), binner.transform(df)])


@register("heckman")
def heckman(ctx: InferenceContext) -> InferenceResult:
    acc, rej = ctx.accepted, ctx.rejected
    feats = ctx.features
    excl = list(getattr(ctx, "exclusion", []) or [])

    rng = np.random.default_rng(ctx.seed + 7)
    n_tot = len(acc) + len(rej)
    if n_tot > HECKMAN_MAX_N:
        f = HECKMAN_MAX_N / n_tot
        ia = np.sort(rng.choice(len(acc), max(int(len(acc) * f), 100), False))
        ir = np.sort(rng.choice(len(rej), max(int(len(rej) * f), 100), False))
        acc_f, rej_f = acc.iloc[ia], rej.iloc[ir]
        sw_a = ctx.w_accepted * len(acc) / len(acc_f)
        sw_r = ctx.w_rejected * len(rej) / len(rej_f)
    else:
        acc_f, rej_f, sw_a, sw_r = acc, rej, ctx.w_accepted, ctx.w_rejected

    stacked = pd.concat([acc_f[feats + excl], rej_f[feats + excl]],
                        ignore_index=True)
    s = np.r_[np.ones(len(acc_f)), np.zeros(len(rej_f))]
    y = np.r_[acc_f["bad"].to_numpy("float64"), np.zeros(len(rej_f))]
    w = np.r_[np.full(len(acc_f), sw_a), np.full(len(rej_f), sw_r)]
    w = w / w.mean()          # scale-free; keeps the Hessian well conditioned

    sel_num = ctx.numeric + [c for c in excl if c not in ctx.categorical]
    sel_cat = ctx.categorical
    bin_sel = WOEBinner(sel_num, sel_cat, ctx.woe_cfg).fit(stacked, s, w)
    bin_out = WOEBinner(ctx.numeric, ctx.categorical, ctx.woe_cfg).fit(
        acc_f[feats], acc_f["bad"].to_numpy("float64"),
        np.full(len(acc_f), 1.0))

    Z = _design(bin_sel, stacked)
    X = _design(bin_out, stacked)
    fit = fit_bivariate_probit(Z, X, s, y, w)

    beta, gamma, rho = fit["beta"], fit["gamma"], fit["rho"]

    def score_fn(df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(_design(bin_out, df) @ beta)

    # E[Y | S=0, x] = Phi2(-w, v, -rho) / Phi(-w): the model's own statement
    # about the applicants it never saw repay. Computed once on the full
    # declined set, and reused for both the point estimate and the claimed
    # view below, so the two can never disagree.
    wI_r = _design(bin_sel, rej[feats + excl]) @ gamma
    v_r = _design(bin_out, rej[feats]) @ beta
    p_rej = np.clip(bvn_cdf(-wI_r, v_r, -rho)
                    / np.clip(norm.cdf(-wI_r), 1e-9, 1.0), 0.0, 1.0)
    est_rej = float(p_rej.mean())

    obs = float(np.mean(acc["bad"].to_numpy()))
    wa, wr = ctx.w_accepted * len(acc), ctx.w_rejected * len(rej)
    est_ttd = (wa * obs + wr * est_rej) / (wa + wr)

    diag = {k: fit[k] for k in
            ("rho", "se_rho", "rho_ci95", "wald_z", "wald_p", "lr_stat",
             "lr_p", "loglik", "loglik_restricted", "converged", "n",
             "n_selected")}
    diag["exclusion"] = excl
    diag["rho_significant_5pct"] = bool(
        np.isfinite(fit["lr_p"]) and fit["lr_p"] < 0.05)

    # Heckman infers no hard labels, so its claimed view of the population uses
    # that same E[Y | S=0, x] as a fractional label - the direct analogue of
    # what fuzzy augmentation does, built from the selection-corrected model.
    a_df = acc[feats].copy()
    a_df["bad"] = acc["bad"].to_numpy()
    a_df["w"] = ctx.w_accepted
    rb = rej[feats].copy()
    rb["bad"] = 1.0
    rb["w"] = ctx.w_rejected * p_rej
    rg = rej[feats].copy()
    rg["bad"] = 0.0
    rg["w"] = ctx.w_rejected * (1.0 - p_rej)
    view = pd.concat([a_df, rb, rg], ignore_index=True)
    view = view[view["w"] > 1e-9].reset_index(drop=True)

    return InferenceResult(
        name="heckman", score_fn=score_fn,
        est_reject_bad_rate=est_rej, est_ttd_bad_rate=est_ttd,
        diagnostics=diag, scorecard=None, train_frame=view)
