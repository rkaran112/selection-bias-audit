"""Tests that would actually catch a wrong answer.

These do not touch the LendingClub files. Everything here runs on synthetic
data with a known truth, which is the only way to test an estimator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import dblquad
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sba import metrics as M                                  # noqa: E402
from sba.config import ScorecardConfig, WOEConfig             # noqa: E402
from sba.inference.heckman import (bvn_cdf, _negll_grad,      # noqa: E402
                                   fit_bivariate_probit)
from sba.scorecard import fit_scorecard                       # noqa: E402
from sba.woe import WOEBinner                                 # noqa: E402


# ---------------------------------------------------------------- BVN CDF
def test_bvn_cdf_matches_closed_form_at_origin():
    """Phi2(0,0,rho) = 1/4 + arcsin(rho)/(2 pi) exactly."""
    for rho in (-0.9, -0.4, 0.0, 0.3, 0.7, 0.95):
        exact = 0.25 + np.arcsin(rho) / (2 * np.pi)
        got = float(bvn_cdf(np.array([0.0]), np.array([0.0]), rho)[0])
        assert abs(got - exact) < 1e-10, f"rho={rho}"


def test_bvn_cdf_matches_numerical_integration():
    def dens(x, y, r):
        return np.exp(-(x * x - 2 * r * x * y + y * y) / (2 * (1 - r * r))) \
            / (2 * np.pi * np.sqrt(1 - r * r))
    for h, k, r in [(0.5, -1.2, 0.7), (-1.5, 0.3, -0.6), (2.0, 1.0, 0.35)]:
        ref, _ = dblquad(lambda y, x: dens(x, y, r), -12, h, -12, k,
                         epsabs=1e-12, epsrel=1e-12)
        got = float(bvn_cdf(np.array([h]), np.array([k]), r)[0])
        assert abs(got - ref) < 1e-9, f"({h},{k},{r})"


def test_bvn_cdf_marginal_identity():
    h = np.array([-2.0, -0.5, 0.0, 1.3])
    for r in (-0.7, 0.0, 0.4, 0.9):
        got = bvn_cdf(h, np.full_like(h, 8.0), r)
        assert np.max(np.abs(got - norm.cdf(h))) < 1e-9


# ---------------------------------------------------- Heckman likelihood
def _sim_bvp(n, rho, seed, exclusion=True):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cols = [np.ones(n), x1, x2] + ([rng.normal(size=n)] if exclusion else [])
    Z = np.column_stack(cols)
    X = np.column_stack([np.ones(n), x1, x2])
    g = np.array([0.4, -0.9, 0.5] + ([0.8] if exclusion else []))
    b = np.array([-0.9, 0.6, -0.35])
    u = rng.normal(size=n)
    e = rho * u + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    s = ((Z @ g + u) > 0).astype(float)
    y = ((X @ b + e) > 0).astype(float) * s
    return Z, X, s, y, np.ones(n)


def test_heckman_analytic_gradient_matches_finite_difference():
    Z, X, s, y, w = _sim_bvp(3000, 0.45, 1)
    th = np.r_[np.array([0.3, -0.8, 0.4, 0.7]), np.array([-0.8, 0.5, -0.3]),
               np.arctanh(0.2)]
    _, ga = _negll_grad(th, Z, X, s, y, w, 4, 3)
    gn = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th); e[i] = 1e-6
        gn[i] = (_negll_grad(th + e, Z, X, s, y, w, 4, 3)[0]
                 - _negll_grad(th - e, Z, X, s, y, w, 4, 3)[0]) / 2e-6
    assert np.max(np.abs(ga - gn) / (np.abs(gn) + 1.0)) < 1e-4


@pytest.mark.parametrize("rho_true", [0.0, 0.4, -0.5])
def test_heckman_recovers_rho(rho_true):
    Z, X, s, y, w = _sim_bvp(30_000, rho_true, 7)
    f = fit_bivariate_probit(Z, X, s, y, w)
    assert f["converged"]
    assert abs(f["rho"] - rho_true) < 0.06, f"got {f['rho']}"


def test_heckman_does_not_invent_selection_bias_when_there_is_none():
    """The single most important negative result in the repo.

    Asserted as a rejection RATE across seeds, not on one draw. A calibrated
    test rejects a true null about 5% of the time by construction, so a
    single-seed assertion would be flaky for the most boring possible reason
    and would tell us nothing about whether the estimator is honest.

    A wider Monte Carlo (30 seeds, n=12,000) gives a rejection rate of 0.067
    at nominal 0.05 - within Monte Carlo error - and rho centred on -0.004.
    """
    ps, rhos = [], []
    for seed in range(8):
        Z, X, s, y, w = _sim_bvp(12_000, 0.0, 900 + seed)
        f = fit_bivariate_probit(Z, X, s, y, w)
        ps.append(f["lr_p"])
        rhos.append(f["rho"])
    n_reject = sum(p < 0.05 for p in ps)
    assert n_reject <= 2, (
        f"rejected the true null {n_reject}/8 times: the LR test is not "
        f"calibrated, p-values were {[round(p, 4) for p in ps]}")
    assert abs(float(np.mean(rhos))) < 0.05, (
        f"rho is biased away from zero: mean {np.mean(rhos):.4f}")


# ------------------------------------------------------------- metrics
def test_gini_of_random_score_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000).astype(float)
    assert abs(M.gini(y, rng.normal(size=20_000))) < 0.03


def test_gini_of_perfect_score_is_one():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert M.gini(y, y) > 0.999


def test_weighted_auc_equals_unweighted_when_rows_duplicated():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000).astype(float)
    s = y * 0.6 + rng.normal(size=2000)
    w = rng.integers(1, 4, 2000).astype(float)
    idx = np.repeat(np.arange(2000), w.astype(int))
    assert abs(M.auc(y, s, w) - M.auc(y[idx], s[idx])) < 1e-9


def test_effective_sample_size():
    assert M.effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    w = np.r_[np.full(99, 1.0), 1000.0]
    assert M.effective_sample_size(w) < 10


# ----------------------------------------------------------------- WOE
def _toy(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    c = rng.choice(list("ABCD"), size=n)
    p = 1 / (1 + np.exp(-(-1.0 + 1.2 * x + (c == "A") * 0.8)))
    return pd.DataFrame({"x": x, "c": c}), (rng.random(n) < p).astype(float)


def test_woe_is_monotonic_when_asked():
    X, y = _toy()
    b = WOEBinner(["x"], [], WOEConfig(monotonic=True)).fit(X, y)
    rates = [bb.bad_rate for bb in b.features_["x"].bins if not bb.is_missing]
    d = np.diff(rates)
    assert (d >= -1e-9).all() or (d <= 1e-9).all(), rates


def test_woe_respects_min_bin_size():
    X, y = _toy()
    cfg = WOEConfig(min_bin_frac=0.10)
    b = WOEBinner(["x"], [], cfg).fit(X, y)
    ns = [bb.n for bb in b.features_["x"].bins if not bb.is_missing]
    assert min(ns) >= 0.10 * len(X) * 0.999


def test_woe_handles_missing_as_its_own_bin():
    X, y = _toy()
    X = X.copy(); X.loc[:999, "x"] = np.nan
    b = WOEBinner(["x"], [], WOEConfig()).fit(X, y)
    assert any(bb.is_missing for bb in b.features_["x"].bins)
    assert np.isfinite(b.transform(X)).all()


def test_woe_weights_change_the_bins():
    """If weights were ignored, three of four methods would be broken."""
    X, y = _toy()
    w = np.where(X["x"] > 0, 10.0, 1.0)
    a = WOEBinner(["x"], [], WOEConfig()).fit(X, y).features_["x"]
    b = WOEBinner(["x"], [], WOEConfig()).fit(X, y, w).features_["x"]
    assert not np.allclose([bb.woe for bb in a.bins][:2],
                           [bb.woe for bb in b.bins][:2])


# ----------------------------------------------------------- scorecard
def test_scorecard_points_sum_to_the_scaled_score():
    X, y = _toy()
    sc = fit_scorecard(X, y, None, numeric=["x"], categorical=["c"],
                       woe_cfg=WOEConfig(), sc_cfg=ScorecardConfig())
    pts = sc.points(X)
    lo = sc.log_odds_bad(X)
    assert np.allclose(pts, sc.offset + sc.factor * (-lo))


def test_scorecard_beats_chance():
    X, y = _toy()
    sc = fit_scorecard(X, y, None, numeric=["x"], categorical=["c"],
                       woe_cfg=WOEConfig(), sc_cfg=ScorecardConfig())
    assert M.gini(y, sc.predict_proba(X)) > 0.4


# ------------------------------------------------ simulation acceptance
def test_acceptance_mask_hits_the_target_rejection_rate():
    from sba import simulate as S
    from sba.config import Config
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"risk_score": rng.normal(700, 30, 50_000)})
    cfg = Config()
    for ct in ("risk_hard", "risk_soft", "random"):
        for rr in (0.1, 0.5, 0.7):
            m = S.acceptance_mask(df, S.Scenario(ct, rr, 0), cfg)
            assert abs((1 - m.mean()) - rr) < 0.02, (ct, rr, 1 - m.mean())


def test_hard_cutoff_hits_target_on_a_lumpy_discrete_score():
    """Real FICO is discrete with big mass points at the policy floor.

    A plain quantile cut overshoots there - a nominal 10% declines 17.7% on the
    real data - which would silently mislabel the x-axis of the whole sweep.
    """
    from sba import simulate as S
    from sba.config import Config
    rng = np.random.default_rng(0)
    # 40% of the population piled on one value, as at a policy floor
    score = np.where(rng.random(60_000) < 0.4, 662.0,
                     np.round(rng.normal(700, 25, 60_000) / 5) * 5)
    df = pd.DataFrame({"risk_score": score})
    for rr in (0.1, 0.3, 0.5, 0.7):
        m = S.acceptance_mask(df, S.Scenario("risk_hard", rr, 0), Config())
        got = 1 - m.mean()
        assert abs(got - rr) < 0.01, f"asked {rr:.0%}, declined {got:.2%}"


def test_random_cutoff_is_uncorrelated_with_risk():
    from sba import simulate as S
    from sba.config import Config
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"risk_score": rng.normal(700, 30, 50_000)})
    m = S.acceptance_mask(df, S.Scenario("random", 0.5, 0), Config())
    r = np.corrcoef(m.astype(float), df["risk_score"])[0, 1]
    assert abs(r) < 0.02, f"null cutoff correlated with risk: r={r}"
    m2 = S.acceptance_mask(df, S.Scenario("risk_hard", 0.5, 0), Config())
    assert np.corrcoef(m2.astype(float), df["risk_score"])[0, 1] > 0.5
