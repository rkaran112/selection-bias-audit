"""Part 4a/4b - the swap set, and what the cutoff costs.

Two rules govern this module.

First, calibrate rather than assume wherever the data allows. LendingClub
publishes realised cash flows for every terminated loan, so the yield on a good
loan, the loss on a bad one and the loss-given-default are measured from those
cash flows, not asserted. Only the cost of funds, the servicing cost and the
weighted-average-life convention are assumptions, and they live in EconConfig
where a reader can change them and re-run.

Second, the profit number inherits the bias of whichever inference method
supplied the bad rate. Part 3 shows those methods disagree by several
percentage points, so this module reports the estimate under every method
rather than picking one and quoting it as the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EconConfig

N_YIELD_BANDS = 10


# --------------------------------------------------------------------------
@dataclass
class YieldCurve:
    """Realised per-dollar economics, measured from LendingClub cash flows."""
    good_yield: float           # net return per $1 funded on a loan repaid
    bad_yield: float            # net return per $1 funded on a charge-off
    lgd: float                  # 1 - (principal + recoveries) / funded
    by_band: pd.DataFrame
    n_good: int
    n_bad: int


def calibrate_yields(acc: pd.DataFrame, score: np.ndarray,
                     cfg: EconConfig) -> YieldCurve:
    """Per-dollar realised return, overall and by score band.

    net = (principal repaid + interest received + recoveries - amount funded)
          / amount funded
    """
    d = acc.copy()
    d["_score"] = score
    funded = d["funded_amnt"].to_numpy("float64")
    ok = funded > 0
    recovered = (d["total_rec_prncp"].to_numpy("float64")
                 + d["total_rec_int"].to_numpy("float64")
                 + d["recoveries"].to_numpy("float64"))
    d["_net"] = np.where(ok, (recovered - funded) / np.where(ok, funded, 1), np.nan)
    # LGD on the principal only: what fraction of the advance never came back.
    prin_back = (d["total_rec_prncp"].to_numpy("float64")
                 + d["recoveries"].to_numpy("float64"))
    d["_lgd"] = np.where(ok, 1.0 - prin_back / np.where(ok, funded, 1), np.nan)

    good = d[d["bad"] == 0]
    bad = d[d["bad"] == 1]
    edges = np.unique(np.quantile(score, np.linspace(0, 1, N_YIELD_BANDS + 1)[1:-1]))
    d["_band"] = np.digitize(d["_score"], edges, right=True)

    rows = []
    for b, g in d.groupby("_band"):
        gg, bb = g[g["bad"] == 0], g[g["bad"] == 1]
        rows.append({
            "band": int(b), "n": len(g),
            "bad_rate": float(g["bad"].mean()),
            "good_yield": float(gg["_net"].mean()) if len(gg) else np.nan,
            "bad_yield": float(bb["_net"].mean()) if len(bb) else np.nan,
            "lgd": float(bb["_lgd"].clip(0, 1).mean()) if len(bb) else np.nan,
            "mean_int_rate": float(g["int_rate"].mean()),
            "mean_term": float(g["term_months"].mean()),
        })
    by_band = pd.DataFrame(rows).sort_values("band").reset_index(drop=True)

    lgd = float(bad["_lgd"].clip(0, 1).mean()) if len(bad) else cfg.lgd_fallback
    return YieldCurve(
        good_yield=float(good["_net"].mean()),
        bad_yield=float(bad["_net"].mean()),
        lgd=lgd, by_band=by_band, n_good=len(good), n_bad=len(bad))


# --------------------------------------------------------------------------
@dataclass
class SwapSet:
    swap_in: pd.DataFrame       # declined today, approved by the corrected model
    swap_out: pd.DataFrame      # approved today, declined by the corrected model
    w_book: float               # weighted size of today's book
    cutoff_baseline: float
    cutoff_corrected: float


def _weighted_quantile(values: np.ndarray, q: float, w: np.ndarray) -> float:
    o = np.argsort(values)
    v, ww = values[o], w[o]
    c = np.cumsum(ww) / ww.sum()
    return float(np.interp(q, c, v))


def build_swap_set(applicants: pd.DataFrame, pd_baseline: np.ndarray,
                   pd_corrected: np.ndarray, accepted_flag: np.ndarray,
                   design_w: np.ndarray) -> SwapSet:
    """Volume-neutral swap, on the design-weighted applicant population.

    The corrected model approves exactly as many applicants as the lender
    approves today. Holding volume fixed isolates the quality of the ranking;
    it is not a recommendation to grow the book.

    The weights matter: accepted and declined applicants are sampled at very
    different rates, so an unweighted quantile would place the cutoff on a
    population that does not exist.
    """
    d = applicants.copy().reset_index(drop=True)
    d["pd_baseline"] = pd_baseline
    d["pd_corrected"] = pd_corrected
    d["accepted_today"] = accepted_flag.astype(bool)
    d["design_w"] = design_w

    w_book = float(d.loc[d["accepted_today"], "design_w"].sum())
    share = w_book / float(d["design_w"].sum())
    thr_corr = _weighted_quantile(d["pd_corrected"].to_numpy(), share,
                                  d["design_w"].to_numpy())
    thr_base = _weighted_quantile(d["pd_baseline"].to_numpy(), share,
                                  d["design_w"].to_numpy())
    d["accepted_corrected"] = d["pd_corrected"] <= thr_corr

    swap_in = d[~d["accepted_today"] & d["accepted_corrected"]].copy()
    swap_out = d[d["accepted_today"] & ~d["accepted_corrected"]].copy()
    return SwapSet(swap_in, swap_out, w_book, thr_base, thr_corr)


# --------------------------------------------------------------------------
def _band_of(score: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(score, edges, right=True)


def expected_profit(applicants: pd.DataFrame, pd_hat: np.ndarray,
                    band: np.ndarray, yc: YieldCurve, cfg: EconConfig
                    ) -> np.ndarray:
    """Expected dollars of profit per applicant, using band-level realised
    yields and a stated funding + servicing drag."""
    bb = yc.by_band.set_index("band")
    gy = bb["good_yield"].reindex(band).fillna(yc.good_yield).to_numpy()
    by = bb["bad_yield"].reindex(band).fillna(yc.bad_yield).to_numpy()
    term = bb["mean_term"].reindex(band).fillna(36.0).to_numpy()

    amount = applicants["amount_requested"].to_numpy("float64")
    amount = np.where(np.isfinite(amount), amount, np.nanmedian(amount))
    wal_years = cfg.wal_fraction_of_term * term / 12.0
    drag = (cfg.cost_of_funds_annual + cfg.servicing_cost_annual) * wal_years

    per_dollar = (1.0 - pd_hat) * gy + pd_hat * by - drag
    return amount * per_dollar


def breakeven_bad_rate(yc: YieldCurve, cfg: EconConfig,
                       term_months: float = 36.0,
                       cost_of_funds: float | None = None) -> float:
    """The bad rate at which a loan exactly breaks even.

    Reported prominently because the profit answer lives or dies on whether the
    swap-in set sits above or below this line, and the line moves a long way
    with the funding-cost assumption.
    """
    cof = cfg.cost_of_funds_annual if cost_of_funds is None else cost_of_funds
    drag = (cof + cfg.servicing_cost_annual) \
        * cfg.wal_fraction_of_term * term_months / 12.0
    return float((yc.good_yield - drag) / (yc.good_yield - yc.bad_yield))


def bootstrap_profit(profit: np.ndarray, w: np.ndarray, cfg: EconConfig,
                     seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap over applicants, carrying the design weights.

    This captures sampling error in the swap set ONLY. It does not capture the
    far larger uncertainty in the inferred bad rate itself, which Part 3
    quantifies separately and which the README states in plain words.
    """
    rng = np.random.default_rng(seed)
    n = len(profit)
    if n == 0:
        return 0.0, 0.0, 0.0
    wp = profit * w
    idx = rng.integers(0, n, size=(cfg.n_bootstrap, n))
    totals = wp[idx].sum(axis=1)
    lo = (1.0 - cfg.ci_level) / 2.0
    return (float(wp.sum()), float(np.quantile(totals, lo)),
            float(np.quantile(totals, 1.0 - lo)))


def analyse(applicants: pd.DataFrame, pd_baseline: np.ndarray,
            pd_corrected: np.ndarray, accepted_flag: np.ndarray,
            design_w: np.ndarray, yc: YieldCurve, cfg: EconConfig,
            seed: int) -> dict:
    """Full swap-set + profit picture for one inference method."""
    ss = build_swap_set(applicants, pd_baseline, pd_corrected, accepted_flag,
                        design_w)
    edges = np.unique(np.quantile(pd_baseline,
                                  np.linspace(0, 1, N_YIELD_BANDS + 1)[1:-1]))

    w_in = ss.swap_in["design_w"].to_numpy("float64")
    # How much does this model's ranking agree with the decision the lender
    # actually made? A low value means the swap set is large mostly because a
    # four-field model ranks differently from a full bureau file - not because
    # a pile of good business is sitting in the decline bin. Reported so the
    # swap set is never read as a list of people to go and approve.
    agree = float(np.corrcoef(pd_corrected,
                              1.0 - accepted_flag.astype(float))[0, 1])
    out = {
        "n_swap_in_rows": len(ss.swap_in), "n_swap_out_rows": len(ss.swap_out),
        "book_applicants": ss.w_book,
        "swap_in_applicants": float(w_in.sum()),
        "swap_rate": float(w_in.sum() / max(ss.w_book, 1.0)),
        "model_vs_actual_decision_corr": agree,
        "breakeven_bad_rate_36m": breakeven_bad_rate(yc, cfg, 36.0),
        "breakeven_bad_rate_60m": breakeven_bad_rate(yc, cfg, 60.0),
    }

    if len(ss.swap_in):
        p_in = ss.swap_in["pd_corrected"].to_numpy("float64")
        b_in = _band_of(ss.swap_in["pd_baseline"].to_numpy("float64"), edges)
        prof_in = expected_profit(ss.swap_in, p_in, b_in, yc, cfg)
        tot, lo, hi = bootstrap_profit(prof_in, w_in, cfg, seed)
        out.update({
            "swap_in_inferred_bad_rate": float(np.average(p_in, weights=w_in)),
            "swap_in_mean_amount": float(np.average(
                ss.swap_in["amount_requested"].fillna(
                    ss.swap_in["amount_requested"].median()), weights=w_in)),
            "profit_forgone_population": tot,
            "profit_forgone_ci_lo": lo,
            "profit_forgone_ci_hi": hi,
            "profit_per_swap_in_applicant": float(np.average(prof_in,
                                                             weights=w_in)),
            "frac_swap_in_profitable": float(
                np.average((prof_in > 0).astype(float), weights=w_in)),
        })
    if len(ss.swap_out):
        w_out = ss.swap_out["design_w"].to_numpy("float64")
        p_out = ss.swap_out["pd_corrected"].to_numpy("float64")
        b_out = _band_of(ss.swap_out["pd_baseline"].to_numpy("float64"), edges)
        prof_out = expected_profit(ss.swap_out, p_out, b_out, yc, cfg)
        out["swap_out_inferred_bad_rate"] = float(np.average(p_out,
                                                             weights=w_out))
        out["profit_avoided_population"] = float(-np.sum(prof_out * w_out))
    return out


PROFILE_FIELDS = ["risk_score", "dti", "emp_length_yrs", "amount_requested"]


def swap_in_profile(applicants: pd.DataFrame, pd_baseline: np.ndarray,
                    pd_corrected: np.ndarray, accepted_flag: np.ndarray,
                    design_w: np.ndarray) -> pd.DataFrame:
    """Who actually flips from decline to approve, and how do they compare?

    "How many" is not an answer to "which applicants". This puts the swap-in
    set side by side with today's book and with the declines that stay
    declined, so a credit officer can see whether the model is reaching for
    near-miss applicants or for a different population entirely.
    """
    ss = build_swap_set(applicants, pd_baseline, pd_corrected, accepted_flag,
                        design_w)
    d = applicants.copy().reset_index(drop=True)
    d["design_w"] = design_w
    d["accepted_today"] = accepted_flag.astype(bool)
    declined = d[~d["accepted_today"]]
    groups = {
        "approved today": d[d["accepted_today"]],
        "swap-in (would approve)": ss.swap_in,
        "swap-out (would decline)": ss.swap_out,
        "declined, stays declined": declined.loc[
            ~declined.index.isin(ss.swap_in.index)],
    }
    rows = []
    for name, g in groups.items():
        if not len(g):
            continue
        w = g["design_w"].to_numpy("float64")
        row = {"group": name, "applicants": float(w.sum())}
        for f in PROFILE_FIELDS:
            v = pd.to_numeric(g[f], errors="coerce").to_numpy("float64")
            ok = np.isfinite(v)
            row[f] = (float(np.average(v[ok], weights=w[ok]))
                      if ok.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_grid(applicants: pd.DataFrame, pd_baseline: np.ndarray,
                     pd_corrected: np.ndarray, accepted_flag: np.ndarray,
                     design_w: np.ndarray, yc: YieldCurve, cfg: EconConfig,
                     seed: int) -> pd.DataFrame:
    """Re-run the profit calculation across the funding-cost assumption.

    The point estimate is one cell of this table. Publishing only that cell
    would hide how much of the answer is assumption rather than measurement.
    """
    from dataclasses import replace
    rows = []
    for cof in cfg.sensitivity_cost_of_funds:
        c2 = replace(cfg, cost_of_funds_annual=cof, n_bootstrap=200)
        r = analyse(applicants, pd_baseline, pd_corrected, accepted_flag,
                    design_w, yc, c2, seed)
        rows.append({"cost_of_funds_annual": cof,
                     "breakeven_bad_rate_36m": r["breakeven_bad_rate_36m"],
                     "profit_forgone_population":
                         r.get("profit_forgone_population", np.nan),
                     "swap_in_inferred_bad_rate":
                         r.get("swap_in_inferred_bad_rate", np.nan)})
    return pd.DataFrame(rows)
