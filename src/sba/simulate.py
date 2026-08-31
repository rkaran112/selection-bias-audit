"""Part 3 - the simulation harness.

Reject inference is normally unfalsifiable: you cannot check an inferred bad
rate against outcomes that do not exist. This module manufactures the missing
ground truth.

Take only accepted loans, where every outcome is known. Impose an artificial
cutoff and throw away the outcomes below it, pretending those applicants were
declined. Every method then runs on a synthetic reject population whose true
bad rate we know exactly but never show it. Sweeping the cutoff from 10% to 70%
rejection shows where each method stops working.

Three cutoff shapes are run:

  risk_hard  deterministic threshold on the risk score. This is the textbook
             description of a credit policy and the hardest case for any
             selection model, because approval is a deterministic function of
             an observed covariate and there is nothing left to identify.
  risk_soft  the same threshold with judgemental noise, which is what real
             underwriting looks like once overrides and manual review are in.
  random     the NULL. Selection is independent of risk, so the accepted sample
             is unbiased and the correct answer is "no correction needed". Any
             method that reports a large correction here is hallucinating.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit

from . import config as C
from . import metrics as M
from .inference import METHOD_ORDER, base as ibase
from .scorecard import fit_scorecard

EVAL_FRAC = 0.30


@dataclass(frozen=True)
class Scenario:
    cutoff_type: str
    rejection_rate: float
    replicate: int

    @property
    def key(self) -> str:
        return f"{self.cutoff_type}@{self.rejection_rate:.0%}#{self.replicate}"


# --------------------------------------------------------------------------
def _scenario_seed(sc: Scenario, base: int) -> int:
    """Deterministic across processes.

    Python's built-in hash() is randomised per interpreter for str, so using it
    here would make the run irreproducible between sessions - which is exactly
    the guarantee this repo makes. crc32 is stable.
    """
    key = f"{sc.cutoff_type}|{sc.rejection_rate:.6f}|{sc.replicate}".encode()
    return (zlib.crc32(key) + base) % (2 ** 31)


def acceptance_mask(df: pd.DataFrame, sc: Scenario, cfg: C.Config
                    ) -> np.ndarray:
    """True where the artificial policy approves the applicant."""
    rng = np.random.default_rng(_scenario_seed(sc, cfg.seed))
    n = len(df)
    if sc.cutoff_type == "random":
        return rng.random(n) >= sc.rejection_rate

    score = df["risk_score"].to_numpy("float64")
    # A missing bureau score is itself a decline reason, so treat it as the
    # worst rank rather than dropping the row.
    score = np.where(np.isnan(score), np.nanmin(score) - 1.0, score)

    if sc.cutoff_type == "risk_hard":
        # A discrete score cannot land on an arbitrary rejection rate. FICO has
        # large mass points - 662 is the modal value, at LendingClub's policy
        # floor - so a plain `score > quantile` overshoots badly: a nominal 10%
        # cutoff actually declines 17.7%. Ties at the threshold are therefore
        # broken by a seeded draw, which is what a real policy does when some
        # secondary criterion decides the boundary cases. The swept rate is then
        # the rate that was actually applied, which is what the charts claim.
        thr = np.quantile(score, sc.rejection_rate)
        below = score < thr
        at = score == thr
        shortfall = sc.rejection_rate * n - below.sum()
        at_reject = np.zeros(n, dtype=bool)
        if at.any() and shortfall > 0:
            frac = min(shortfall / at.sum(), 1.0)
            at_reject = at & (rng.random(n) < frac)
        return ~(below | at_reject)

    if sc.cutoff_type == "risk_soft":
        z = (score - score.mean()) / max(score.std(), 1e-9)
        b = cfg.sim.soft_slope
        target = 1.0 - sc.rejection_rate

        def gap(a):
            return expit(a + b * z).mean() - target

        a = brentq(gap, -50, 50, xtol=1e-10)
        return rng.random(n) < expit(a + b * z)

    raise ValueError(f"unknown cutoff_type {sc.cutoff_type!r}")


def _claimed_metrics(view: pd.DataFrame, score: np.ndarray) -> dict:
    """What the method itself would report, from its own labelled view."""
    return {"claimed_gini": M.gini(view["bad"].to_numpy(), score,
                                   view["w"].to_numpy()),
            "claimed_ks": M.ks(view["bad"].to_numpy(), score,
                               view["w"].to_numpy())}


# --------------------------------------------------------------------------
def run_scenario(pool: pd.DataFrame, sc: Scenario, cfg: C.Config,
                 methods: list | None = None,
                 reject_bad_scale: float = 2.0) -> list[dict]:
    methods = methods or METHOD_ORDER
    feats = C.COMMON_FEATURES + C.COMMON_CATEGORICAL

    rng = np.random.default_rng(cfg.seed + 977 * sc.replicate)
    idx = rng.choice(len(pool), size=min(cfg.sim.pool_size, len(pool)),
                     replace=False)
    d = pool.iloc[np.sort(idx)].reset_index(drop=True)

    is_eval = rng.random(len(d)) < EVAL_FRAC
    d = d.assign(_eval=is_eval)
    d = d.assign(_acc=acceptance_mask(d, sc, cfg))

    fit_d = d[~d["_eval"]].reset_index(drop=True)
    ev_d = d[d["_eval"]].reset_index(drop=True)

    acc = fit_d[fit_d["_acc"]].reset_index(drop=True)
    rej = fit_d[~fit_d["_acc"]].reset_index(drop=True)
    if len(acc) < 2000 or len(rej) < 500:
        return []

    # ---- the truth we deliberately hid -----------------------------------
    y_ev = ev_d["bad"].to_numpy("float64")
    truth_ttd_bad = float(y_ev.mean())
    truth_rej_bad = float(ev_d.loc[~ev_d["_acc"], "bad"].mean())
    truth_acc_bad = float(ev_d.loc[ev_d["_acc"], "bad"].mean())

    # ---- oracle: what a scorecard fitted on the WHOLE pool would achieve --
    oracle = fit_scorecard(fit_d[feats], fit_d["bad"].to_numpy(), None,
                           numeric=C.COMMON_FEATURES,
                           categorical=C.COMMON_CATEGORICAL,
                           woe_cfg=cfg.woe, sc_cfg=cfg.scorecard)
    oracle_gini = M.gini(y_ev, oracle.predict_proba(ev_d[feats]))

    # ---- the biased baseline every lender has ----------------------------
    baseline = fit_scorecard(acc[feats], acc["bad"].to_numpy(), None,
                             numeric=C.COMMON_FEATURES,
                             categorical=C.COMMON_CATEGORICAL,
                             woe_cfg=cfg.woe, sc_cfg=cfg.scorecard)
    base_pd_ev = baseline.predict_proba(ev_d[feats])
    base_actual_gini = M.gini(y_ev, base_pd_ev)
    ev_acc = ev_d["_acc"].to_numpy()
    base_reported_gini = M.gini(y_ev[ev_acc], base_pd_ev[ev_acc])

    common = {
        "cutoff_type": sc.cutoff_type, "rejection_rate": sc.rejection_rate,
        "replicate": sc.replicate,
        "n_pool": len(d), "n_accepted": len(acc), "n_rejected": len(rej),
        # Nominal vs realised: the score is discrete, so a quantile cut cannot
        # always land exactly on the requested rate. Recorded, not assumed.
        "realised_rejection_rate": len(rej) / max(len(acc) + len(rej), 1),
        "truth_ttd_bad_rate": truth_ttd_bad,
        "truth_reject_bad_rate": truth_rej_bad,
        "truth_accepted_bad_rate": truth_acc_bad,
        "oracle_ttd_gini": oracle_gini,
        "baseline_reported_gini": base_reported_gini,
        "baseline_actual_ttd_gini": base_actual_gini,
    }

    common["reject_bad_scale"] = reject_bad_scale
    rows = [{
        **common, "method": "none (biased baseline)",
        "est_ttd_bad_rate": truth_acc_bad,
        "est_reject_bad_rate": np.nan,
        "claimed_gini": base_reported_gini,
        "actual_ttd_gini": base_actual_gini,
        "bad_rate_bias": truth_acc_bad - truth_ttd_bad,
        "gini_bias": base_reported_gini - base_actual_gini,
        "gini_uplift": 0.0,
        "diagnostics": {},
    }]

    # ---- the four corrections --------------------------------------------
    ctx = ibase.InferenceContext(
        accepted=acc, rejected=rej, baseline=baseline,
        numeric=C.COMMON_FEATURES, categorical=C.COMMON_CATEGORICAL,
        woe_cfg=cfg.woe, sc_cfg=cfg.scorecard,
        seed=cfg.seed + 31 * sc.replicate, w_accepted=1.0, w_rejected=1.0,
        reject_bad_scale=reject_bad_scale)

    for name in methods:
        try:
            res = ibase.REGISTRY[name](ctx)
        except Exception as exc:                    # noqa: BLE001
            rows.append({**common, "method": name, "error": repr(exc)})
            continue
        s_ev = res.score(ev_d[feats])
        actual = M.gini(y_ev, s_ev)
        claimed = np.nan
        if res.train_frame is not None:
            view = res.train_frame
            claimed = _claimed_metrics(view, res.score(view[feats]))["claimed_gini"]
        rows.append({
            **common, "method": name,
            "est_ttd_bad_rate": res.est_ttd_bad_rate,
            "est_reject_bad_rate": res.est_reject_bad_rate,
            "claimed_gini": claimed,
            "actual_ttd_gini": actual,
            "bad_rate_bias": res.est_ttd_bad_rate - truth_ttd_bad,
            "reject_bad_rate_bias": res.est_reject_bad_rate - truth_rej_bad,
            "gini_bias": claimed - actual,
            "gini_uplift": actual - base_actual_gini,
            "diagnostics": res.diagnostics,
        })
    return rows


# --------------------------------------------------------------------------
def run_sweep(pool: pd.DataFrame, cfg: C.Config, methods=None,
              progress=None) -> pd.DataFrame:
    out = []
    scenarios = [Scenario(ct, rr, rep)
                 for ct in cfg.sim.cutoff_types
                 for rr in cfg.sim.rejection_rates
                 for rep in range(cfg.sim.n_replicates)]
    for i, sc in enumerate(scenarios, 1):
        if progress:
            progress(i, len(scenarios), sc)
        out.extend(run_scenario(pool, sc, cfg, methods))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def scale_sensitivity(pool: pd.DataFrame, cfg: C.Config,
                      scales=(1.0, 1.5, 2.0, 3.0),
                      methods=("parcelling", "fuzzy"),
                      cutoff_types=("risk_hard", "random"),
                      n_reps: int = 2) -> pd.DataFrame:
    """How much of parcelling's and fuzzy's error is the scale factor?

    Both methods need an exogenous multiple for how much worse declines are
    than approvals. Ranking them last while holding that multiple fixed at a
    number they did not choose would be a strawman, so this re-runs them across
    a range of it.

    Two things fall out. On a risk-based cutoff their error is dominated by the
    multiple, and a well-chosen one would score respectably - except that
    nothing in the data tells you which one is well-chosen. On the NULL cutoff
    only a multiple of exactly 1.0 avoids inventing a correction, and you would
    have to already know there was no selection bias in order to pick it. The
    null-case failure is structural, not a tuning problem.
    """
    # A subset of the swept rates: this establishes a mechanism, and the full
    # sweep in run_sweep already covers the rate dimension.
    rates = [r for r in cfg.sim.rejection_rates if 0.2 <= r <= 0.6] \
        or list(cfg.sim.rejection_rates)
    out = []
    for scale in scales:
        for ct in cutoff_types:
            for rr in rates:
                for rep in range(n_reps):
                    rows = run_scenario(pool, Scenario(ct, rr, rep), cfg,
                                        methods=list(methods),
                                        reject_bad_scale=scale)
                    out.extend(r for r in rows
                               if r.get("method") in set(methods))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return (df.groupby(["method", "reject_bad_scale", "cutoff_type"],
                       as_index=False)
            .agg(abs_bad_rate_bias=("bad_rate_bias",
                                    lambda s: s.abs().mean()),
                 bad_rate_bias=("bad_rate_bias", "mean"),
                 abs_gini_bias=("gini_bias", lambda s: s.abs().mean()))
            .sort_values(["cutoff_type", "method", "reject_bad_scale"]))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and across-replicate SD of every bias measure."""
    keep = df[df.get("error").isna()] if "error" in df else df
    g = keep.groupby(["cutoff_type", "rejection_rate", "method"], as_index=False)
    agg = g.agg(
        bad_rate_bias=("bad_rate_bias", "mean"),
        bad_rate_bias_sd=("bad_rate_bias", "std"),
        abs_bad_rate_bias=("bad_rate_bias", lambda s: s.abs().mean()),
        gini_bias=("gini_bias", "mean"),
        gini_bias_sd=("gini_bias", "std"),
        abs_gini_bias=("gini_bias", lambda s: s.abs().mean()),
        gini_uplift=("gini_uplift", "mean"),
        gini_uplift_sd=("gini_uplift", "std"),
        actual_ttd_gini=("actual_ttd_gini", "mean"),
        truth_ttd_bad_rate=("truth_ttd_bad_rate", "mean"),
        oracle_ttd_gini=("oracle_ttd_gini", "mean"),
        realised_rejection_rate=("realised_rejection_rate", "mean"),
        n_reps=("replicate", "nunique"),
    )
    return agg.sort_values(["cutoff_type", "rejection_rate", "method"])


def degradation_verdict(summary: pd.DataFrame, tol_pp: float = 2.0,
                        cutoff_types=("risk_hard", "risk_soft")) -> dict:
    """Where each method stops working, as a sentence with the evidence.

    The brief for this repo asked for an answer shaped like "method X is least
    biased up to a 50% rejection rate, beyond which all four fail" - a ranking
    with a breaking point attached, rather than a preference. This computes the
    breaking point: the highest swept rejection rate at which a method's mean
    absolute bad-rate bias has stayed within tolerance at every rate up to and
    including it.
    """
    s = summary[summary["cutoff_type"].isin(cutoff_types)]
    g = (s.groupby(["method", "rejection_rate"], as_index=False)
         ["abs_bad_rate_bias"].mean())
    rates = sorted(g["rejection_rate"].unique())

    holds: dict[str, float | None] = {}
    for method, gm in g.groupby("method"):
        gm = gm.set_index("rejection_rate")["abs_bad_rate_bias"]
        last = None
        for r in rates:
            if r in gm.index and 100 * gm[r] <= tol_pp:
                last = r
            else:
                break
        holds[method] = last

    survivors = {m: r for m, r in holds.items()
                 if r is not None and m != BASELINE}
    if not survivors:
        sentence = (f"No method held its estimate of the through-the-door bad "
                    f"rate within {tol_pp:.0f} pp even at the mildest "
                    f"rejection rate swept ({rates[0]:.0%}).")
    else:
        best_r = max(survivors.values())
        best = sorted(m for m, r in survivors.items() if r == best_r)
        beyond = ("beyond which every method exceeds it"
                  if best_r < rates[-1] else
                  "and holds across the whole swept range")
        sentence = (
            f"{' and '.join(best)} held within {tol_pp:.0f} pp up to a "
            f"{best_r:.0%} rejection rate, {beyond}.")
        # Whether any correction actually beat doing nothing is the question a
        # lender is really asking, and it is easy to lose behind a ranking.
        base_r = holds.get(BASELINE)
        if base_r is not None and base_r >= best_r:
            sentence += (f" Applying no correction at all held equally far "
                         f"({base_r:.0%}), so on this evidence none of the "
                         f"four earned its place.")
    return {"tolerance_pp": tol_pp, "holds_up_to": holds,
            "swept_rates": rates, "verdict": sentence,
            "baseline_holds_up_to": holds.get(BASELINE)}


def rank_methods(summary: pd.DataFrame, cutoff_types=("risk_hard", "risk_soft")
                 ) -> pd.DataFrame:
    """Rank on absolute bad-rate bias, then Gini bias, then instability.

    Deliberately a ranking with the evidence attached, not a preference.
    """
    s = summary[summary["cutoff_type"].isin(cutoff_types)]
    r = s.groupby("method", as_index=False).agg(
        mean_abs_bad_rate_bias=("abs_bad_rate_bias", "mean"),
        mean_abs_gini_bias=("abs_gini_bias", "mean"),
        mean_gini_uplift=("gini_uplift", "mean"),
        instability=("gini_bias_sd", "mean"),
    )
    return r.sort_values(["mean_abs_bad_rate_bias", "mean_abs_gini_bias"])


BASELINE = "none (biased baseline)"


def null_case_check(sim: pd.DataFrame, tol_bad: float = 0.01) -> pd.DataFrame:
    """Under a random cutoff the honest answer is 'no correction needed'.

    A fixed tolerance would be arbitrary, so each method is compared against
    the no-correction baseline on the SAME scenarios. The baseline's error
    under a random cutoff is pure sampling noise - it is what "doing nothing"
    costs - so a method only counts as hallucinating if its error is reliably
    worse than that, by a paired test across replicates and rejection rates,
    and by a margin worth caring about.
    """
    s = sim[sim["cutoff_type"] == "random"].copy()
    if "error" in s:
        s = s[s["error"].isna()]
    s["abs_bias"] = s["bad_rate_bias"].abs()
    key = ["rejection_rate", "replicate"]
    base = s[s["method"] == BASELINE].set_index(key)["abs_bias"]

    rows = []
    for method, g in s.groupby("method"):
        g = g.set_index(key)
        pair = g.join(base.rename("base_abs"), how="inner").dropna(
            subset=["abs_bias", "base_abs"])
        delta = (pair["abs_bias"] - pair["base_abs"]).to_numpy("float64")
        n = len(delta)
        mean_d = float(delta.mean()) if n else np.nan
        if method == BASELINE or n < 3 or np.allclose(delta, 0):
            p = 1.0
        else:
            sd = delta.std(ddof=1)
            p = (1.0 if sd <= 0 else
                 float(stats.t.sf(mean_d / (sd / np.sqrt(n)), df=n - 1)))
        rows.append({
            "method": method,
            "mean_abs_bad_rate_bias": float(pair["abs_bias"].mean()),
            "baseline_abs_bad_rate_bias": float(pair["base_abs"].mean()),
            "excess_over_baseline": mean_d,
            "p_one_sided": p,
            "n_pairs": n,
            "mean_abs_gini_bias": float(g["gini_bias"].abs().mean()),
        })
    r = pd.DataFrame(rows)
    r["hallucinates"] = (r["excess_over_baseline"] > tol_bad) & (
        r["p_one_sided"] < 0.05) & (r["method"] != BASELINE)
    r["verdict"] = np.where(
        r["hallucinates"],
        "FAILS null: manufactures a correction where none exists",
        "passes null: no correction reported, as there should not be")
    return r.sort_values("mean_abs_bad_rate_bias").reset_index(drop=True)
