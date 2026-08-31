"""One command. Builds every number in the README, the PDF and the deck.

    python run_all.py

Everything downstream reads outputs/headline_numbers.json, so no figure,
table or slide can drift from the numbers this script produced.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from sba import config as C                           # noqa: E402
from sba import data, economics, geo, metrics as M    # noqa: E402
from sba import plots, simulate as S                  # noqa: E402
from sba.inference import METHOD_ORDER, base as ibase  # noqa: E402
from sba.scorecard import fit_scorecard, train_test_split_seeded  # noqa: E402

T0 = time.time()
OUT: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / name, index=False)


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ==========================================================================
def main() -> None:
    C.ensure_dirs()
    cfg = C.Config()
    rng_seed = cfg.seed
    np.random.seed(rng_seed)

    OUT["meta"] = {
        "seed": rng_seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": json.loads(cfg.to_json()),
    }

    # ---------------------------------------------------------------- data
    log("loading LendingClub accepted + rejected ...")
    panel = data.load_panel(cfg)
    acc, rej = panel.accepted, panel.rejected
    feats = C.COMMON_FEATURES + C.COMMON_CATEGORICAL

    OUT["data"] = {
        "accepted_rows_total": panel.n_accepted_total,
        "accepted_rows_terminal": panel.n_accepted_terminal,
        "accepted_rows_used": len(acc),
        "indeterminate_fraction": panel.indeterminate_frac,
        "rejected_rows_total": panel.n_rejected_total,
        "rejected_rows_sampled_raw": panel.n_rejected_sampled_raw,
        "rejected_rows_used": len(rej),
        "unscored_reject_fraction": panel.unscored_reject_frac,
        "require_reject_score": cfg.require_reject_score,
        "modelled_accept_rate": panel.modelled_accept_rate,
        "reject_sample_rate": cfg.reject_sample_rate,
        "design_weight_accepted": panel.w_accepted,
        "design_weight_rejected": panel.w_rejected,
        "population_accept_rate": panel.population_accept_rate,
        "observed_bad_rate_accepted": float(acc["bad"].mean()),
        "coverage": panel.coverage.to_dict("records"),
        "accepted_score_p01": float(acc["risk_score"].quantile(0.01)),
        "rejected_score_missing_pct": float(
            100 * rej["risk_score"].isna().mean()),
        "scored_rejects_above_accepted_p01_pct": float(
            100 * (rej["risk_score"]
                   >= acc["risk_score"].quantile(0.01)).mean()),
    }
    save_table(panel.coverage, "00_field_coverage.csv")
    log(f"  accepted {len(acc):,} (of {panel.n_accepted_total:,}), "
        f"rejected {len(rej):,} (of {panel.n_rejected_total:,}), "
        f"accept rate {panel.population_accept_rate:.2%}")
    plots.fig_overlap(acc, rej)

    # ------------------------------------------------------- PART 1: baseline
    log("PART 1 - baseline scorecard on accepted loans only")
    tr, te = train_test_split_seeded(acc, cfg.scorecard.test_frac, rng_seed)
    baseline = fit_scorecard(tr[feats], tr["bad"].to_numpy(), None,
                             numeric=C.COMMON_FEATURES,
                             categorical=C.COMMON_CATEGORICAL,
                             woe_cfg=cfg.woe, sc_cfg=cfg.scorecard)
    m_tr = baseline.evaluate(tr[feats], tr["bad"].to_numpy())
    m_te = baseline.evaluate(te[feats], te["bad"].to_numpy())
    OUT["part1_baseline"] = {
        "train": m_tr, "test": m_te,
        "n_train": len(tr), "n_test": len(te),
        "iv": baseline.binner.iv_table().to_dict("records"),
        "pdo": cfg.scorecard.pdo, "base_points": cfg.scorecard.base_points,
        "base_odds": cfg.scorecard.base_odds,
    }
    save_table(baseline.points_table(), "01_scorecard_points.csv")
    save_table(baseline.binner.bin_table(), "02_woe_bins.csv")
    save_table(baseline.binner.iv_table(), "03_information_value.csv")
    plots.fig_baseline(te["bad"].to_numpy(), baseline.predict_proba(te[feats]))
    log(f"  reported Gini (test) = {m_te['gini']:.4f}  "
        f"KS = {m_te['ks']:.4f}  AUC = {m_te['auc']:.4f}")

    # --------------------------------------- PART 2: four corrections, real data
    log("PART 2 - reject inference on the real declined population")
    ctx = ibase.InferenceContext(
        accepted=acc, rejected=rej, baseline=baseline,
        numeric=C.COMMON_FEATURES, categorical=C.COMMON_CATEGORICAL,
        woe_cfg=cfg.woe, sc_cfg=cfg.scorecard, seed=rng_seed,
        w_accepted=panel.w_accepted, w_rejected=panel.w_rejected)

    real: dict = {}
    scores_real: dict = {}
    for name in METHOD_ORDER:
        t = time.time()
        res = ibase.REGISTRY[name](ctx)
        scores_real[name] = res
        claimed = np.nan
        if res.train_frame is not None:
            v = res.train_frame
            claimed = M.gini(v["bad"].to_numpy(), res.score(v[feats]),
                             v["w"].to_numpy())
        real[name] = {
            "est_reject_bad_rate": res.est_reject_bad_rate,
            "est_ttd_bad_rate": res.est_ttd_bad_rate,
            "claimed_ttd_gini": float(claimed),
            "gini_vs_baseline_reported": float(claimed - m_te["gini"]),
            "diagnostics": res.diagnostics,
            "seconds": time.time() - t,
        }
        log(f"  {name:12s} TTD bad rate {res.est_ttd_bad_rate:.4f}  "
            f"claimed Gini {claimed:.4f}  ({time.time() - t:.1f}s)")

    # Heckman a second time WITH an exclusion restriction, so the reader can
    # see how much of rho is identified by data and how much by functional form.
    log("  heckman with exclusion restriction (application year) ...")
    ctx_ex = ibase.InferenceContext(
        accepted=acc, rejected=rej, baseline=baseline,
        numeric=C.COMMON_FEATURES, categorical=C.COMMON_CATEGORICAL,
        woe_cfg=cfg.woe, sc_cfg=cfg.scorecard, seed=rng_seed,
        w_accepted=panel.w_accepted, w_rejected=panel.w_rejected,
        exclusion=["app_year"])
    res_ex = ibase.REGISTRY["heckman"](ctx_ex)
    real["heckman_with_exclusion"] = {
        "est_reject_bad_rate": res_ex.est_reject_bad_rate,
        "est_ttd_bad_rate": res_ex.est_ttd_bad_rate,
        "diagnostics": res_ex.diagnostics,
    }
    OUT["part2_real_data"] = real
    hk, hx = real["heckman"]["diagnostics"], res_ex.diagnostics
    log(f"  rho (no exclusion)   = {hk['rho']:+.4f} "
        f"(se {hk['se_rho']:.4f}, LR p = {hk['lr_p']:.3g})")
    log(f"  rho (with exclusion) = {hx['rho']:+.4f} "
        f"(se {hx['se_rho']:.4f}, LR p = {hx['lr_p']:.3g})")

    save_table(pd.DataFrame([
        {"method": k,
         "est_reject_bad_rate": v["est_reject_bad_rate"],
         "est_ttd_bad_rate": v["est_ttd_bad_rate"],
         "claimed_ttd_gini": v.get("claimed_ttd_gini")}
        for k, v in real.items()]), "04_real_data_methods.csv")

    # positivity: IPW is only valid where declined applicants could plausibly
    # have been approved. Report how badly that fails here.
    ipw_d = real["ipw"]["diagnostics"]
    OUT["part2_positivity"] = {
        "propensity_auc": ipw_d["propensity_auc"],
        "min_propensity_accepted": ipw_d["min_propensity_accepted"],
        "ess_ratio": ipw_d["ess_ratio"],
        "frac_weights_capped": ipw_d["frac_weights_capped"],
    }

    # ------------------------------------------------- PART 3: the harness
    log("PART 3 - simulation harness (manufactured ground truth)")
    sim_rows = S.run_sweep(
        acc, cfg,
        progress=lambda i, n, sc: log(f"  scenario {i:>2}/{n}  {sc.key}"))
    sim = pd.DataFrame(sim_rows)
    summary = S.summarise(sim)
    ranking = S.rank_methods(summary)
    nullcase = S.null_case_check(sim)
    verdict = S.degradation_verdict(summary)

    save_table(sim.drop(columns=["diagnostics"], errors="ignore"),
               "05_simulation_raw.csv")
    save_table(summary, "06_simulation_summary.csv")
    save_table(ranking, "07_method_ranking.csv")
    save_table(nullcase, "08_null_case.csv")

    plots.fig_sweep(summary)
    plots.fig_sweep(summary, ycol="gini_bias", sdcol="gini_bias_sd",
                    name="03b_sweep_gini.png", scale=1.0,
                    ylabel="claimed Gini minus actual Gini",
                    suptitle="Bias in the Gini each method reports")
    plots.fig_gini_levels(summary)
    plots.fig_null_case(nullcase)

    rho_rows = []
    for _, r in sim[sim["method"] == "heckman"].iterrows():
        d = r.get("diagnostics") or {}
        if "rho" in d:
            rho_rows.append({"cutoff_type": r["cutoff_type"],
                             "rejection_rate": r["rejection_rate"],
                             "rho": d["rho"], "se_rho": d["se_rho"],
                             "lr_p": d["lr_p"]})
    rho_df = pd.DataFrame(rho_rows)
    if len(rho_df):
        rho_agg = rho_df.groupby(["cutoff_type", "rejection_rate"],
                                 as_index=False).mean(numeric_only=True)
        save_table(rho_agg, "09_heckman_rho_sweep.csv")
        plots.fig_rho(rho_agg)
        OUT["part3_rho_sweep"] = rho_agg.to_dict("records")

    # Is the parcelling/fuzzy verdict an artefact of the scale factor we picked?
    log("  scale-factor sensitivity for parcelling / fuzzy ...")
    scale_df = S.scale_sensitivity(acc, cfg)
    if len(scale_df):
        save_table(scale_df, "15_scale_factor_sensitivity.csv")
        OUT["part3_scale_sensitivity"] = scale_df.to_dict("records")
        for ct in scale_df["cutoff_type"].unique():
            sub = scale_df[scale_df["cutoff_type"] == ct]
            best = sub.loc[sub.groupby("method")["abs_bad_rate_bias"].idxmin()]
            for _, r in best.iterrows():
                log(f"    {ct:10s} {r['method']:11s} best at scale "
                    f"{r['reject_bad_scale']:.1f} -> "
                    f"{100 * r['abs_bad_rate_bias']:.2f} pp")

    OUT["part3_simulation"] = {
        "summary": summary.to_dict("records"),
        "ranking": ranking.to_dict("records"),
        "null_case": nullcase.to_dict("records"),
        "pool_size": cfg.sim.pool_size,
        "n_replicates": cfg.sim.n_replicates,
        "rejection_rates": list(cfg.sim.rejection_rates),
        "degradation": verdict,
    }
    log("  ranking (risk-based cutoffs, best first):")
    for _, r in ranking.iterrows():
        log(f"    {r['method']:24s} |bad-rate bias| "
            f"{100 * r['mean_abs_bad_rate_bias']:5.2f} pp   "
            f"|Gini bias| {r['mean_abs_gini_bias']:.4f}   "
            f"uplift {r['mean_gini_uplift']:+.4f}")
    log(f"  VERDICT: {verdict['verdict']}")
    for _, r in nullcase.iterrows():
        log(f"    NULL {r['method']:20s} {r['verdict']}")

    # ------------------------------------------------ PART 4: what it costs
    log("PART 4 - swap set, money, geography")
    yc = economics.calibrate_yields(acc, baseline.predict_proba(acc[feats]),
                                    cfg.econ)
    OUT["part4_yields"] = {
        "good_yield_per_dollar": yc.good_yield,
        "bad_yield_per_dollar": yc.bad_yield,
        "empirical_lgd": yc.lgd,
        "n_good": yc.n_good, "n_bad": yc.n_bad,
        "assumptions": {
            "wal_fraction_of_term": cfg.econ.wal_fraction_of_term,
            "cost_of_funds_annual": cfg.econ.cost_of_funds_annual,
            "servicing_cost_annual": cfg.econ.servicing_cost_annual,
            "ci_level": cfg.econ.ci_level,
            "n_bootstrap": cfg.econ.n_bootstrap,
        },
    }
    save_table(yc.by_band, "10_realised_yields_by_band.csv")
    log(f"  realised yield: good {yc.good_yield:+.4f}/$  "
        f"bad {yc.bad_yield:+.4f}/$  empirical LGD {yc.lgd:.3f}")

    applicants = pd.concat([acc[feats + ["accepted"]],
                            rej[feats + ["accepted"]]], ignore_index=True)
    design_w = np.r_[np.full(len(acc), panel.w_accepted),
                     np.full(len(rej), panel.w_rejected)]
    pd_base_all = baseline.predict_proba(applicants[feats])
    acc_flag = applicants["accepted"].to_numpy().astype(bool)

    be36 = economics.breakeven_bad_rate(yc, cfg.econ, 36.0)
    be60 = economics.breakeven_bad_rate(yc, cfg.econ, 60.0)
    log(f"  break-even bad rate: {be36:.1%} (36m)  {be60:.1%} (60m) "
        f"at {cfg.econ.cost_of_funds_annual:.1%} cost of funds")

    econ_rows = []
    for name in METHOD_ORDER:
        res = scores_real[name]
        pd_corr = res.score(applicants[feats])
        e = economics.analyse(applicants, pd_base_all, pd_corr, acc_flag,
                              design_w, yc, cfg.econ, rng_seed)
        e["method"] = name
        econ_rows.append(e)
        if "profit_forgone_population" in e:
            log(f"  {name:12s} swap-in {e['swap_in_applicants']:>10,.0f} "
                f"({e['swap_rate']:.1%} of book)  inferred bad "
                f"{e['swap_in_inferred_bad_rate']:.3f}  profit "
                f"${e['profit_forgone_population'] / 1e6:,.0f}m "
                f"[{e['profit_forgone_ci_lo'] / 1e6:,.0f}, "
                f"{e['profit_forgone_ci_hi'] / 1e6:,.0f}]")
    econ_df = pd.DataFrame(econ_rows)
    save_table(econ_df, "11_swap_set_economics.csv")
    plots.fig_swap(econ_df)

    # "How many flip" is not an answer to "which applicants flip".
    prof = economics.swap_in_profile(
        applicants, pd_base_all,
        scores_real["heckman"].score(applicants[feats]), acc_flag, design_w)
    save_table(prof, "14_swap_in_profile.csv")
    OUT["part4_swap_profile"] = prof.to_dict("records")
    log("  swap-in profile (Heckman-corrected model):")
    for _, r in prof.iterrows():
        log(f"    {r['group']:26s} n={r['applicants']:>11,.0f}  "
            f"score {r['risk_score']:6.1f}  dti {r['dti']:5.1f}  "
            f"amt ${r['amount_requested']:,.0f}")

    # The profit answer is assumption-sensitive; publish the grid, not one cell.
    sens = economics.sensitivity_grid(
        applicants, pd_base_all, scores_real["heckman"].score(applicants[feats]),
        acc_flag, design_w, yc, cfg.econ, rng_seed)
    save_table(sens, "13_profit_sensitivity.csv")

    vals = econ_df["profit_forgone_population"].dropna()
    OUT["part4_economics"] = {
        "by_method": econ_df.to_dict("records"),
        "range_low": float(vals.min()) if len(vals) else None,
        "range_high": float(vals.max()) if len(vals) else None,
        "breakeven_bad_rate_36m": be36,
        "breakeven_bad_rate_60m": be60,
        "sensitivity_to_cost_of_funds": sens.to_dict("records"),
        "all_methods_agree_sign": bool(len(vals) and
                                       (vals > 0).all() or (vals < 0).all()),
    }

    # geography
    stacked = data.stack_for_propensity(panel)
    zres, disp = geo.analyse(stacked, C.COMMON_FEATURES, cfg.woe, cfg.scorecard)
    save_table(zres, "12_zip3_residuals.csv")
    plots.fig_geo(zres, disp)
    OUT["part4_geography"] = {
        "dispersion": disp,
        "top_excess": zres.head(10).to_dict("records"),
        "bottom_excess": zres.tail(10).to_dict("records"),
        "min_apps_per_zip3": geo.MIN_APPS_PER_ZIP,
        "disclaimer": ("No protected attribute is inferred anywhere in this "
                       "analysis. The claim is unexplained geographic "
                       "variation in the decline rate, and nothing stronger."),
    }
    log(f"  {disp['n_zip3']} ZIP3s, dispersion ratio "
        f"{disp['dispersion_ratio']:.1f} (1.0 = risk explains everything), "
        f"{disp['n_flagged_abs_z_gt_3']} flagged vs "
        f"{disp['n_flagged_expected_by_chance']:.1f} expected by chance")

    # ---------------------------------------------------------------- write
    OUT["meta"]["runtime_seconds"] = round(time.time() - T0, 1)
    path = C.OUTPUTS / "headline_numbers.json"
    path.write_text(json.dumps(jsonable(OUT), indent=2))
    log(f"wrote {path}")
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
