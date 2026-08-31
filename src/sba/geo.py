"""Part 4c - residual geographic disparity in the decline decision.

What this does: model each applicant's probability of being declined from
their risk characteristics ALONE - no state, no ZIP - then aggregate to the
3-digit ZIP and ask whether the observed number of declines differs from what
risk predicts by more than sampling noise allows.

What this does NOT do, and cannot: say anything about race, ethnicity, sex,
age, or any other protected attribute. This repo never joins to census data,
never infers demographics from geography, and makes no proxy claim. A 3-digit
ZIP covers hundreds of thousands of people. The only claim available here is
"applications from this ZIP were declined more often than their recorded risk
characteristics predict", and the explanation for that could be branch
footprint, marketing mix, local income documentation practice, an omitted risk
variable this thin feature set cannot see, or something that warrants a closer
look. Distinguishing between those is a fair-lending review, not a script.

This output is a flag for where to look. It is not a finding of discrimination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .config import WOEConfig, ScorecardConfig
from .woe import WOEBinner

MIN_APPS_PER_ZIP = 500          # below this the residual is mostly noise
FLAG_Z = 3.0


def fit_risk_only_decline_model(stacked: pd.DataFrame, risk_features: list,
                                woe_cfg: WOEConfig, sc_cfg: ScorecardConfig,
                                weights: np.ndarray | None = None):
    """P(declined | risk characteristics), deliberately blind to geography."""
    y = 1 - stacked["accepted"].to_numpy("float64")
    w = np.ones(len(stacked)) if weights is None else weights
    binner = WOEBinner(risk_features, [], woe_cfg).fit(stacked, y, w)
    Z = binner.transform(stacked)
    m = LogisticRegression(C=sc_cfg.C, max_iter=2000, solver="lbfgs")
    m.fit(Z, y, sample_weight=w)
    return binner, m, m.predict_proba(Z)[:, 1]


def zip_residuals(stacked: pd.DataFrame, p_decline: np.ndarray,
                  min_apps: int = MIN_APPS_PER_ZIP) -> pd.DataFrame:
    """Observed minus risk-predicted declines per 3-digit ZIP, standardised.

    Under the null that risk fully explains the decision, each ZIP's count of
    declines is Poisson-binomial with mean sum(p) and variance sum(p(1-p)), so
    the standardised residual is approximately standard normal.

    Deliberately computed on unweighted sample counts. Approvals and declines
    are sampled at different rates, so the decline share in this frame is not
    the platform's true decline rate - but both groups are simple random
    samples drawn independently of geography, so that distortion is uniform
    across ZIPs and is absorbed by the model's intercept. It shifts the
    absolute level and leaves the between-ZIP comparison, which is the whole
    point here, intact. Applying design weights instead would break the
    binomial variance the test relies on.
    """
    d = pd.DataFrame({
        "zip3": stacked["zip3"].to_numpy(),
        "declined": (1 - stacked["accepted"].to_numpy()).astype("float64"),
        "p": p_decline,
    })
    g = d.groupby("zip3", as_index=False).agg(
        n_apps=("declined", "size"),
        observed_declines=("declined", "sum"),
        expected_declines=("p", "sum"),
        var=("p", lambda s: float(np.sum(s * (1 - s)))),
    )
    g = g[g["n_apps"] >= min_apps].reset_index(drop=True)
    g["observed_rate"] = g["observed_declines"] / g["n_apps"]
    g["expected_rate"] = g["expected_declines"] / g["n_apps"]
    g["excess_rate_pp"] = 100.0 * (g["observed_rate"] - g["expected_rate"])
    g["z"] = (g["observed_declines"] - g["expected_declines"]) \
        / np.sqrt(np.maximum(g["var"], 1e-9))
    g["flagged"] = g["z"].abs() > FLAG_Z
    return g.sort_values("z", ascending=False).reset_index(drop=True)


def dispersion_test(zres: pd.DataFrame) -> dict:
    """Is there more between-ZIP variation than risk plus noise can produce?"""
    z = zres["z"].to_numpy("float64")
    z = z[np.isfinite(z)]
    J = len(z)
    if J == 0:
        return {"n_zip3": 0}
    chi2 = float(np.sum(z ** 2))
    p = float(stats.chi2.sf(chi2, df=J))
    n_flag = int((np.abs(z) > FLAG_Z).sum())
    expected_flag = 2 * stats.norm.sf(FLAG_Z) * J
    return {
        "n_zip3": J,
        "chi2": chi2,
        "chi2_df": J,
        "chi2_p": p,
        "dispersion_ratio": chi2 / J,      # 1.0 under the null
        "sd_of_z": float(np.std(z)),
        "n_flagged_abs_z_gt_3": n_flag,
        "n_flagged_expected_by_chance": float(expected_flag),
        "excess_flags": n_flag - float(expected_flag),
        "max_excess_decline_pp": float(zres["excess_rate_pp"].max()),
        "min_excess_decline_pp": float(zres["excess_rate_pp"].min()),
        "iqr_excess_decline_pp": float(
            zres["excess_rate_pp"].quantile(0.75)
            - zres["excess_rate_pp"].quantile(0.25)),
    }


def analyse(stacked: pd.DataFrame, risk_features: list, woe_cfg: WOEConfig,
            sc_cfg: ScorecardConfig, weights=None) -> tuple[pd.DataFrame, dict]:
    _, _, p = fit_risk_only_decline_model(stacked, risk_features, woe_cfg,
                                          sc_cfg, weights)
    zres = zip_residuals(stacked, p)
    return zres, dispersion_test(zres)
