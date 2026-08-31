"""Central configuration. Every knob that moves a headline number lives here."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
DOCS = ROOT / "docs"
SLIDES = ROOT / "slides"

SEED = 20260901

# ---------------------------------------------------------------- data files
ACCEPTED_FILE = "accepted_2007_to_2018Q4.csv.gz"
REJECTED_FILE = "rejected_2007_to_2018Q4.csv.gz"
KAGGLE_SLUG = "wordsforthewise/lending-club"

# --------------------------------------------------------------- the overlap
# These six fields are the ENTIRE intersection of the accepted and rejected
# schemas. Everything this tool can say about rejected applicants is a function
# of these and nothing else. See README "The thin-overlap ceiling".
COMMON_FEATURES = ["risk_score", "dti", "emp_length_yrs", "amount_requested"]
COMMON_CATEGORICAL = ["state"]
COMMON_CONTEXT = ["zip3", "app_year"]  # used for geography / vintage, not scoring

# --------------------------------------------------------------- target
# Terminal statuses only. Anything still open is "indeterminate" and dropped:
# including live loans would understate the bad rate by counting not-yet-
# defaulted loans as good.
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}


@dataclass
class WOEConfig:
    max_bins: int = 8
    min_bin_frac: float = 0.05
    monotonic: bool = True
    smoothing: float = 0.5          # Laplace count added to good/bad per bin
    rare_cat_frac: float = 0.01     # categories below this share are pooled


@dataclass
class ScorecardConfig:
    pdo: float = 20.0               # points to double the odds
    base_points: float = 600.0
    base_odds: float = 50.0         # good:bad at base_points
    C: float = 1.0                  # inverse L2 strength on the WOE logistic
    test_frac: float = 0.30


@dataclass
class SimConfig:
    """Part 3 — the synthetic-truth harness."""
    pool_size: int = 60_000         # accepted loans drawn per replicate
    rejection_rates: tuple = (0.10, 0.30, 0.50, 0.70)
    cutoff_types: tuple = ("risk_hard", "risk_soft", "random")
    n_replicates: int = 5
    soft_slope: float = 2.5         # logit slope for the probabilistic cutoff


@dataclass
class EconConfig:
    """Part 4 - every money assumption, stated in one place.

    The Part 4 conclusion is genuinely sensitive to these numbers, so run_all
    emits a sensitivity grid over them rather than quoting one figure as though
    it had been measured.
    """
    # Weighted-average-life of an amortising instalment loan as a fraction of
    # its stated term. 36m loan repaid on schedule has WAL ~= 19-20 months.
    wal_fraction_of_term: float = 0.55
    # Blended cost of funds for a balance-sheet lender over 2007-2018. Today's
    # 4.5%+ would be anachronistic for this vintage; 3.0% is defensible, and the
    # sensitivity grid shows what 2.0% and 4.5% do to the answer.
    cost_of_funds_annual: float = 0.030
    servicing_cost_annual: float = 0.010
    sensitivity_cost_of_funds: tuple = (0.020, 0.030, 0.045)
    # LGD: calibrated from LendingClub recoveries where possible, else this.
    lgd_fallback: float = 0.80
    # Bootstrap replicates for the profit confidence interval.
    n_bootstrap: int = 2000
    ci_level: float = 0.90


@dataclass
class Config:
    seed: int = SEED
    # Row caps keep one command runnable on a laptop. Both are seeded samples;
    # design weights restore the population accept rate (see data.load_panel).
    max_accepted: int = 400_000
    # Rejected file is 27,648,741 rows; a seeded Bernoulli sample at this
    # rate is drawn in one streaming pass. Design weight is exactly 1/rate,
    # so the population acceptance rate survives into the propensity model.
    reject_sample_rate: float = 0.03
    # 67% of declined applications carry no Risk_Score. A scorecard fitted on
    # accepted loans - where the score is NEVER missing - has no bin for
    # "missing", so it would score those applicants as merely average, which is
    # both wrong and flattering. Restricting the reject-inference population to
    # scored declines is the honest option; the cost is stated in the README.
    require_reject_score: bool = True
    woe: WOEConfig = field(default_factory=WOEConfig)
    scorecard: ScorecardConfig = field(default_factory=ScorecardConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    econ: EconConfig = field(default_factory=EconConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_INTERIM, OUTPUTS, FIGURES, TABLES, DOCS, SLIDES):
        d.mkdir(parents=True, exist_ok=True)
