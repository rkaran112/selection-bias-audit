"""Load and harmonise the two LendingClub files onto their common schema.

The accepted file has ~150 columns. The rejected file has 9. This module throws
away the 140-odd columns that exist only for approved applicants, because a
model that uses them cannot be applied to a rejected applicant. That discipline
is the whole point of the exercise, and it is also the binding constraint on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

from . import config as C

# pandas' C parser materialises all ~150 columns of the accepted file before
# discarding the ones we did not ask for, which on a gzip source takes tens of
# minutes. pyarrow pushes the column selection into the reader and threads the
# decode: same result, ~30 seconds.
_ARROW_BLOCK = 1 << 24

# --------------------------------------------------------------------------
# column subsets
# --------------------------------------------------------------------------
ACC_USECOLS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "issue_d", "loan_status",
    "emp_length", "dti", "zip_code", "addr_state",
    "fico_range_low", "fico_range_high",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries",
    "grade",
]
REJ_USECOLS = [
    "Amount Requested", "Application Date", "Risk_Score",
    "Debt-To-Income Ratio", "Zip Code", "State", "Employment Length",
]

_EMP_MAP = {
    "< 1 year": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
    "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
    "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0,
}

# Plausibility windows. Values outside are set missing, not clipped: a
# self-reported DTI of 50,000% is not "a very high DTI", it is a bad record.
DTI_RANGE = (0.0, 100.0)
SCORE_RANGE = (300.0, 900.0)
AMT_RANGE = (500.0, 40_000.0)


def _emp_to_years(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().map(_EMP_MAP).astype("float32")


def _pct_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string").str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    ).astype("float32")


def _zip3(s: pd.Series) -> pd.Series:
    return s.astype("string").str.slice(0, 3).str.zfill(3)


def _window(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.where((s >= lo) & (s <= hi))


# --------------------------------------------------------------------------
# accepted
# --------------------------------------------------------------------------
def load_accepted(cfg: C.Config) -> pd.DataFrame:
    cache = C.DATA_INTERIM / "accepted.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    path = C.DATA_RAW / C.ACCEPTED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data`.")
    df = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=_ARROW_BLOCK),
        convert_options=pacsv.ConvertOptions(
            include_columns=ACC_USECOLS,
            column_types={c: pa.string() for c in
                          ("zip_code", "addr_state", "loan_status",
                           "emp_length", "term", "grade", "issue_d",
                           "int_rate")}),
    ).to_pandas()
    # A handful of trailing rows in the Kaggle export are footer junk with no
    # loan_amnt. Drop them rather than let them become a phantom bin.
    df = df[df["loan_amnt"].notna()].reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    # LendingClub reports FICO as a 4-point band; the midpoint is the score.
    out["risk_score"] = _window(
        (df["fico_range_low"].astype("float32")
         + df["fico_range_high"].astype("float32")) / 2.0, *SCORE_RANGE)
    out["dti"] = _window(df["dti"].astype("float32"), *DTI_RANGE)
    out["emp_length_yrs"] = _emp_to_years(df["emp_length"])
    out["amount_requested"] = _window(
        df["loan_amnt"].astype("float32"), *AMT_RANGE)
    out["state"] = df["addr_state"].fillna("XX").astype(str)
    out["zip3"] = _zip3(df["zip_code"]).astype(str)
    # issue_d is the funding date, not the application date. The rejected file
    # carries the application date. The gap is roughly 2-4 weeks; it matters
    # only for vintage alignment, and is documented as a known mismatch.
    out["app_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y",
                                     errors="coerce")
    out["app_year"] = out["app_date"].dt.year.astype("float32")

    status = df["loan_status"].fillna("").astype(str)
    out["bad"] = np.where(status.isin(C.BAD_STATUSES), 1.0,
                          np.where(status.isin(C.GOOD_STATUSES), 0.0, np.nan)
                          ).astype("float32")
    out["loan_status"] = status
    out["grade"] = df["grade"].fillna("NA").astype(str)

    # economics
    out["int_rate"] = _pct_to_float(df["int_rate"])
    out["term_months"] = pd.to_numeric(
        df["term"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce").astype("float32")
    for c in ["funded_amnt", "total_rec_prncp", "total_rec_int",
              "recoveries", "total_pymnt"]:
        out[c] = df[c].astype("float32")
    out["accepted"] = np.int8(1)

    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


# --------------------------------------------------------------------------
# rejected  (27.6M rows -> one streaming pass, seeded Bernoulli sample)
# --------------------------------------------------------------------------
def load_rejected(cfg: C.Config) -> tuple[pd.DataFrame, int]:
    cache = C.DATA_INTERIM / "rejected.parquet"
    meta = C.DATA_INTERIM / "rejected_meta.json"
    if cache.exists() and meta.exists():
        return pd.read_parquet(cache), json.loads(meta.read_text())["n_total"]

    path = C.DATA_RAW / C.REJECTED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data`.")

    rng = np.random.default_rng(cfg.seed)
    rate = cfg.reject_sample_rate
    chunks, n_total = [], 0
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=_ARROW_BLOCK),
        convert_options=pacsv.ConvertOptions(
            include_columns=REJ_USECOLS,
            column_types={c: pa.string() for c in
                          ("Zip Code", "State", "Employment Length",
                           "Debt-To-Income Ratio", "Application Date")}),
    )
    # Stream: 27.6M rows never exist in memory at once, only the ~3% kept.
    for batch in reader:
        n = batch.num_rows
        n_total += n
        keep = rng.random(n) < rate
        if keep.any():
            chunks.append(batch.filter(pa.array(keep)))
    raw = pa.Table.from_batches(chunks).to_pandas()

    out = pd.DataFrame(index=raw.index)
    out["risk_score"] = _window(
        pd.to_numeric(raw["Risk_Score"], errors="coerce").astype("float32"),
        *SCORE_RANGE)
    out["dti"] = _window(_pct_to_float(raw["Debt-To-Income Ratio"]), *DTI_RANGE)
    out["emp_length_yrs"] = _emp_to_years(raw["Employment Length"])
    out["amount_requested"] = _window(
        pd.to_numeric(raw["Amount Requested"], errors="coerce").astype("float32"),
        *AMT_RANGE)
    out["state"] = raw["State"].fillna("XX").astype(str)
    out["zip3"] = _zip3(raw["Zip Code"]).astype(str)
    out["app_date"] = pd.to_datetime(raw["Application Date"], errors="coerce")
    out["app_year"] = out["app_date"].dt.year.astype("float32")
    out["bad"] = np.float32(np.nan)          # unobservable, by construction
    out["accepted"] = np.int8(0)

    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    meta.write_text(json.dumps(
        {"n_total": int(n_total), "n_sampled": int(len(out)),
         "rate": rate, "seed": cfg.seed}, indent=2))
    return out, n_total


# --------------------------------------------------------------------------
@dataclass
class Panel:
    """The modelling panel plus every count needed to audit the sampling."""
    accepted: pd.DataFrame          # terminal-status accepted loans (sampled)
    rejected: pd.DataFrame          # sampled rejected applications
    n_accepted_total: int           # before status filter / sampling
    n_accepted_terminal: int
    n_rejected_total: int
    w_accepted: float               # design weight (population / sample)
    w_rejected: float
    indeterminate_frac: float
    coverage: pd.DataFrame          # per-field missingness on both sides
    n_rejected_sampled_raw: int = 0     # before the scored-decline filter
    unscored_reject_frac: float = 0.0   # share dropped by that filter
    unscored_by_year: pd.DataFrame | None = None

    @property
    def population_accept_rate(self) -> float:
        """Against ALL declines - the true platform-level approval rate."""
        a = self.n_accepted_total
        return a / (a + self.n_rejected_total)

    @property
    def modelled_accept_rate(self) -> float:
        """Against SCORED declines - the population the models actually see."""
        wa = self.w_accepted * len(self.accepted)
        wr = self.w_rejected * len(self.rejected)
        return wa / (wa + wr)


def _coverage(acc: pd.DataFrame, rej: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in C.COMMON_FEATURES + C.COMMON_CATEGORICAL + ["zip3", "app_year"]:
        rows.append({
            "field": f,
            "accepted_missing_pct": round(100.0 * acc[f].isna().mean(), 3),
            "rejected_missing_pct": round(100.0 * rej[f].isna().mean(), 3),
        })
    return pd.DataFrame(rows)


def load_panel(cfg: C.Config) -> Panel:
    acc_all = load_accepted(cfg)
    rej, n_rej_total = load_rejected(cfg)

    n_acc_total = len(acc_all)
    acc_term = acc_all[acc_all["bad"].notna()].copy()
    n_acc_term = len(acc_term)
    indeterminate = 1.0 - n_acc_term / max(n_acc_total, 1)

    rng = np.random.default_rng(cfg.seed + 1)
    if n_acc_term > cfg.max_accepted:
        idx = np.sort(rng.choice(n_acc_term, size=cfg.max_accepted,
                                 replace=False))
        acc = acc_term.iloc[idx].reset_index(drop=True)
    else:
        acc = acc_term.reset_index(drop=True)

    # Two thirds of declines carry no bureau score. An accepts-only scorecard
    # never sees a missing score, so it has no bin for one and would rate those
    # applicants as average - which is exactly backwards. Drop them, and record
    # what was dropped so the README can be honest about the coverage lost.
    n_rej_raw = len(rej)
    by_year = (rej.assign(_unscored=rej["risk_score"].isna())
               .groupby("app_year", as_index=False)
               .agg(n=("_unscored", "size"),
                    unscored_pct=("_unscored", lambda s: 100.0 * s.mean())))
    unscored_frac = float(rej["risk_score"].isna().mean())
    # Coverage is measured BEFORE the scored-decline filter. Measuring it after
    # would report risk_score as 0% missing on declines, which is true of the
    # modelling sample and deeply misleading about the source data.
    coverage = _coverage(acc, rej)
    if cfg.require_reject_score:
        rej = rej[rej["risk_score"].notna()].reset_index(drop=True)

    # Design weights lift each sample back to its population. The propensity
    # model needs these or it estimates a wildly wrong acceptance rate.
    # After the filter the sample represents SCORED declines only, and each
    # sampled row still stands for 1 / sample_rate of them.
    w_acc = n_acc_total / len(acc)
    w_rej = n_rej_total / max(n_rej_raw, 1)

    return Panel(
        accepted=acc, rejected=rej.reset_index(drop=True),
        n_accepted_total=n_acc_total, n_accepted_terminal=n_acc_term,
        n_rejected_total=n_rej_total, w_accepted=w_acc, w_rejected=w_rej,
        indeterminate_frac=indeterminate, coverage=coverage,
        n_rejected_sampled_raw=n_rej_raw, unscored_reject_frac=unscored_frac,
        unscored_by_year=by_year,
    )


def stack_for_propensity(panel: Panel) -> pd.DataFrame:
    """Accepted + rejected on the common schema, with design weights."""
    cols = (C.COMMON_FEATURES + C.COMMON_CATEGORICAL
            + ["zip3", "app_year", "accepted", "bad"])
    a = panel.accepted[cols].copy()
    a["design_w"] = panel.w_accepted
    r = panel.rejected[cols].copy()
    r["design_w"] = panel.w_rejected
    return pd.concat([a, r], ignore_index=True)
