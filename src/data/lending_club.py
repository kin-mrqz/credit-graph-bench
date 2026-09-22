"""Lending Club accepted loans 2007-2018Q4 (Kaggle mirror).

Finished loans only: 'Charged Off' = 1, 'Fully Paid' = 0; everything else (Current, Late, ...) is dropped.
Only origination-time columns are kept (configs/lending_club.yaml: feature_cols and denylist).
issue_d gives a real time axis: train <= 2015, val 2016, test 2017-2018.
zip_code and emp_title are kept as `attrs` for shared-attribute edges, not as features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..paths import raw_dir
from .base import Bundle, encode_categoricals, subsample

RAW_CANDIDATES = ["accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv.gz"]
LABEL_MAP = {"Fully Paid": 0.0, "Charged Off": 1.0,
             "Does not meet the credit policy. Status:Fully Paid": 0.0,
             "Does not meet the credit policy. Status:Charged Off": 1.0}
ATTR_COLS = ["zip_code", "emp_title", "addr_state"]


def raw_present(cfg: dict) -> dict[str, bool]:
    d = raw_dir("lending_club")
    return {f: (d / f).exists() for f in RAW_CANDIDATES}


def _raw_path():
    d = raw_dir("lending_club")
    for f in RAW_CANDIDATES:
        if (d / f).exists():
            return d / f
    raise FileNotFoundError(f"No Lending Club file under {d}. Run scripts/fetch_data.sh lending_club.")


def _parse_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def _emp_length(s: pd.Series) -> pd.Series:
    m = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
    def f(v):
        if pd.isna(v):
            return np.nan
        if v in m:
            return m[v]
        d = re.findall(r"\d+", str(v))
        return float(d[0]) if d else np.nan
    return s.map(f).astype(float)


def _normalise_title(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.lower().str.strip()
    t = t.str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True)
    t = t.replace({"": pd.NA, "na": pd.NA, "none": pd.NA, "self": "self employed", "selfemployed": "self employed"})
    return t


def build(cfg: dict, quick: bool = False) -> Bundle:
    feature_cols = list(cfg["feature_cols"])
    usecols = sorted(set(feature_cols + ["id", "loan_status", "issue_d"] + ATTR_COLS))
    df = pd.read_csv(_raw_path(), usecols=usecols, low_memory=False)

    df["default"] = df["loan_status"].map(LABEL_MAP)
    df = df[df["default"].notna()]                      # finished loans only
    df["issue_d"] = _parse_month(df["issue_d"])
    df = df[df["issue_d"].notna()]
    if quick:
        df = subsample(df, cfg["quick"]["n_rows"], seed=0, stratify=df["default"])
    df = df.sort_values("issue_d", kind="stable").reset_index(drop=True)

    y = df["default"].astype(float)
    ids = df["id"]
    time = df["issue_d"]
    attrs = pd.DataFrame({
        "zip_code": df["zip_code"].astype("string").str.slice(0, 3),
        "emp_title": _normalise_title(df["emp_title"]),
        "addr_state": df["addr_state"].astype("string"),
    })

    X = df[[c for c in feature_cols if c in df.columns]].copy()
    if "term" in X.columns:
        X["term"] = X["term"].astype("string").str.extract(r"(\d+)")[0].astype(float)
    if "int_rate" in X.columns and X["int_rate"].dtype == object:
        X["int_rate"] = X["int_rate"].astype("string").str.rstrip("%").astype(float)
    if "revol_util" in X.columns and X["revol_util"].dtype == object:
        X["revol_util"] = X["revol_util"].astype("string").str.rstrip("%").astype(float)
    if "emp_length" in X.columns:
        X["emp_length"] = _emp_length(X["emp_length"])
    if "earliest_cr_line" in X.columns:
        ecl = _parse_month(X["earliest_cr_line"])
        X["credit_history_months"] = ((time - ecl).dt.days / 30.44).astype(float)
        X = X.drop(columns=["earliest_cr_line"])
    if "grade" in X.columns:
        X["grade"] = X["grade"].map({g: i for i, g in enumerate("ABCDEFG")}).astype(float)
    if "sub_grade" in X.columns:
        X["sub_grade"] = X["sub_grade"].map(lambda v: (ord(str(v)[0]) - 65) * 5 + int(str(v)[1]) if isinstance(v, str) and len(v) == 2 else np.nan).astype(float)

    cats = [c for c in cfg.get("categorical", []) if c in X.columns and X[c].dtype == object]
    X, cats = encode_categoricals(X, cats)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return Bundle(
        name="lending_club", X=X, y=y, ids=ids, time=time, subgroups={}, attrs=attrs, categorical=cats,
        meta={"denylist_applied": [c for c in cfg["denylist"]], "source": cfg["source"], "quick": quick,
              "label_rule": "Charged Off=1, Fully Paid=0, in-progress loans dropped",
              "time_split": {"train_end": cfg["split"]["train_end"], "val_end": cfg["split"]["val_end"]}},
    )


def instructions() -> str:
    return "Kaggle dataset wordsforthewise/lending-club, file accepted_2007_to_2018Q4.csv(.gz). scripts/fetch_data.sh lending_club."
