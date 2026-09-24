"""Home Credit Default Risk (Kaggle, 2018).

application_train.csv: 307,511 loans, TARGET ~8% positive. Six child tables keyed by SK_ID_CURR
(bureau, previous_application, installments_payments, POS_CASH_balance, credit_card_balance) and
SK_ID_BUREAU (bureau_balance).

Two products:
    build()                  tabular Bundle: application columns + per-customer aggregates of every child table
    load_relational_tables() bureau and previous_application node tables for the heterogeneous graph
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import raw_dir
from .base import Bundle, apply_denylist, encode_categoricals, is_text_like, subsample

FILES = {
    "app": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "prev": "previous_application.csv",
    "inst": "installments_payments.csv",
    "pos": "POS_CASH_balance.csv",
    "cc": "credit_card_balance.csv",
}


def raw_present(cfg: dict) -> dict[str, bool]:
    d = raw_dir("home_credit")
    return {f: (d / f).exists() for f in FILES.values()}


def _read(name: str, usecols=None, nrows=None) -> pd.DataFrame:
    path = raw_dir("home_credit") / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/fetch_data.sh home_credit (needs Kaggle credentials).")
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)


def _numeric_aggregates(df: pd.DataFrame, key: str, prefix: str, stats=("mean", "max", "min", "sum")) -> pd.DataFrame:
    """Per-key aggregates of numeric columns plus a row count. Object columns become frequency of the top level."""
    num_cols = [c for c in df.select_dtypes("number").columns if not c.startswith("SK_ID")]
    g = df.groupby(key)
    out = g[num_cols].agg(list(stats))
    out.columns = [f"{prefix}_{a}_{b}" for a, b in out.columns]
    out[f"{prefix}_count"] = g.size()
    obj_cols = [c for c in df.columns if is_text_like(df[c])][:6]  # cap width
    for c in obj_cols:
        top = df[c].mode(dropna=True)
        if len(top):
            out[f"{prefix}_{c}_is_{str(top.iloc[0])[:12]}"] = (df[c] == top.iloc[0]).groupby(df[key]).mean()
    return out


def build(cfg: dict, quick: bool = False) -> Bundle:
    app = _read("app")
    label, id_col = cfg["label"], cfg["id_col"]
    if quick:
        app = subsample(app, cfg["quick"]["n_rows"], seed=0, stratify=app[label])
    app = app.reset_index(drop=True)
    keep_ids = set(app[id_col].tolist())

    # ---- bureau + bureau_balance (bureau_balance rolled up to bureau first)
    bureau = _read("bureau")
    bureau = bureau[bureau[id_col].isin(keep_ids)]
    bb = _read("bureau_balance")
    bb = bb[bb["SK_ID_BUREAU"].isin(set(bureau["SK_ID_BUREAU"]))]
    bb["late"] = bb["STATUS"].isin(list("12345")).astype(np.int8)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_months=("MONTHS_BALANCE", "size"), bb_late_frac=("late", "mean"),
                                           bb_oldest=("MONTHS_BALANCE", "min"))
    bureau = bureau.merge(bb_agg, how="left", left_on="SK_ID_BUREAU", right_index=True)
    agg_bureau = _numeric_aggregates(bureau, id_col, "bureau")

    # ---- previous applications and their payment tables (keyed by SK_ID_CURR directly)
    prev = _read("prev")
    prev = prev[prev[id_col].isin(keep_ids)]
    agg_prev = _numeric_aggregates(prev, id_col, "prev")

    inst = _read("inst")
    inst = inst[inst[id_col].isin(keep_ids)]
    inst["days_late"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["underpaid"] = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]).clip(lower=0)
    agg_inst = _numeric_aggregates(inst, id_col, "inst", stats=("mean", "max", "sum"))

    pos = _read("pos")
    pos = pos[pos[id_col].isin(keep_ids)]
    agg_pos = _numeric_aggregates(pos, id_col, "pos", stats=("mean", "max"))

    cc = _read("cc")
    cc = cc[cc[id_col].isin(keep_ids)]
    agg_cc = _numeric_aggregates(cc, id_col, "cc", stats=("mean", "max"))

    df = app
    for agg in (agg_bureau, agg_prev, agg_inst, agg_pos, agg_cc):
        df = df.merge(agg, how="left", left_on=id_col, right_index=True)
    # counts are 0, not missing, when a customer has no child rows
    for c in [c for c in df.columns if c.endswith("_count")]:
        df[c] = df[c].fillna(0)

    y = df[label].astype(float)
    ids = df[id_col]
    subgroups = {}
    if "CODE_GENDER" in df.columns:
        subgroups["sex"] = df["CODE_GENDER"].astype(str)
    if "DAYS_BIRTH" in df.columns:
        age = (-df["DAYS_BIRTH"] / 365.25)
        subgroups["age"] = pd.cut(age, bins=[0, 30, 40, 50, 120], labels=["<30", "30-39", "40-49", "50+"]).astype(str)

    X, dropped = apply_denylist(df, cfg["denylist"])
    X, cats = encode_categoricals(X)   # object columns -> codes
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return Bundle(
        name="home_credit", X=X, y=y, ids=ids, time=None, subgroups=subgroups, attrs=None, categorical=cats,
        meta={"denylist_applied": dropped, "source": cfg["source"], "quick": quick,
              "child_tables": ["bureau", "bureau_balance", "previous_application", "installments_payments",
                               "POS_CASH_balance", "credit_card_balance"]},
    )


def load_relational_tables(ids: pd.Series, max_children: int = 50) -> dict[str, pd.DataFrame]:
    """Node tables for the reduced heterogeneous graph.

    Returns {'bureau': df, 'prev_app': df}. Each has a 'customer_pos' column (position of the owning
    customer in the Bundle order) and numeric feature columns only. At most `max_children` rows per
    customer are kept per table (most recent first) to bound degree.
    """
    id_col = "SK_ID_CURR"
    pos_of = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    out = {}

    bureau = _read("bureau")
    bureau = bureau[bureau[id_col].isin(pos_of.index)]
    bureau = bureau.sort_values("DAYS_CREDIT", ascending=False).groupby(id_col).head(max_children)
    bureau["customer_pos"] = pos_of.loc[bureau[id_col].to_numpy()].to_numpy()
    feats, _ = encode_categoricals(bureau.drop(columns=[id_col, "SK_ID_BUREAU"]))
    out["bureau"] = feats.replace([np.inf, -np.inf], np.nan).astype(np.float32).reset_index(drop=True)

    prev = _read("prev")
    prev = prev[prev[id_col].isin(pos_of.index)]
    prev = prev.sort_values("DAYS_DECISION", ascending=False).groupby(id_col).head(max_children)
    prev["customer_pos"] = pos_of.loc[prev[id_col].to_numpy()].to_numpy()
    feats, _ = encode_categoricals(prev.drop(columns=[id_col, "SK_ID_PREV"]))
    out["prev_app"] = feats.replace([np.inf, -np.inf], np.nan).astype(np.float32).reset_index(drop=True)
    return out


def instructions() -> str:
    return "Kaggle competition 'home-credit-default-risk'. Accept the rules, then scripts/fetch_data.sh home_credit."
