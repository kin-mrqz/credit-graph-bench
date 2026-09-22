"""UCI 350: Default of Credit Card Clients (Taiwan, 2005).

30,000 rows, 23 features, label 'default payment next month' (~22% positive).
No ids that link rows, no dates, no shared attributes. The only graphs here are synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import raw_dir
from .base import Bundle, apply_denylist, encode_categoricals, subsample

RAW_NAME = "default of credit card clients.xls"


def raw_present(cfg: dict) -> dict[str, bool]:
    return {RAW_NAME: (raw_dir("taiwan") / RAW_NAME).exists()}


def _read_raw() -> pd.DataFrame:
    path = raw_dir("taiwan") / RAW_NAME
    if path.exists():
        df = pd.read_excel(path, header=1)   # first row is X1..X23 codes, second row is names
        return df
    # Fallback: fetch from the UCI repository client
    try:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=350)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df.insert(0, "ID", np.arange(1, len(df) + 1))
        df = df.rename(columns={"Y": "default payment next month", "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION",
                                "X4": "MARRIAGE", "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4",
                                "X10": "PAY_5", "X11": "PAY_6", "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
                                "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6", "X18": "PAY_AMT1",
                                "X19": "PAY_AMT2", "X20": "PAY_AMT3", "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6"})
        return df
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(f"Taiwan raw file missing at {path} and ucimlrepo fetch failed: {exc}") from exc


def build(cfg: dict, quick: bool = False) -> Bundle:
    df = _read_raw()
    label = cfg["label"]
    if label not in df.columns:
        # some exports name it differently
        cand = [c for c in df.columns if "default" in str(c).lower()]
        if not cand:
            raise KeyError(f"label column '{label}' not found; columns: {list(df.columns)[:10]}...")
        df = df.rename(columns={cand[0]: label})
    if quick:
        df = subsample(df, cfg["quick"]["n_rows"], seed=0, stratify=df[label])
    df = df.reset_index(drop=True)

    y = df[label].astype(float)
    ids = df[cfg["id_col"]]
    subgroups = {}
    for k, col in cfg.get("subgroups", {}).items():
        if col in df.columns:
            s = df[col]
            subgroups[k] = pd.cut(s, bins=[0, 30, 40, 50, 120], labels=["<30", "30-39", "40-49", "50+"]).astype(str) if k == "age" else s.astype(str)

    X, dropped = apply_denylist(df, cfg["denylist"])
    X, cats = encode_categoricals(X, [c for c in cfg.get("categorical", []) if c in X.columns])
    X = X.astype(np.float32)
    return Bundle(
        name="taiwan", X=X, y=y, ids=ids, time=None, subgroups=subgroups, attrs=None, categorical=cats,
        meta={"denylist_applied": dropped, "source": cfg["source"], "quick": quick},
    )


def instructions() -> str:
    return f"Place '{RAW_NAME}' under {raw_dir('taiwan')} (scripts/fetch_data.sh taiwan) or let the loader use ucimlrepo id=350."
