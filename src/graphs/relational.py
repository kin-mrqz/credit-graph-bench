"""Reduced heterogeneous graph for Home Credit: customer, bureau record, previous application.

Saved as one npz with:
    x_customer (Nc, Fc)  from the Bundle (imputed, standardised)
    x_bureau   (Nb, Fb)  bureau node features
    x_prev_app (Np, Fp)  previous-application node features
    ei_customer__has_bureau   (2, Eb)   customer -> bureau
    ei_customer__has_prev_app (2, Ep)   customer -> prev_app
Reverse relations are added at load time by src.models.rgcn.

`allowed_customers` restricts child rows to those customers (clean-room teacher graph).
Because edges only go customer -> own child rows, a test customer's edges never touch another
customer; still, the clean variant drops test customers' children so the teacher never sees them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

NODE_TYPES = ["customer", "bureau", "prev_app"]
RELATIONS = [("customer", "has_bureau", "bureau"), ("customer", "has_prev_app", "prev_app")]


def _prep(df: pd.DataFrame, fit_rows: np.ndarray | None = None) -> np.ndarray:
    X = df.to_numpy(dtype=np.float32)
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    fit = X if fit_rows is None else X[fit_rows]
    imp.fit(fit)
    sc.fit(imp.transform(fit))
    return np.nan_to_num(sc.transform(imp.transform(X)).astype(np.float32))


def build_hetero(X_customer: pd.DataFrame, tables: dict[str, pd.DataFrame], allowed_customers: np.ndarray | None = None,
                 fit_rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n_c = len(X_customer)
    out = {"x_customer": _prep(X_customer, fit_rows)}
    allowed = None
    if allowed_customers is not None:
        allowed = np.zeros(n_c, dtype=bool)
        allowed[np.asarray(allowed_customers, dtype=np.int64)] = True
    meta = {"n_customer": int(n_c), "allowed_customers": None if allowed is None else int(allowed.sum())}
    for key, (_, rel, dst_type) in zip(["bureau", "prev_app"], RELATIONS):
        df = tables[key]
        cust = df["customer_pos"].to_numpy(dtype=np.int64)
        if allowed is not None:
            keep = allowed[cust]
            df, cust = df[keep], cust[keep]
        feats = df.drop(columns=["customer_pos"])
        out[f"x_{dst_type}"] = _prep(feats) if len(feats) else np.zeros((0, feats.shape[1]), np.float32)
        out[f"ei_customer__{rel}"] = np.vstack([cust, np.arange(len(cust), dtype=np.int64)])
        meta[f"n_{dst_type}"] = int(len(cust))
        meta[f"features_{dst_type}"] = list(feats.columns)
    out["_meta"] = meta
    return out


def save_hetero(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = payload.pop("_meta", {})
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    payload["_meta"] = meta
    return path


def load_hetero(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No hetero graph at {path}. Run notebook 03_build_graphs for home_credit/relational.")
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    meta_path = path.with_suffix(".json")
    payload["_meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return payload


def hetero_stats(payload: dict) -> dict:
    n_c = payload["x_customer"].shape[0]
    stats = {"n_customer": int(n_c)}
    for _, rel, dst in RELATIONS:
        ei = payload[f"ei_customer__{rel}"]
        deg = np.bincount(ei[0], minlength=n_c)
        stats[f"n_{dst}"] = int(ei.shape[1])
        stats[f"{dst}_per_customer_mean"] = float(deg.mean())
        stats[f"{dst}_per_customer_max"] = int(deg.max()) if len(deg) else 0
        stats[f"customers_without_{dst}"] = float((deg == 0).mean())
    return stats
