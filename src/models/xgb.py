"""XGBoost with early stopping on validation AUC. Accepts soft targets in [0, 1] for distillation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..device import xgb_device

DEFAULTS = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist"}


def _dmatrix(X, y=None, weight=None) -> xgb.DMatrix:
    if isinstance(X, pd.DataFrame):
        return xgb.DMatrix(X, label=y, weight=weight, missing=np.nan, feature_names=[str(c) for c in X.columns])
    return xgb.DMatrix(np.asarray(X, dtype=np.float32), label=y, weight=weight, missing=np.nan)


def fit_xgb(X_tr, y_tr, X_val, y_val, params: dict, seed: int = 0, weight_tr=None, verbose: bool = False) -> xgb.Booster:
    p = {**DEFAULTS, **params}
    num_boost_round = int(p.pop("num_boost_round", 1000))
    early = int(p.pop("early_stopping_rounds", 100))
    p["seed"] = int(seed)
    p["device"] = xgb_device()
    dtr = _dmatrix(X_tr, np.asarray(y_tr, dtype=np.float32), weight_tr)
    dval = _dmatrix(X_val, np.asarray(y_val, dtype=np.float32))
    booster = xgb.train(p, dtr, num_boost_round=num_boost_round, evals=[(dval, "val")],
                        early_stopping_rounds=early, verbose_eval=100 if verbose else False)
    return booster


def predict_xgb(booster: xgb.Booster, X) -> np.ndarray:
    best = getattr(booster, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return booster.predict(_dmatrix(X), iteration_range=rng).astype(np.float32)


def importance(booster: xgb.Booster, kind: str = "gain") -> pd.Series:
    s = pd.Series(booster.get_score(importance_type=kind), dtype=float)
    return s.sort_values(ascending=False)


def contributions(booster: xgb.Booster, X, max_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Per-row SHAP-style contributions (pred_contribs) on a sample. Last column is the bias."""
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    c = booster.predict(_dmatrix(X), pred_contribs=True)
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(c.shape[1] - 1)]
    return pd.DataFrame(c, columns=cols + ["bias"])


def save_xgb(booster: xgb.Booster, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return path


def load_xgb(path: Path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    return b


def summary(booster: xgb.Booster) -> dict:
    return {"best_iteration": int(getattr(booster, "best_iteration", -1)),
            "best_score": float(getattr(booster, "best_score", float("nan"))),
            "n_features": int(booster.num_features())}
