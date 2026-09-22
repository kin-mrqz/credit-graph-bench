"""Logistic regression baseline: the shape of a traditional scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_lr(columns: list[str], categorical: list[str], C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> Pipeline:
    cat = [c for c in categorical if c in columns]
    num = [c for c in columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True))]), cat),
    ], remainder="drop", sparse_threshold=0.3)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_lr(X_tr: pd.DataFrame, y_tr, categorical: list[str], C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> Pipeline:
    m = make_lr(list(X_tr.columns), categorical, C=C, max_iter=max_iter, seed=seed)
    m.fit(X_tr, np.asarray(y_tr, dtype=int))
    return m


def predict_lr(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


def coefficients(model: Pipeline) -> pd.Series:
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_.ravel()
    return pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)
