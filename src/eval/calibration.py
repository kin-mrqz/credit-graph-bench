"""Probability calibration fitted on validation, applied to test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt", "none"}:
            raise ValueError(method)
        self.method = method
        self._m = None

    def fit(self, p_val, y_val) -> "Calibrator":
        p_val = np.asarray(p_val, dtype=float).ravel()
        y_val = np.asarray(y_val, dtype=int).ravel()
        if self.method == "isotonic":
            self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val, y_val)
        elif self.method == "platt":
            z = _logit(p_val).reshape(-1, 1)
            self._m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_val)
        return self

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float).ravel()
        if self.method == "none" or self._m is None:
            return p
        if self.method == "isotonic":
            return self._m.predict(p)
        return self._m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Per-bin predicted vs observed default rate. strategy: 'quantile' (equal count) or 'uniform'."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "pred_mean": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    return pd.DataFrame(rows)
