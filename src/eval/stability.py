"""Does the model keep working over time? Cohort metrics and population stability index."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .metrics import METRIC_FNS


def cohort_metrics(y, p, cohort, metrics=("auc", "pr_auc", "capture_10"), min_n: int = 200) -> pd.DataFrame:
    """One row per cohort value (e.g. issue year or quarter)."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float), "cohort": np.asarray(cohort)})
    rows = []
    for c, g in df.groupby("cohort", sort=True):
        if len(g) < min_n:
            continue
        row = {"cohort": c, "n": int(len(g)), "base_rate": float(g["y"].mean())}
        for m in metrics:
            fn: Callable = METRIC_FNS[m]
            row[m] = fn(g["y"].to_numpy(), g["p"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population stability index between two score distributions, bins from `expected` quantiles.

    < 0.10 stable, 0.10-0.25 watch, > 0.25 shifted.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_cnt = np.histogram(expected, bins=edges)[0].astype(float)
    a_cnt = np.histogram(actual, bins=edges)[0].astype(float)
    e_pct = np.clip(e_cnt / e_cnt.sum(), eps, None)
    a_pct = np.clip(a_cnt / a_cnt.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_label(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "watch"
    return "shifted"


def year_cohort(time) -> np.ndarray:
    return pd.to_datetime(pd.Series(time)).dt.year.to_numpy()


def quarter_cohort(time) -> np.ndarray:
    t = pd.to_datetime(pd.Series(time))
    return (t.dt.year.astype(str) + "Q" + t.dt.quarter.astype(str)).to_numpy()
