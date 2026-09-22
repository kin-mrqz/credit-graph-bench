"""Paired bootstrap over test rows.

The main question in every notebook is "is model B better than baseline A on the same rows?".
Resampling rows with replacement and recomputing metric(B) - metric(A) on each resample gives a
confidence interval for the difference. An interval that includes zero is reported as
'no evidence of lift'.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .metrics import METRIC_FNS


def bootstrap_ci(y, p, metric: Callable | str = "auc", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(y[idx], p[idx])
    vals = vals[~np.isnan(vals)]
    return {
        "point": float(fn(y, p)),
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }


def paired_bootstrap(y, p_base, p_new, metric: Callable | str = "auc", n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """CI for metric(p_new) - metric(p_base) on identical resampled rows."""
    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric
    y = np.asarray(y).ravel()
    p_base = np.asarray(p_base).ravel()
    p_new = np.asarray(p_new).ravel()
    assert len(y) == len(p_base) == len(p_new), "paired bootstrap needs aligned predictions"
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(y[idx], p_new[idx]) - fn(y[idx], p_base[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "metric"),
        "base": float(fn(y, p_base)),
        "new": float(fn(y, p_new)),
        "diff": float(fn(y, p_new) - fn(y, p_base)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_new_worse": float(np.mean(diffs <= 0)),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot": int(len(diffs)),
    }


def compare_many(y, p_base, candidates: dict[str, np.ndarray], metrics=("auc", "pr_auc", "capture_10"),
                 n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Paired bootstrap of every candidate against the baseline for several metrics. Returns rows."""
    rows = []
    for name, p in candidates.items():
        for m in metrics:
            r = paired_bootstrap(y, p_base, p, m, n_boot=n_boot, seed=seed)
            r["model"] = name
            rows.append(r)
    return rows
