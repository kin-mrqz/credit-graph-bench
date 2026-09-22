"""Ranking, calibration, and capture metrics for binary default prediction.

All functions take y (0/1) and p (predicted probability of default) as 1-d arrays.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _clean(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    ok = ~(np.isnan(y) | np.isnan(p))
    return y[ok].astype(int), p[ok]


def auc(y, p) -> float:
    y, p = _clean(y, p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    a = auc(y, p)
    return float(2 * a - 1) if not np.isnan(a) else a


def pr_auc(y, p) -> float:
    y, p = _clean(y, p)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative distributions of scores for bads vs goods."""
    y, p = _clean(y, p)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-p)
    ys = y[order]
    tpr = np.cumsum(ys) / ys.sum()
    fpr = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(tpr - fpr)))


def capture_at(y, p, frac: float = 0.10) -> float:
    """Share of all defaults that fall in the riskiest `frac` of scores."""
    y, p = _clean(y, p)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * len(y))))
    top = np.argsort(-p)[:k]
    return float(y[top].sum() / y.sum())


def bad_rate_at(y, p, frac: float = 0.10) -> float:
    """Default rate inside the riskiest `frac` of scores."""
    y, p = _clean(y, p)
    k = max(1, int(np.ceil(frac * len(y))))
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def brier(y, p) -> float:
    y, p = _clean(y, p)
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = _clean(y, p)
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def log_loss(y, p, eps: float = 1e-7) -> float:
    y, p = _clean(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


METRIC_FNS = {
    "auc": auc,
    "gini": gini,
    "pr_auc": pr_auc,
    "ks": ks,
    "capture_10": lambda y, p: capture_at(y, p, 0.10),
    "capture_5": lambda y, p: capture_at(y, p, 0.05),
    "bad_rate_top10": lambda y, p: bad_rate_at(y, p, 0.10),
    "brier": brier,
    "ece": ece,
    "log_loss": log_loss,
}

# Higher is better for these; lower is better for brier, ece, log_loss
HIGHER_IS_BETTER = {"auc", "gini", "pr_auc", "ks", "capture_10", "capture_5", "bad_rate_top10"}


def all_metrics(y, p, prefix: str = "") -> dict[str, float]:
    y, p = _clean(y, p)
    out = {f"{prefix}{k}": fn(y, p) for k, fn in METRIC_FNS.items()}
    out[f"{prefix}n"] = int(len(y))
    out[f"{prefix}positives"] = int(y.sum())
    out[f"{prefix}base_rate"] = float(y.mean()) if len(y) else float("nan")
    return out
