"""Translate ranking quality into decisions a credit officer recognises.

Convention: p is probability of default. Approving the `approval` fraction means approving the
lowest-p rows. Bad rate is the default rate among approved rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _order(p):
    return np.argsort(np.asarray(p, dtype=float).ravel())  # safest first


def bad_rate_at_approval(y, p, approval: float) -> float:
    y = np.asarray(y, dtype=float).ravel()
    k = max(1, int(round(approval * len(y))))
    return float(y[_order(p)[:k]].mean())


def approval_at_bad_rate(y, p, target_bad_rate: float) -> float:
    """Largest approval fraction whose approved-set default rate is <= target_bad_rate."""
    y = np.asarray(y, dtype=float).ravel()
    ys = y[_order(p)]
    cum_bad = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(cum_bad <= target_bad_rate)
    if len(ok) == 0:
        return 0.0
    return float((ok.max() + 1) / len(ys))


def matched_bad_rate_table(y, p_base, candidates: dict[str, np.ndarray], base_approval: float = 0.8) -> pd.DataFrame:
    """Fix the baseline's bad rate at `base_approval`; report each candidate's approval at that bad rate,
    and each candidate's bad rate at the baseline approval."""
    target = bad_rate_at_approval(y, p_base, base_approval)
    rows = [{"model": "baseline", "approval_at_matched_bad_rate": base_approval, "bad_rate_at_base_approval": target,
             "approval_delta_pp": 0.0, "bad_rate_delta_pp": 0.0}]
    for name, p in candidates.items():
        appr = approval_at_bad_rate(y, p, target)
        br = bad_rate_at_approval(y, p, base_approval)
        rows.append({"model": name, "approval_at_matched_bad_rate": appr, "bad_rate_at_base_approval": br,
                     "approval_delta_pp": 100 * (appr - base_approval), "bad_rate_delta_pp": 100 * (br - target)})
    out = pd.DataFrame(rows)
    out.attrs["target_bad_rate"] = target
    return out


def swap_set(y, p_old, p_new, approval: float = 0.8) -> dict:
    """Who does the new model approve that the old model declined (swap-in) and vice versa (swap-out),
    and what were their default rates?"""
    y = np.asarray(y, dtype=float).ravel()
    k = max(1, int(round(approval * len(y))))
    old_appr = np.zeros(len(y), bool)
    old_appr[_order(p_old)[:k]] = True
    new_appr = np.zeros(len(y), bool)
    new_appr[_order(p_new)[:k]] = True
    swap_in = new_appr & ~old_appr
    swap_out = old_appr & ~new_appr
    return {
        "approval": approval,
        "n_swap_in": int(swap_in.sum()),
        "n_swap_out": int(swap_out.sum()),
        "swap_in_bad_rate": float(y[swap_in].mean()) if swap_in.any() else float("nan"),
        "swap_out_bad_rate": float(y[swap_out].mean()) if swap_out.any() else float("nan"),
        "both_approved_bad_rate": float(y[old_appr & new_appr].mean()) if (old_appr & new_appr).any() else float("nan"),
        "net_defaults_avoided": int(y[swap_out].sum() - y[swap_in].sum()),
    }
