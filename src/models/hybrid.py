"""Tier 3: GNN embeddings appended to tabular features, scored by XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .xgb import contributions


def attach_embeddings(X: pd.DataFrame, emb: np.ndarray, prefix: str = "emb_") -> pd.DataFrame:
    """Row-align `emb` (same row order as X) and concatenate as float32 columns."""
    assert len(X) == len(emb), f"rows differ: X {len(X)} vs emb {len(emb)}"
    E = pd.DataFrame(np.asarray(emb, dtype=np.float32), columns=[f"{prefix}{i}" for i in range(emb.shape[1])], index=X.index)
    return pd.concat([X.reset_index(drop=True), E.reset_index(drop=True)], axis=1)


def contribution_share(booster, X: pd.DataFrame, prefix: str = "emb_", max_rows: int = 5000) -> dict:
    """Share of total |contribution| that comes from embedding columns vs tabular columns."""
    c = contributions(booster, X, max_rows=max_rows).drop(columns=["bias"]).abs()
    emb_cols = [col for col in c.columns if col.startswith(prefix)]
    tab_cols = [col for col in c.columns if not col.startswith(prefix)]
    tot = c.sum().sum()
    per_col = c.sum().sort_values(ascending=False)
    return {
        "emb_share": float(c[emb_cols].sum().sum() / tot) if tot else float("nan"),
        "tab_share": float(c[tab_cols].sum().sum() / tot) if tot else float("nan"),
        "n_emb_cols": len(emb_cols),
        "top_features": per_col.head(20).to_dict(),
    }
