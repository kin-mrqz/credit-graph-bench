"""Tier 1: hand-built graph features a bank could compute with database queries.

All label-derived features use TRAINING labels only. Validation and test labels are never read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import EdgeList


def graph_features(edges: EdgeList, X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                   feature_cols: list[str] | None = None, max_mean_cols: int = 30) -> pd.DataFrame:
    """Returns a DataFrame indexed by position with columns prefixed ``g_``."""
    n = edges.n_nodes
    A = edges.to_scipy(symmetric=True)          # undirected adjacency, binary
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)

    out = {"g_degree": deg, "g_has_edges": (deg > 0).astype(np.float32), "g_log_degree": np.log1p(deg)}

    # out/in degree on the directed graph if it was directed
    if not edges.meta.get("symmetrized"):
        out["g_out_degree"] = np.bincount(edges.src, minlength=n).astype(np.float32)
        out["g_in_degree"] = np.bincount(edges.dst, minlength=n).astype(np.float32)

    # edge-type counts
    if edges.edge_type is not None:
        for t in np.unique(edges.edge_type):
            m = edges.edge_type == t
            out[f"g_type{int(t)}_count"] = np.bincount(edges.src[m], minlength=n).astype(np.float32) + \
                np.bincount(edges.dst[m], minlength=n).astype(np.float32)

    # 2-hop reach
    out["g_two_hop"] = np.asarray(A @ deg.reshape(-1, 1)).ravel()

    # neighbour label rate from training labels only (self excluded automatically: no self loops)
    y = np.asarray(y, dtype=float)
    lab = train_mask & ~np.isnan(y)
    l_vec = np.where(lab, np.nan_to_num(y), 0.0)
    c_vec = lab.astype(np.float32)
    n_lab_nb = np.asarray(A @ c_vec.reshape(-1, 1)).ravel()
    bad_nb = np.asarray(A @ l_vec.reshape(-1, 1)).ravel()
    out["g_nb_train_labelled"] = n_lab_nb
    out["g_nb_bad_count"] = bad_nb
    out["g_nb_bad_rate"] = np.where(n_lab_nb > 0, bad_nb / np.where(n_lab_nb > 0, n_lab_nb, 1), -1.0)

    # neighbour feature means (first max_mean_cols numeric columns to bound width)
    Xf = np.nan_to_num(np.asarray(X, dtype=np.float32))
    cols = list(range(min(Xf.shape[1], max_mean_cols)))
    nb_mean = (A @ Xf[:, cols]) / deg_safe.reshape(-1, 1)
    names = feature_cols if feature_cols is not None else [f"x{i}" for i in range(Xf.shape[1])]
    for j, c in enumerate(cols):
        out[f"g_nbmean_{names[c]}"] = nb_mean[:, j]
        out[f"g_nbdiff_{names[c]}"] = Xf[:, c] - nb_mean[:, j]

    df = pd.DataFrame(out).astype(np.float32)
    df.index.name = "position"
    return df


def pagerank(edges: EdgeList, alpha: float = 0.85, iters: int = 50) -> np.ndarray:
    A = edges.to_scipy(symmetric=True)
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.where(deg > 0, deg, 1.0)) @ A
    n = edges.n_nodes
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - alpha) / n + alpha * (P.T @ r)
    return r.astype(np.float32)
