"""Degree-preserving random rewiring: the control arm for every synthetic graph.

If a teacher trained on this graph gives the student the same lift as the kNN teacher, the lift
comes from soft-label distillation, not from structure.

Method: keep every source endpoint, permute the destination endpoints. This preserves each node's
out-degree exactly and the in-degree multiset exactly, and destroys the feature-space meaning of edges.
Self-loops and duplicates created by the shuffle are dropped, so the edge count may fall slightly.
"""

from __future__ import annotations

import numpy as np

from .base import EdgeList


def rewire(edges: EdgeList, seed: int = 0) -> EdgeList:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(edges.n_edges)
    out = EdgeList(edges.src.copy(), edges.dst[perm], edges.n_nodes,
                   None if edges.edge_type is None else edges.edge_type[perm],
                   None if edges.edge_time is None else edges.edge_time[perm],
                   {**edges.meta, "builder": "random_control", "rewired_from": edges.meta.get("builder", "?"), "seed": seed})
    out = out.dedup()
    out.meta["n_edges_before_dedup"] = int(edges.n_edges)
    return out
