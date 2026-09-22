"""Shared-attribute edges: connect rows that share a normalised attribute value (zip3, employer).

Groups of size <= max_clique are fully connected. Larger groups would create near-complete subgraphs
(e.g. everyone with employer 'teacher'), so each member is connected to `max_degree` random other
members instead. Only `allowed` positions take part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EdgeList


def build_shared_attribute(attrs: pd.DataFrame, columns: list[str], allowed: np.ndarray, n_nodes: int,
                           max_clique: int = 20, max_degree: int = 10, min_group: int = 2, seed: int = 0) -> EdgeList:
    rng = np.random.default_rng(seed)
    allowed = np.asarray(allowed, dtype=np.int64)
    srcs, dsts, types = [], [], []
    stats = {}
    for t, col in enumerate(columns):
        vals = attrs[col].iloc[allowed]
        ok = vals.notna().to_numpy()
        groups = pd.Series(allowed[ok]).groupby(vals[ok].to_numpy())
        n_groups = n_clique = n_sampled = 0
        for _, members in groups:
            m = members.to_numpy()
            if len(m) < min_group:
                continue
            n_groups += 1
            if len(m) <= max_clique:
                n_clique += 1
                a, b = np.triu_indices(len(m), k=1)
                srcs.append(m[a]); dsts.append(m[b])
                types.append(np.full(len(a), t, dtype=np.int16))
            else:
                n_sampled += 1
                for i in m:
                    others = rng.choice(m, size=min(max_degree, len(m) - 1) + 1, replace=False)
                    others = others[others != i][:max_degree]
                    srcs.append(np.full(len(others), i)); dsts.append(others)
                    types.append(np.full(len(others), t, dtype=np.int16))
        stats[col] = {"groups_used": n_groups, "cliques": n_clique, "sampled_groups": n_sampled}
    if not srcs:
        return EdgeList(np.array([], dtype=np.int64), np.array([], dtype=np.int64), n_nodes, np.array([], dtype=np.int16),
                        meta={"builder": "shared_attribute", "columns": columns, "stats": stats})
    edges = EdgeList(np.concatenate(srcs), np.concatenate(dsts), n_nodes, np.concatenate(types),
                     meta={"builder": "shared_attribute", "variant": "clean", "columns": columns, "max_clique": max_clique,
                           "max_degree": max_degree, "stats": stats, "n_allowed": int(len(allowed)), "seed": seed})
    return edges.dedup()


def query_shared_attribute(attrs: pd.DataFrame, columns: list[str], allowed: np.ndarray, query: np.ndarray, n_nodes: int,
                           max_degree: int = 10, seed: int = 0) -> EdgeList:
    """query -> allowed edges: each query row connects to up to `max_degree` allowed rows sharing each attribute.
    Query rows never connect to each other. Used to attach test rows at scoring time."""
    rng = np.random.default_rng(seed)
    allowed = np.asarray(allowed, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    srcs, dsts, types = [], [], []
    for t, col in enumerate(columns):
        a_vals = attrs[col].iloc[allowed]
        ok = a_vals.notna().to_numpy()
        members = pd.Series(allowed[ok]).groupby(a_vals[ok].to_numpy()).apply(lambda s: s.to_numpy())
        q_vals = attrs[col].iloc[query]
        for q, v in zip(query, q_vals.to_numpy()):
            if pd.isna(v) or v not in members.index:
                continue
            m = members.loc[v]
            pick = m if len(m) <= max_degree else rng.choice(m, max_degree, replace=False)
            srcs.append(np.full(len(pick), q)); dsts.append(pick)
            types.append(np.full(len(pick), t, dtype=np.int16))
    if not srcs:
        return EdgeList(np.array([], dtype=np.int64), np.array([], dtype=np.int64), n_nodes, np.array([], dtype=np.int16),
                        meta={"builder": "shared_attribute", "variant": "query", "columns": columns})
    e = EdgeList(np.concatenate(srcs), np.concatenate(dsts), n_nodes, np.concatenate(types),
                 meta={"builder": "shared_attribute", "variant": "query", "columns": columns, "max_degree": max_degree,
                       "n_query": len(query), "seed": seed})
    return e.dedup()
