"""Leakage guards. No dataset download required."""

from __future__ import annotations

import numpy as np
import pytest

from src.graphs.base import EdgeList, assert_no_test_nodes
from src.graphs.random_control import rewire


def _toy_edges(n: int = 10) -> EdgeList:
    src = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    return EdgeList(src, dst, n_nodes=n, meta={"variant": "toy"})


def test_assert_no_test_nodes_passes_on_clean_graph():
    edges = _toy_edges()
    test_idx = np.array([8, 9], dtype=np.int64)
    assert_no_test_nodes(edges, test_idx, "toy")


def test_assert_no_test_nodes_raises_when_test_touched():
    edges = _toy_edges()
    test_idx = np.array([0, 9], dtype=np.int64)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_test_nodes(edges, test_idx, "toy")


def test_drop_nodes_removes_all_incident_edges():
    edges = _toy_edges().drop_nodes(np.array([0, 5]))
    remaining = set(edges.src.tolist()) | set(edges.dst.tolist())
    assert 0 not in remaining
    assert 5 not in remaining
    assert_no_test_nodes(edges, np.array([0, 5]))


def test_rewire_does_not_introduce_test_nodes():
    rng = np.random.default_rng(0)
    allowed = np.arange(18)
    src = np.repeat(allowed, 3)
    dst = (src + rng.integers(1, 17, size=len(src))) % 18
    base = EdgeList(src.astype(np.int64), dst.astype(np.int64), n_nodes=20).dedup()
    out = rewire(base, seed=0)
    forbidden = np.array([18, 19], dtype=np.int64)
    assert_no_test_nodes(out, forbidden, "rewired")
    assert out.n_nodes == 20


def test_concat_query_edges_do_not_link_test_to_test():
    clean = EdgeList(np.array([0, 1]), np.array([1, 2]), n_nodes=6)
    query = EdgeList(np.array([5, 5]), np.array([0, 1]), n_nodes=6)
    full = EdgeList.concat(clean, query)
    test_idx = np.array([5])
    test_test = int((np.isin(full.src, test_idx) & np.isin(full.dst, test_idx)).sum())
    assert test_test == 0
