"""Graph builders. All operate on Bundle positions (0..n-1) and produce an :class:`EdgeList`.

File naming under ``paths.graphs_dir(dataset)``:

    native:           edges_native.npz (full, real edges incl. test nodes)   edges_native_clean.npz (test nodes dropped)
    knn:              edges_knn_k{k}.npz (clean)                            edges_knn_k{k}_query.npz (test -> allowed)
    random_control:   edges_random_control_k{k}.npz                          edges_random_control_k{k}_query.npz
    shared_attribute: edges_shared_attribute.npz (clean)                     edges_shared_attribute_query.npz
    relational:       hetero_relational.npz (full)                           hetero_relational_clean.npz

``variant="clean"`` graphs contain no test node. ``variant="full"`` graphs attach test nodes for scoring
(clean + query edges, or the real full graph) without ever using a test label.
"""

from __future__ import annotations

from pathlib import Path

from ..paths import graphs_dir
from .base import EdgeList, assert_no_test_nodes  # noqa: F401

HOMOGENEOUS = {"native", "knn", "random_control", "shared_attribute"}


def default_k(cfg: dict) -> int | None:
    return cfg.get("graph", {}).get("knn", {}).get("default_k")


def edge_paths(dataset: str, graph_builder: str, cfg: dict, quick: bool = False, k: int | None = None) -> dict[str, Path]:
    """Paths for the clean and query files of a homogeneous builder (query may not exist for native)."""
    d = graphs_dir(dataset, quick)
    k = k or default_k(cfg)
    if graph_builder == "native":
        return {"full": d / "edges_native.npz", "clean": d / "edges_native_clean.npz"}
    if graph_builder in {"knn", "random_control"}:
        return {"clean": d / f"edges_{graph_builder}_k{k}.npz", "query": d / f"edges_{graph_builder}_k{k}_query.npz"}
    if graph_builder == "shared_attribute":
        return {"clean": d / "edges_shared_attribute.npz", "query": d / "edges_shared_attribute_query.npz"}
    if graph_builder == "relational":
        return {"full": d / "hetero_relational.npz", "clean": d / "hetero_relational_clean.npz"}
    raise ValueError(f"unknown graph_builder '{graph_builder}'")


def load_edges(dataset: str, graph_builder: str, cfg: dict, quick: bool = False, k: int | None = None,
               variant: str = "full") -> EdgeList:
    """Homogeneous edge list. variant 'clean' has no test nodes; 'full' attaches test nodes for scoring."""
    if graph_builder not in HOMOGENEOUS:
        raise ValueError(f"{graph_builder} is not a homogeneous builder; use src.graphs.relational.load_hetero")
    p = edge_paths(dataset, graph_builder, cfg, quick, k)
    if variant == "clean":
        return EdgeList.load(p["clean"])
    if "full" in p:
        return EdgeList.load(p["full"])
    clean = EdgeList.load(p["clean"])
    if p["query"].exists():
        return EdgeList.concat(clean, EdgeList.load(p["query"]))
    return clean
