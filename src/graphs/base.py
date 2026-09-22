"""EdgeList: a homogeneous graph over Bundle positions, saved as npz."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass
class EdgeList:
    src: np.ndarray
    dst: np.ndarray
    n_nodes: int
    edge_type: np.ndarray | None = None
    edge_time: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    def copy_with(self, mask: np.ndarray) -> "EdgeList":
        return EdgeList(
            self.src[mask], self.dst[mask], self.n_nodes,
            None if self.edge_type is None else self.edge_type[mask],
            None if self.edge_time is None else self.edge_time[mask],
            dict(self.meta),
        )

    def dedup(self) -> "EdgeList":
        """Drop self-loops and duplicate (src, dst) pairs, keeping the first occurrence."""
        keep = self.src != self.dst
        key = self.src.astype(np.int64) * self.n_nodes + self.dst.astype(np.int64)
        _, first = np.unique(key, return_index=True)
        m = np.zeros(self.n_edges, dtype=bool)
        m[first] = True
        return self.copy_with(keep & m)

    def symmetrized(self) -> "EdgeList":
        """Add reverse edges. Edge type / time are copied to the reverse edge."""
        out = EdgeList(
            np.concatenate([self.src, self.dst]), np.concatenate([self.dst, self.src]), self.n_nodes,
            None if self.edge_type is None else np.concatenate([self.edge_type, self.edge_type]),
            None if self.edge_time is None else np.concatenate([self.edge_time, self.edge_time]),
            {**self.meta, "symmetrized": True},
        )
        return out.dedup()

    def drop_nodes(self, nodes: np.ndarray) -> "EdgeList":
        """Remove every edge that touches any node in `nodes`. Used for clean-room teacher graphs."""
        bad = np.zeros(self.n_nodes, dtype=bool)
        bad[np.asarray(nodes, dtype=np.int64)] = True
        keep = ~(bad[self.src] | bad[self.dst])
        out = self.copy_with(keep)
        out.meta["dropped_nodes"] = int(bad.sum())
        return out

    def restrict_to(self, allowed: np.ndarray) -> "EdgeList":
        ok = np.zeros(self.n_nodes, dtype=bool)
        ok[np.asarray(allowed, dtype=np.int64)] = True
        return self.copy_with(ok[self.src] & ok[self.dst])

    @staticmethod
    def concat(a: "EdgeList", b: "EdgeList") -> "EdgeList":
        assert a.n_nodes == b.n_nodes, "cannot concat edge lists over different node sets"
        et = None
        if a.edge_type is not None and b.edge_type is not None:
            et = np.concatenate([a.edge_type, b.edge_type])
        tm = None
        if a.edge_time is not None and b.edge_time is not None:
            tm = np.concatenate([a.edge_time, b.edge_time])
        return EdgeList(np.concatenate([a.src, b.src]), np.concatenate([a.dst, b.dst]), a.n_nodes, et, tm,
                        {**a.meta, "concat_with": b.meta.get("variant", "?"), "symmetrized": False}).dedup()

    def filter_time(self, max_time: int) -> "EdgeList":
        if self.edge_time is None:
            return self
        return self.copy_with(self.edge_time <= max_time)

    # ------------------------------------------------------------------ views
    def to_scipy(self, symmetric: bool = True) -> sp.csr_matrix:
        e = self.symmetrized() if symmetric and not self.meta.get("symmetrized") else self
        A = sp.coo_matrix((np.ones(e.n_edges, dtype=np.float32), (e.src, e.dst)), shape=(self.n_nodes, self.n_nodes))
        return A.tocsr()

    def degrees(self, symmetric: bool = True) -> np.ndarray:
        e = self.symmetrized() if symmetric and not self.meta.get("symmetrized") else self
        return np.bincount(e.src, minlength=self.n_nodes)

    def edge_index(self) -> np.ndarray:
        """(2, E) int64, the PyG convention."""
        return np.vstack([self.src, self.dst]).astype(np.int64)

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"src": self.src.astype(np.int64), "dst": self.dst.astype(np.int64), "n_nodes": np.asarray(self.n_nodes)}
        if self.edge_type is not None:
            arrays["edge_type"] = self.edge_type
        if self.edge_time is not None:
            arrays["edge_time"] = self.edge_time
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({**self.meta, "n_edges": self.n_edges, "n_nodes": self.n_nodes},
                                                        indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EdgeList":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No edge list at {path}. Run notebook 03_build_graphs first.")
        with np.load(path) as z:
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            return cls(z["src"], z["dst"], int(z["n_nodes"]),
                       z["edge_type"] if "edge_type" in z.files else None,
                       z["edge_time"] if "edge_time" in z.files else None, meta)


# ---------------------------------------------------------------------- checks and stats
def assert_no_test_nodes(edges: EdgeList, test_idx: np.ndarray, name: str = "graph") -> None:
    """Leakage guard: no edge may touch a test position."""
    is_test = np.zeros(edges.n_nodes, dtype=bool)
    is_test[np.asarray(test_idx, dtype=np.int64)] = True
    n_bad = int((is_test[edges.src] | is_test[edges.dst]).sum())
    if n_bad:
        raise AssertionError(f"LEAKAGE: {name} has {n_bad} edges touching test nodes")


def degree_stats(edges: EdgeList) -> dict:
    d = edges.degrees()
    return {"n_nodes": int(edges.n_nodes), "n_edges": int(edges.n_edges), "isolated_frac": float((d == 0).mean()),
            "deg_mean": float(d.mean()), "deg_median": float(np.median(d)), "deg_p95": float(np.quantile(d, 0.95)),
            "deg_max": int(d.max()) if len(d) else 0}


def label_homophily(edges: EdgeList, y: np.ndarray, train_mask: np.ndarray) -> dict:
    """Among edges where both endpoints are labelled training nodes: P(same label) and
    P(neighbour bad | node bad) vs base rate. Uses training labels only."""
    y = np.asarray(y, dtype=float)
    ok = train_mask[edges.src] & train_mask[edges.dst] & ~np.isnan(y[edges.src]) & ~np.isnan(y[edges.dst])
    if not ok.any():
        return {"n_pairs": 0}
    ys, yd = y[edges.src[ok]], y[edges.dst[ok]]
    base = float(y[train_mask & ~np.isnan(y)].mean())
    bad_src = ys == 1
    return {
        "n_pairs": int(ok.sum()),
        "same_label_frac": float((ys == yd).mean()),
        "p_bad_neighbor_given_bad": float(yd[bad_src].mean()) if bad_src.any() else float("nan"),
        "p_bad_neighbor_given_good": float(yd[~bad_src].mean()) if (~bad_src).any() else float("nan"),
        "base_rate": base,
    }
