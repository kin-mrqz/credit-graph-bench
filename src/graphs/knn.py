"""kNN edges in standardised feature space. The fully synthetic graph builder.

Two kinds of edge files are produced per k:

    edges_knn_k{k}.npz         "clean": edges among allowed rows (train + val) only. Teachers use this.
    edges_knn_k{k}_query.npz   "query": each test row -> its k nearest allowed rows. Test rows never
                               connect to each other and their labels are never used. Tiers 1-3 attach
                               these at scoring time, which is what a bank would do for a new applicant.

The embedding (imputer, scaler, PCA) is fit on allowed rows only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import EdgeList


class KnnIndex:
    def __init__(self, pca_dims: int | None = 16, seed: int = 0):
        self.pca_dims = pca_dims
        self.seed = seed
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()
        self.pca: PCA | None = None
        self.nn: NearestNeighbors | None = None
        self.allowed: np.ndarray | None = None
        self.n_nodes = 0

    def _embed(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = self.imp.fit_transform(X) if fit else self.imp.transform(X)
        Z = self.sc.fit_transform(Z) if fit else self.sc.transform(Z)
        if self.pca_dims and Z.shape[1] > self.pca_dims:
            if fit:
                self.pca = PCA(n_components=self.pca_dims, random_state=self.seed).fit(Z)
            Z = self.pca.transform(Z)
        return np.nan_to_num(Z).astype(np.float32)

    def fit(self, X: np.ndarray, allowed: np.ndarray, k_max: int) -> KnnIndex:
        self.allowed = np.asarray(allowed, dtype=np.int64)
        self.n_nodes = X.shape[0]
        Za = self._embed(X[self.allowed], fit=True)
        self.nn = NearestNeighbors(n_neighbors=k_max + 1, algorithm="auto", n_jobs=-1).fit(Za)
        self._Za = Za
        return self

    def clean_edges(self, ks: list[int]) -> dict[int, EdgeList]:
        """Allowed -> allowed edges for every k in ks (one neighbour search at max k)."""
        k_max = max(ks)
        _, ind = self.nn.kneighbors(self._Za, n_neighbors=k_max + 1)
        out = {}
        for k in ks:
            out[k] = self._to_edges(np.arange(len(self.allowed)), ind, k, exclude_self=True,
                                    meta={"builder": "knn", "variant": "clean", "k": int(k)})
        return out

    def query_edges(self, X: np.ndarray, query: np.ndarray, ks: list[int]) -> dict[int, EdgeList]:
        """query -> allowed edges. Query rows are embedded with the fitted transform."""
        query = np.asarray(query, dtype=np.int64)
        Zq = self._embed(X[query], fit=False)
        k_max = max(ks)
        _, ind = self.nn.kneighbors(Zq, n_neighbors=k_max)
        out = {}
        for k in ks:
            src = np.repeat(query, k)
            dst = self.allowed[ind[:, :k].ravel()]
            out[k] = EdgeList(src, dst, self.n_nodes, meta={"builder": "knn", "variant": "query", "k": int(k),
                                                            "n_query": len(query)}).dedup()
        return out

    def _to_edges(self, local_src: np.ndarray, ind: np.ndarray, k: int, exclude_self: bool, meta: dict) -> EdgeList:
        src_local = np.repeat(local_src, ind.shape[1])
        dst_local = ind.ravel()
        keep = src_local != dst_local if exclude_self else np.ones(len(src_local), bool)
        e = EdgeList(self.allowed[src_local[keep]], self.allowed[dst_local[keep]], self.n_nodes,
                     meta={**meta, "pca_dims": self.pca_dims, "n_allowed": len(self.allowed), "seed": self.seed})
        return _cap_per_source(e, k).dedup()


def _cap_per_source(edges: EdgeList, k: int) -> EdgeList:
    """Keep the first k edges per source (kneighbors returns them nearest-first)."""
    order = np.argsort(edges.src, kind="stable")
    src_sorted = edges.src[order]
    first = np.r_[0, np.flatnonzero(np.diff(src_sorted)) + 1]
    counts = np.diff(np.r_[first, len(src_sorted)])
    rank = np.arange(len(src_sorted)) - np.repeat(first, counts)
    keep = np.zeros(edges.n_edges, dtype=bool)
    keep[order[rank < k]] = True
    return edges.copy_with(keep)


def build_knn_all(X: np.ndarray, allowed: np.ndarray, query: np.ndarray, ks: list[int], pca_dims: int | None = 16,
                  seed: int = 0) -> tuple[dict[int, EdgeList], dict[int, EdgeList]]:
    """Convenience: (clean edges per k, query edges per k)."""
    idx = KnnIndex(pca_dims, seed).fit(X, allowed, max(ks))
    return idx.clean_edges(ks), idx.query_edges(X, query, ks)
