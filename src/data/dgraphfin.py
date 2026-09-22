"""DGraph-Fin (Finvolution, NeurIPS 2022 Datasets and Benchmarks).

dgraphfin.npz keys: x (N,17) float, y (N,) int in {0 normal, 1 fraud/overdue, 2/3 background},
edge_index (E,2), edge_type (E,), edge_timestamp (E,), train_mask / valid_mask / test_mask (index arrays).

The label is defined by overdue non-repayment, so we treat it as a default proxy.
Background nodes (y in {2,3}) are unlabelled but kept: the paper shows removing them hurts.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from ..paths import raw_dir
from .base import Bundle

ZIP_NAME = "DGraphFin.zip"
NPZ_NAME = "dgraphfin.npz"


def raw_present(cfg: dict) -> dict[str, bool]:
    d = raw_dir("dgraphfin")
    return {ZIP_NAME: (d / ZIP_NAME).exists(), NPZ_NAME: (d / NPZ_NAME).exists()}


def _npz_path():
    d = raw_dir("dgraphfin")
    npz = d / NPZ_NAME
    if npz.exists():
        return npz
    z = d / ZIP_NAME
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            member = [m for m in zf.namelist() if m.endswith(".npz")][0]
            zf.extract(member, d)
            extracted = d / member
            if extracted != npz:
                extracted.replace(npz)
        return npz
    raise FileNotFoundError(f"Download DGraphFin.zip from https://dgraph.xinye.com into {d}")


def load_raw() -> dict[str, np.ndarray]:
    with np.load(_npz_path()) as z:
        return {k: z[k] for k in z.files}


def build(cfg: dict, quick: bool = False) -> Bundle:
    """Tabular Bundle for all N nodes (position == node id). Also stashes the official split and
    edge arrays in ``meta`` for notebook 01 to write out."""
    raw = load_raw()
    x = raw["x"].astype(np.float32)
    y_raw = raw["y"].astype(int)
    n = x.shape[0]

    keep = np.arange(n)
    if quick:
        # keep a random node subset plus all labelled positives; edges are filtered to the subset
        rng = np.random.default_rng(0)
        n_keep = min(n, cfg["quick"]["n_rows"])
        pos = np.flatnonzero(y_raw == 1)
        rest = rng.choice(np.setdiff1d(np.arange(n), pos), max(0, n_keep - len(pos)), replace=False)
        keep = np.sort(np.concatenate([pos, rest]))
        x, y_raw = x[keep], y_raw[keep]
        n = len(keep)
    remap = None
    if quick:
        remap = -np.ones(raw["x"].shape[0], dtype=np.int64)
        remap[keep] = np.arange(n)

    y = np.where(np.isin(y_raw, [0, 1]), y_raw.astype(float), np.nan)

    cols = [f"f{i}" for i in range(x.shape[1])]
    X = pd.DataFrame(x, columns=cols)
    if cfg.get("features", {}).get("add_missing_flags", True):
        flags = (x == -1).astype(np.float32)
        X = pd.concat([X, pd.DataFrame(flags, columns=[f"m{i}" for i in range(x.shape[1])])], axis=1)

    # official split (index arrays over original node ids)
    def _map(idx):
        idx = np.asarray(idx, dtype=np.int64)
        if remap is None:
            return idx
        m = remap[idx]
        return m[m >= 0]

    split = {"train": _map(raw["train_mask"]), "val": _map(raw["valid_mask"]), "test": _map(raw["test_mask"])}

    ei = raw["edge_index"].astype(np.int64)
    if ei.shape[0] == 2 and ei.shape[1] != 2:
        ei = ei.T
    src, dst = ei[:, 0], ei[:, 1]
    et = raw["edge_type"].astype(np.int16)
    ets = raw["edge_timestamp"].astype(np.int32)
    if remap is not None:
        s2, d2 = remap[src], remap[dst]
        ok = (s2 >= 0) & (d2 >= 0)
        src, dst, et, ets = s2[ok], d2[ok], et[ok], ets[ok]

    bundle = Bundle(
        name="dgraphfin", X=X.astype(np.float32), y=pd.Series(y), ids=pd.Series(keep if quick else np.arange(n)),
        time=None, subgroups={}, attrs=None, categorical=[],
        meta={"source": cfg["source"], "quick": quick, "n_background": int(np.isnan(y).sum()),
              "label_rule": "1 = overdue non-repayment after reminders (paper calls it fraud); 0 = normal; NaN = background"},
    )
    # transient payload for notebook 01 (not saved by Bundle.save)
    bundle.meta["_official_split"] = split
    bundle.meta["_edges"] = {"src": src, "dst": dst, "edge_type": et, "edge_time": ets, "n_nodes": n}
    return bundle


def instructions() -> str:
    return "Manual: download DGraphFin.zip from https://dgraph.xinye.com and place it under data_raw/dgraphfin/raw/."
