"""Frozen train / validation / test splits.

A split is a dict of int64 index arrays over Bundle positions:
    {"train": ..., "val": ..., "test": ...}
Splits are written once to splits/<dataset>/split.npz and never regenerated (except in quick mode,
which writes to splits/<dataset>_quick/ and is gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import split_dir

PARTS = ("train", "val", "test")


def stratified_split(y: np.ndarray, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> dict[str, np.ndarray]:
    """Stratified random split over labelled rows only. Unlabelled rows belong to no part."""
    y = np.asarray(y, dtype=float)
    labelled = np.flatnonzero(~np.isnan(y))
    rng = np.random.default_rng(seed)
    parts = {p: [] for p in PARTS}
    for cls in np.unique(y[labelled]):
        idx = labelled[y[labelled] == cls]
        idx = rng.permutation(idx)
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr : n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va :])
    return {p: np.sort(np.concatenate(v)).astype(np.int64) for p, v in parts.items()}


def time_split(time: pd.Series, train_end: str, val_end: str, labelled: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Out-of-time split: train <= train_end < val <= val_end < test."""
    t = pd.to_datetime(time).to_numpy()
    tr_end = np.datetime64(pd.Timestamp(train_end))
    va_end = np.datetime64(pd.Timestamp(val_end))
    ok = ~pd.isna(t)
    if labelled is not None:
        ok &= labelled
    train = np.flatnonzero(ok & (t <= tr_end))
    val = np.flatnonzero(ok & (t > tr_end) & (t <= va_end))
    test = np.flatnonzero(ok & (t > va_end))
    return {"train": train.astype(np.int64), "val": val.astype(np.int64), "test": test.astype(np.int64)}


def from_index_arrays(train, val, test) -> dict[str, np.ndarray]:
    return {"train": np.asarray(train, dtype=np.int64), "val": np.asarray(val, dtype=np.int64), "test": np.asarray(test, dtype=np.int64)}


def assert_valid_split(split: dict[str, np.ndarray], n: int) -> None:
    for p in PARTS:
        if p not in split:
            raise AssertionError(f"split missing part '{p}'")
        idx = split[p]
        if len(idx) == 0:
            raise AssertionError(f"split part '{p}' is empty")
        if idx.min() < 0 or idx.max() >= n:
            raise AssertionError(f"split part '{p}' has out-of-range positions (n={n})")
    tr, va, te = (set(split[p].tolist()) for p in PARTS)
    if tr & va or tr & te or va & te:
        raise AssertionError("split parts overlap")


def masks(split: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    out = {}
    for p in PARTS:
        m = np.zeros(n, dtype=bool)
        m[split[p]] = True
        out[p] = m
    return out


def sizes(split: dict[str, np.ndarray]) -> dict[str, int]:
    return {p: int(len(split[p])) for p in PARTS}


def save_split(dataset: str, split: dict[str, np.ndarray], quick: bool = False, note: str = "") -> Path:
    d = split_dir(dataset, quick)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "split.npz", **{p: split[p] for p in PARTS})
    (d / "split.json").write_text(json.dumps({"sizes": sizes(split), "note": note}, indent=2), encoding="utf-8")
    return d / "split.npz"


def load_split(dataset: str, quick: bool = False) -> dict[str, np.ndarray]:
    path = split_dir(dataset, quick) / "split.npz"
    if not path.exists():
        raise FileNotFoundError(f"No split at {path}. Run notebook 01_load_and_split for '{dataset}' first.")
    with np.load(path) as z:
        return {p: z[p].astype(np.int64) for p in PARTS}
