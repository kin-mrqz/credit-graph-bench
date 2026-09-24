"""The Bundle: one tabular dataset in a fixed on-disk layout.

Every loader returns a Bundle. Every later stage reads a Bundle from ``processed_dir``.
Rows are in a fixed order; ``position`` (0..n-1) is the node id used by every graph builder,
split file, prediction file, and embedding matrix.

Files under processed_dir:
    features.parquet   X, numeric only (categoricals encoded as int codes, -1 = missing)
    index.parquet      ids, y (float, NaN = unlabelled), time (optional), subgroup columns
    attrs.parquet      optional high-cardinality attributes for shared-attribute edges
    meta.json          categorical columns, feature names, denylist applied, notes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def is_text_like(series: pd.Series) -> bool:
    """True for object, pandas/pyarrow string, and categorical columns.

    ``read_csv`` with PyArrow yields ``string[pyarrow]``, which is not ``dtype == object``.
    Those columns still have to be encoded before a float cast.
    """
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def encode_categoricals(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Encode text/category columns as int codes (-1 for missing). Returns (df, categorical_cols)."""
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if is_text_like(df[c])]
    for c in cols:
        if c in df.columns:
            df[c] = pd.Categorical(df[c].astype("string")).codes.astype(np.int32)
    return df, list(cols)


@dataclass
class Bundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    time: pd.Series | None = None
    subgroups: dict[str, pd.Series] = field(default_factory=dict)
    attrs: pd.DataFrame | None = None
    categorical: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic views
    def __len__(self) -> int:
        return len(self.y)

    @property
    def n(self) -> int:
        return len(self.y)

    def labelled_mask(self) -> np.ndarray:
        return self.y.notna().to_numpy()

    def y_int(self) -> np.ndarray:
        """Labels as int with -1 for unlabelled rows."""
        return np.where(self.y.notna(), self.y.fillna(-1).astype(int), -1)

    def X_numpy(self, fill: float = 0.0) -> np.ndarray:
        return self.X.fillna(fill).to_numpy(dtype=np.float32)

    def positive_rate(self) -> float:
        return float(self.y.dropna().mean()) if self.y.notna().any() else float("nan")

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_rows": int(self.n),
            "n_features": int(self.X.shape[1]),
            "n_labelled": int(self.labelled_mask().sum()),
            "positive_rate": self.positive_rate(),
            "has_time": self.time is not None,
            "n_categorical": len(self.categorical),
            "has_attrs": self.attrs is not None,
        }

    # ------------------------------------------------------------------ io
    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.X.reset_index(drop=True).to_parquet(out_dir / "features.parquet", index=False)
        idx = pd.DataFrame({"position": np.arange(self.n), "id": self.ids.to_numpy(), "y": self.y.to_numpy(dtype=float)})
        if self.time is not None:
            idx["time"] = pd.to_datetime(self.time).to_numpy()
        for k, s in self.subgroups.items():
            idx[f"sub_{k}"] = s.to_numpy()
        idx.to_parquet(out_dir / "index.parquet", index=False)
        if self.attrs is not None:
            self.attrs.reset_index(drop=True).to_parquet(out_dir / "attrs.parquet", index=False)
        meta = {
            **self.meta,
            "name": self.name,
            "categorical": self.categorical,
            "feature_names": list(self.X.columns),
            "summary": self.summary(),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> "Bundle":
        out_dir = Path(out_dir)
        if not (out_dir / "features.parquet").exists():
            raise FileNotFoundError(f"No processed bundle at {out_dir}. Run notebook 01_load_and_split first.")
        X = pd.read_parquet(out_dir / "features.parquet")
        idx = pd.read_parquet(out_dir / "index.parquet")
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        subgroups = {c[4:]: idx[c] for c in idx.columns if c.startswith("sub_")}
        attrs = pd.read_parquet(out_dir / "attrs.parquet") if (out_dir / "attrs.parquet").exists() else None
        return cls(
            name=meta["name"],
            X=X,
            y=idx["y"],
            ids=idx["id"],
            time=idx["time"] if "time" in idx.columns else None,
            subgroups=subgroups,
            attrs=attrs,
            categorical=meta.get("categorical", []),
            meta={k: v for k, v in meta.items() if k not in {"categorical", "feature_names", "summary"}},
        )


def apply_denylist(df: pd.DataFrame, denylist: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop denylisted columns. Returns (df, dropped_columns_actually_present)."""
    present = [c for c in denylist if c in df.columns]
    return df.drop(columns=present), present


def subsample(df: pd.DataFrame, n_rows: int | None, seed: int = 0, stratify: pd.Series | None = None) -> pd.DataFrame:
    """Quick-mode row subsample. Keeps every positive if stratify is given and positives are rare."""
    if n_rows is None or len(df) <= n_rows:
        return df
    rng = np.random.default_rng(seed)
    if stratify is not None:
        pos = np.flatnonzero(stratify.to_numpy() == 1)
        neg = np.flatnonzero(stratify.to_numpy() != 1)
        n_pos = min(len(pos), max(1, n_rows // 5))
        n_neg = n_rows - n_pos
        keep = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        keep.sort()
        return df.iloc[keep]
    return df.iloc[np.sort(rng.choice(len(df), n_rows, replace=False))]
