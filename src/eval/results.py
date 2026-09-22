"""Read and write per-unit results so the report notebook can work from runs/ alone.

Per run_dir:
    predictions/<model>.parquet   position, y, p_val / p_test (long format: split, position, y, p)
    metrics.json                  {"models": {name: {"val": {...}, "test": {...}}}, "compare": [...], "params": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import RUNS_ROOT, parse_run_dir
from .metrics import all_metrics


# ----------------------------------------------------------------------------- predictions
def save_predictions(run_dir: Path, model: str, parts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    """parts = {"val": (positions, y, p), "test": (positions, y, p), ...}"""
    d = Path(run_dir) / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for split, (pos, y, p) in parts.items():
        frames.append(pd.DataFrame({"split": split, "position": np.asarray(pos, dtype=np.int64),
                                    "y": np.asarray(y, dtype=np.float32), "p": np.asarray(p, dtype=np.float32)}))
    path = d / f"{model}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def load_predictions(run_dir: Path, model: str, split: str = "test") -> pd.DataFrame | None:
    path = Path(run_dir) / "predictions" / f"{model}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["split"] == split].sort_values("position").reset_index(drop=True)


def list_prediction_models(run_dir: Path) -> list[str]:
    d = Path(run_dir) / "predictions"
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join two prediction frames on position. Returns (y, p_a, p_b)."""
    m = a.merge(b, on="position", suffixes=("_a", "_b"))
    return m["y_a"].to_numpy(), m["p_a"].to_numpy(), m["p_b"].to_numpy()


def baseline_predictions(dataset: str, seed: int, model: str = "xgb", split: str = "test",
                         stage: str = "02_baselines_tabular", graph_builder: str = "none") -> pd.DataFrame:
    """Reference predictions (stage 02 XGBoost by default) for the same dataset and seed."""
    from ..paths import run_dir

    d = run_dir(stage, dataset, graph_builder, seed)
    df = load_predictions(d, model, split)
    if df is None:
        raise FileNotFoundError(f"No '{model}' predictions in {d}. Run {stage} for {dataset}/seed {seed} first.")
    return df


def compare_against(base_df: pd.DataFrame, positions: np.ndarray, y: np.ndarray, candidates: dict[str, np.ndarray],
                    metrics=("auc", "pr_auc", "capture_10"), n_boot: int = 1000, seed: int = 0,
                    baseline_name: str = "xgb") -> list[dict]:
    """Paired bootstrap of each candidate (given as arrays over `positions`) against a saved baseline frame."""
    from .bootstrap import paired_bootstrap

    rows = []
    for name, p in candidates.items():
        cand = pd.DataFrame({"position": np.asarray(positions), "y": np.asarray(y), "p": np.asarray(p)})
        yy, p_base, p_new = align(base_df, cand)
        for m in metrics:
            r = paired_bootstrap(yy, p_base, p_new, m, n_boot=n_boot, seed=seed)
            r.update({"model": name, "baseline": baseline_name, "n_aligned": int(len(yy))})
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- metrics
def save_metrics(run_dir: Path, payload: dict) -> Path:
    path = Path(run_dir) / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_metrics(run_dir: Path) -> dict | None:
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def metrics_block(y_val, p_val, y_test, p_test) -> dict:
    return {"val": all_metrics(y_val, p_val), "test": all_metrics(y_test, p_test)}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------- collection for the report
def collect(runs_root: Path | None = None, stages: list[str] | None = None) -> pd.DataFrame:
    """Tidy table of every metric in every metrics.json under runs/.

    Columns: stage, dataset, graph_builder, seed, model, split, metric, value
    """
    root = Path(runs_root or RUNS_ROOT)
    rows = []
    for mpath in root.glob("*/*/*/*/metrics.json"):
        unit = parse_run_dir(mpath.parent)
        if stages and unit["stage"] not in stages:
            continue
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for model, splits in payload.get("models", {}).items():
            for split, metrics in splits.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        rows.append({**unit, "model": model, "split": split, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows)


def collect_compare(runs_root: Path | None = None) -> pd.DataFrame:
    """Every paired-bootstrap comparison row stored under metrics.json['compare']."""
    root = Path(runs_root or RUNS_ROOT)
    rows = []
    for mpath in root.glob("*/*/*/*/metrics.json"):
        unit = parse_run_dir(mpath.parent)
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for r in payload.get("compare", []):
            rows.append({**unit, **r})
    return pd.DataFrame(rows)


def unit_status(runs_root: Path | None = None) -> pd.DataFrame:
    """DONE / FAILED / missing for every unit in configs/stages.yaml."""
    from ..plan_units import stage_order, units_for

    root = Path(runs_root or RUNS_ROOT)
    rows = []
    for st in stage_order():
        for ds, gb, sd in units_for(st):
            d = root / st / ds / gb / str(sd)
            status = "done" if (d / "DONE").exists() else "failed" if (d / "FAILED").exists() else "missing"
            rows.append({"stage": st, "dataset": ds, "graph_builder": gb, "seed": sd, "status": status})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, split: str = "test", metrics=("auc", "pr_auc", "capture_10", "brier", "ece")) -> pd.DataFrame:
    """mean +/- std over seeds, one row per (stage, dataset, graph_builder, model)."""
    sub = df[(df["split"] == split) & (df["metric"].isin(metrics))]
    g = sub.groupby(["stage", "dataset", "graph_builder", "model", "metric"])["value"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out.pivot_table(index=["stage", "dataset", "graph_builder", "model"], columns="metric",
                           values=["mean", "std", "count"]).reset_index()


def fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
