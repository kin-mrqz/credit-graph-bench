"""Every path in the project is derived here. Notebooks never hardcode a path.

Environment variables:
    DATA_ROOT   where raw and processed data live (default: <repo>/data_raw)
    RUNS_ROOT   where per-unit outputs live      (default: <repo>/runs)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data_raw")).expanduser()
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", REPO_ROOT / "runs")).expanduser()
SPLITS_ROOT = REPO_ROOT / "splits"
CONFIGS_ROOT = REPO_ROOT / "configs"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
REPORT_ROOT = REPO_ROOT / "report"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset / "raw"


def processed_dir(dataset: str, quick: bool = False) -> Path:
    return DATA_ROOT / dataset / ("processed_quick" if quick else "processed")


def graphs_dir(dataset: str, quick: bool = False) -> Path:
    return processed_dir(dataset, quick) / "graphs"


def split_dir(dataset: str, quick: bool = False) -> Path:
    return SPLITS_ROOT / (f"{dataset}_quick" if quick else dataset)


def run_dir(stage: str, dataset: str, graph_builder: str = "none", seed: int | str = 0) -> Path:
    """runs/<stage>/<dataset>/<graph_builder>/<seed>. One unit of work writes here."""
    return RUNS_ROOT / stage / dataset / graph_builder / str(seed)


def parse_run_dir(path: Path) -> dict:
    """Inverse of run_dir. Used by results collection."""
    rel = path.resolve().relative_to(RUNS_ROOT.resolve())
    stage, dataset, graph_builder, seed = rel.parts[:4]
    return {"stage": stage, "dataset": dataset, "graph_builder": graph_builder, "seed": int(seed)}


def describe() -> dict:
    return {
        "REPO_ROOT": str(REPO_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "RUNS_ROOT": str(RUNS_ROOT),
        "SPLITS_ROOT": str(SPLITS_ROOT),
    }
