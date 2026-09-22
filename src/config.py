"""Load YAML configs from configs/."""

from __future__ import annotations

from functools import cache
from typing import Any

import yaml

from .paths import CONFIGS_ROOT

DATASETS = ["taiwan", "home_credit", "lending_club", "dgraphfin"]
GRAPH_DATASETS = {"dgraphfin": "native", "home_credit": "relational", "lending_club": "shared_attribute"}


@cache
def load_config(dataset: str) -> dict[str, Any]:
    path = CONFIGS_ROOT / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for dataset '{dataset}' at {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def load_stages() -> dict[str, Any]:
    with (CONFIGS_ROOT / "stages.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def quick_override(cfg: dict[str, Any], quick: bool) -> dict[str, Any]:
    """Return a shallow copy of the model section with quick-mode overrides applied."""
    models = {k: dict(v) for k, v in cfg["models"].items()}
    if not quick:
        return models
    q = cfg.get("quick", {})
    for m in models.values():
        if "epochs" in m and "epochs" in q:
            m["epochs"] = q["epochs"]
        if "num_boost_round" in m and "num_boost_round" in q:
            m["num_boost_round"] = q["num_boost_round"]
            m["early_stopping_rounds"] = max(5, q["num_boost_round"] // 5)
    return models
