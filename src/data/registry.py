"""Map a dataset name to its loader module. Loaders are imported lazily."""

from __future__ import annotations

import importlib
from types import ModuleType

_LOADERS = {
    "taiwan": "src.data.taiwan",
    "home_credit": "src.data.home_credit",
    "lending_club": "src.data.lending_club",
    "dgraphfin": "src.data.dgraphfin",
}


def get(dataset: str) -> ModuleType:
    """Return the loader module. Each exposes ``build(cfg, quick) -> Bundle`` and ``raw_present(cfg) -> dict``."""
    if dataset not in _LOADERS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {list(_LOADERS)}")
    return importlib.import_module(_LOADERS[dataset])


def names() -> list[str]:
    return list(_LOADERS)
