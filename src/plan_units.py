"""Expand configs/stages.yaml into (dataset, graph_builder, seed) units.

Usage:
    python -m src.plan_units 05_gnn          -> one line per unit: "dataset graph_builder seed"
    python -m src.plan_units --stages        -> stage names in execution order
    python -m src.plan_units 05_gnn --json   -> JSON list
"""

from __future__ import annotations

import json
import sys

from .config import load_stages


def units_for(stage: str) -> list[tuple[str, str, int]]:
    stages = load_stages()
    if stage not in stages["stages"]:
        raise KeyError(f"Unknown stage '{stage}'. Known: {list(stages['stages'])}")
    default_seeds = stages["default_seeds"]
    out: list[tuple[str, str, int]] = []
    for entry in stages["stages"][stage]:
        seeds = entry.get("seeds", "default")
        seeds = default_seeds if seeds == "default" else seeds
        for gb in entry["graph_builders"]:
            for seed in seeds:
                out.append((entry["dataset"], gb, int(seed)))
    return out


def stage_order() -> list[str]:
    return list(load_stages()["stage_order"])


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    if argv[0] == "--stages":
        print("\n".join(stage_order()))
        return 0
    stage = argv[0]
    units = units_for(stage)
    if "--json" in argv:
        print(json.dumps([{"dataset": d, "graph_builder": g, "seed": s} for d, g, s in units]))
    else:
        for d, g, s in units:
            print(f"{d} {g} {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
