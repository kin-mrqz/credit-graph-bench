"""Local lint. Executes no notebook cell.

Checks:
  1. ruff on src/ and scripts/
  2. nbqa ruff on notebooks/ (code cells only)
  3. import the light parts of src (no torch needed) and byte-compile everything else
  4. every notebook has a cell tagged 'parameters' with the six standard parameters,
     a STAGE constant that matches its filename, and a final cell that calls ckpt.mark_done
  5. configs/stages.yaml references only notebooks that exist

Usage: python scripts/lint.py   (or: make lint)
"""

from __future__ import annotations

import compileall
import importlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REQUIRED_PARAMS = {"dataset", "seed", "graph_builder", "run_dir", "force", "quick"}
LIGHT_MODULES = [
    "src",
    "src.paths",
    "src.config",
    "src.ckpt",
    "src.device",
    "src.plan_units",
    "src.data",
    "src.data.base",
    "src.data.splits",
    "src.data.registry",
    "src.graphs",
    "src.graphs.base",
    "src.graphs.knn",
    "src.graphs.random_control",
    "src.graphs.shared_attribute",
    "src.graphs.features",
    "src.eval",
    "src.eval.metrics",
    "src.eval.bootstrap",
    "src.eval.calibration",
    "src.eval.stability",
    "src.eval.business",
    "src.eval.results",
    "src.models.xgb",
    "src.models.lr",
    "src.models.distill",
    "src.viz",
]

failures: list[str] = []


def section(title: str) -> None:
    print(f"\n== {title}")


def run(cmd: list[str], soft: bool = False) -> bool:
    """Run a python-module tool (ruff, nbqa) with the same interpreter that runs this script."""
    if importlib.util.find_spec(cmd[0]) is None and shutil.which(cmd[0]) is None:
        msg = f"{cmd[0]} not installed (pip install -r requirements-dev.txt)"
        if soft:
            print("   skip:", msg)
            return True
        failures.append(msg)
        print("   FAIL:", msg)
        return False
    full = [sys.executable, "-m", *cmd] if importlib.util.find_spec(cmd[0]) is not None else cmd
    proc = subprocess.run(full, cwd=REPO, capture_output=True, text=True, check=False)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        failures.append(" ".join(cmd))
        print("   FAIL")
        print(out[-4000:])
        return False
    print("   ok" + (f" ({out.splitlines()[-1]})" if out else ""))
    return True


# 1. ruff on python
section("ruff src/ scripts/")
run(["ruff", "check", "src", "scripts"])

# 2. nbqa ruff on notebooks
section("nbqa ruff notebooks/")
run(["nbqa", "ruff", "notebooks"], soft=True)

# 3. imports and byte-compile
section("import light modules")
for mod in LIGHT_MODULES:
    try:
        importlib.import_module(mod)
        print(f"   ok   {mod}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"import {mod}: {exc}")
        print(f"   FAIL {mod}: {exc}")

section("byte-compile src/ (torch modules included)")
if compileall.compile_dir(str(REPO / "src"), quiet=1, force=True):
    print("   ok")
else:
    failures.append("compileall src")
    print("   FAIL")

# 4. notebook structure
section("notebook structure")
nb_dir = REPO / "notebooks"
notebooks = sorted(nb_dir.glob("*.ipynb"))
if not notebooks:
    failures.append("no notebooks found")
for nb_path in notebooks:
    problems = []
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{nb_path.name}: invalid JSON ({exc})")
        print(f"   FAIL {nb_path.name}: invalid JSON")
        continue
    cells = nb.get("cells", [])
    code = [c for c in cells if c.get("cell_type") == "code"]
    param_cells = [c for c in code if "parameters" in c.get("metadata", {}).get("tags", [])]
    if len(param_cells) != 1:
        problems.append(f"expected 1 'parameters' cell, found {len(param_cells)}")
    else:
        src = "".join(param_cells[0]["source"])
        assigned = set(re.findall(r"^(\w+)\s*=", src, flags=re.M))
        missing = REQUIRED_PARAMS - assigned
        if missing:
            problems.append(f"parameters cell missing {sorted(missing)}")
    all_src = "\n".join("".join(c["source"]) for c in code)
    m = re.search(r'STAGE\s*=\s*"([^"]+)"', all_src)
    if not m:
        problems.append("no STAGE constant")
    elif m.group(1) != nb_path.stem:
        problems.append(f"STAGE '{m.group(1)}' != filename '{nb_path.stem}'")
    if "mark_done(" not in all_src:
        problems.append("no ckpt.mark_done call")
    has_outputs = any(c.get("outputs") for c in code)
    if has_outputs:
        problems.append("has outputs (run `make strip`)")
    if problems:
        failures.extend(f"{nb_path.name}: {p}" for p in problems)
        print(f"   FAIL {nb_path.name}: " + "; ".join(problems))
    else:
        print(f"   ok   {nb_path.name} ({len(cells)} cells)")

# 5. stages.yaml references
section("configs/stages.yaml vs notebooks/")
try:
    from src.config import load_stages

    stages = load_stages()
    names = {p.stem for p in notebooks}
    for st in stages["stage_order"]:
        if st not in names:
            failures.append(f"stage '{st}' has no notebook")
            print(f"   FAIL {st}: no notebook")
        elif st not in stages["stages"]:
            failures.append(f"stage '{st}' has no unit list")
            print(f"   FAIL {st}: no unit list")
        else:
            print(f"   ok   {st}")
except Exception as exc:  # noqa: BLE001
    failures.append(f"stages.yaml: {exc}")
    print(f"   FAIL {exc}")

print()
if failures:
    print(f"LINT FAILED ({len(failures)})")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("LINT OK")
