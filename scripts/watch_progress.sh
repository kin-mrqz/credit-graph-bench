#!/usr/bin/env bash
# Report Slurm queue status and notebook unit progress, then repeat.
#
#   bash scripts/watch_progress.sh            # every 10 minutes, until the queue is empty
#   INTERVAL=60 bash scripts/watch_progress.sh
#   bash scripts/watch_progress.sh --once
#
# Logs nothing by itself. To keep it after logout:
#   nohup bash scripts/watch_progress.sh >> runs/progress/watch.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

INTERVAL="${INTERVAL:-600}"
ONCE=0
if [ "${1:-}" = "--once" ]; then
  ONCE=1
fi

report() {
  python - <<'PY'
import os
import subprocess

from src.paths import RUNS_ROOT
from src.plan_units import stage_order, units_for

root = RUNS_ROOT
print(f"[progress] runs={root}")

try:
    q = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%.10i %.26j %.2t %.10M %R"],
        check=False, capture_output=True, text=True,
    )
    lines = [ln for ln in q.stdout.splitlines() if ln.strip()]
except OSError as exc:
    lines = []
    print(f"[progress] squeue unavailable: {exc}")
else:
    if q.returncode != 0:
        print(f"[progress] squeue exit {q.returncode} {(q.stderr or '').strip()}")
    print("[progress] queue:" if lines else "[progress] queue: empty")
    for ln in lines:
        print(f"  {ln}")

print("[progress] units:")
for stage in stage_order():
    units = units_for(stage)
    n_done = n_fail = n_active = 0
    active = []
    failed = []
    for ds, gb, seed in units:
        d = root / stage / ds / gb / str(seed)
        if (d / "DONE").is_file():
            n_done += 1
        elif (d / "FAILED").is_file():
            n_fail += 1
            failed.append(f"{ds}/{gb}/{seed}")
        elif (d / "log.txt").is_file():
            n_active += 1
            active.append(f"{ds}/{gb}/{seed}")
    n_wait = len(units) - n_done - n_fail - n_active
    print(f"  {stage}: done={n_done} failed={n_fail} active={n_active} waiting={n_wait} / {len(units)}")
    for name in active:
        print(f"    active {name}")
    for name in failed:
        print(f"    failed {name}")

logs = sorted((root / "slurm").glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
if logs:
    latest = logs[0]
    print(f"[progress] latest log: {latest}")
    text = latest.read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in text[-8:]:
        print(f"  {ln}")
PY
}

while true; do
  echo
  echo "======== $(date '+%F %T') ========"
  report
  if [ "$ONCE" = "1" ]; then
    break
  fi
  if ! squeue -u "$USER" -h -n cgb_00_env_check,cgb_01_load_and_split,cgb_02_baselines_tabular,cgb_03_build_graphs,cgb_04_graph_features_xgb,cgb_05_gnn,cgb_06_hybrid,cgb_07_lupi_distill,cgb_08_calibration,cgb_09_report 2>/dev/null | grep -q .; then
    echo "[progress] no cgb jobs left in the queue; stopping"
    break
  fi
  echo "[progress] next check in ${INTERVAL}s"
  sleep "$INTERVAL"
done
