#!/usr/bin/env bash
# Run stages 00..08 in order, then 90. Stops at the first stage with any FAILED unit
# and prints what to rerun. Already-DONE units are skipped, so rerunning this script
# after fixing a bug only executes the missing work.
#
#   scripts/run_all.sh                 # everything
#   scripts/run_all.sh 03_build_graphs # start from a stage
#   STOP_ON_FAIL=0 scripts/run_all.sh  # keep going through later stages
#   QUICK=1 scripts/run_all.sh         # smoke test end to end on subsampled data
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
START_FROM="${1:-}"

cd "$REPO_ROOT"
mapfile -t STAGES < <(python -m src.plan_units --stages)

started=0
[ -z "$START_FROM" ] && started=1
overall=0
for st in "${STAGES[@]}"; do
  if [ $started -eq 0 ]; then
    [ "$st" = "$START_FROM" ] && started=1 || continue
  fi
  echo
  echo "================================================================ $st  $(date '+%F %T')"
  if bash scripts/run_stage.sh "$st"; then
    echo "[run_all] $st complete"
  else
    overall=1
    echo "[run_all] $st has failed units."
    echo "[run_all] Inspect:  find $RUNS_ROOT/$st -name FAILED -exec sh -c 'echo; echo \$1; head -n 80 \$1' _ {} \\;"
    echo "[run_all] Rerun only those:  bash scripts/run_stage.sh $st   (DONE units are skipped)"
    if [ "$STOP_ON_FAIL" = "1" ]; then
      echo "[run_all] stopping (STOP_ON_FAIL=1)"
      exit 1
    fi
  fi
done

if [ $overall -eq 0 ]; then
  echo
  echo "[run_all] all stages complete. Exporting report..."
  jupyter nbconvert --to html "$RUNS_ROOT/09_report/all/all/0/executed.ipynb" \
    --output-dir "$REPO_ROOT/report" --output gate_a_report >/dev/null 2>&1 \
    && echo "[run_all] report/gate_a_report.html" || echo "[run_all] nbconvert export failed (run make report)"
fi
exit $overall
