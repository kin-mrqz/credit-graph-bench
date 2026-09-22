#!/usr/bin/env bash
# Run every unit of one stage, as listed in configs/stages.yaml.
#
#   scripts/run_stage.sh 05_gnn
#   QUICK=1 scripts/run_stage.sh 02_baselines_tabular     # smoke run with subsampled data
#   FORCE=1 scripts/run_stage.sh 02_baselines_tabular     # rerun units even if DONE
#   ONLY_DATASET=taiwan scripts/run_stage.sh 05_gnn        # filter units
#
# Continues past failed units (so one bad dataset does not block the others) and
# exits 1 if any unit failed. Failed units are listed at the end.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
STAGE="${1:?usage: run_stage.sh <stage>}"
FORCE="${FORCE:-0}"
QUICK="${QUICK:-0}"
ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_GB="${ONLY_GB:-}"

cd "$REPO_ROOT"
mapfile -t UNITS < <(python -m src.plan_units "$STAGE")
[ ${#UNITS[@]} -gt 0 ] || { echo "[run_stage] no units for $STAGE"; exit 2; }

echo "[run_stage] $STAGE: ${#UNITS[@]} units  (force=$FORCE quick=$QUICK)"
n_ok=0; n_skip=0; n_fail=0; failed=()
for u in "${UNITS[@]}"; do
  read -r ds gb sd <<< "$u"
  [ -n "$ONLY_DATASET" ] && [ "$ds" != "$ONLY_DATASET" ] && continue
  [ -n "$ONLY_GB" ] && [ "$gb" != "$ONLY_GB" ] && continue
  if [ -f "$RUNS_ROOT/$STAGE/$ds/$gb/$sd/DONE" ] && [ "$FORCE" != "1" ]; then
    n_skip=$((n_skip+1)); echo "[run_stage] skip $ds/$gb/$sd"; continue
  fi
  if bash scripts/run_nb.sh "$STAGE" dataset="$ds" graph_builder="$gb" seed="$sd" force="$FORCE" quick="$QUICK"; then
    n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1)); failed+=("$ds/$gb/$sd")
  fi
done

echo "[run_stage] $STAGE summary: ok=$n_ok skip=$n_skip fail=$n_fail"
if [ $n_fail -gt 0 ]; then
  echo "[run_stage] failed units:"
  for f in "${failed[@]}"; do echo "  $STAGE/$f  ->  $RUNS_ROOT/$STAGE/$f/FAILED"; done
  exit 1
fi
