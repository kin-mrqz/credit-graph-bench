#!/usr/bin/env bash
# Run one notebook unit with papermill, detached from any editor.
#
#   scripts/run_nb.sh <stage> [dataset=..] [graph_builder=..] [seed=..] [force=1] [quick=1]
#   scripts/run_nb.sh 02_baselines_tabular dataset=taiwan seed=0
#
# Writes to $RUNS_ROOT/<stage>/<dataset>/<graph_builder>/<seed>/:
#   executed.ipynb   the notebook with outputs (the deliverable for this unit)
#   log.txt          papermill stream
#   DONE             written by the notebook's last cell on success
#   FAILED           written here on a non-zero exit, with the tail of the log
# Skips immediately if DONE exists and force!=1. Exit code 0 on success or skip, 1 on failure.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
KERNEL="${KERNEL:-python3}"

STAGE="${1:?usage: run_nb.sh <stage> [key=value ...]}"; shift
STAGE="$(basename "$STAGE" .ipynb)"
NB="$REPO_ROOT/notebooks/$STAGE.ipynb"
[ -f "$NB" ] || { echo "[run_nb] no notebook $NB"; exit 2; }

dataset="all"; graph_builder="none"; seed="0"; force="0"; quick="${QUICK:-0}"
for kv in "$@"; do
  case "$kv" in
    dataset=*)       dataset="${kv#*=}" ;;
    graph_builder=*) graph_builder="${kv#*=}" ;;
    seed=*)          seed="${kv#*=}" ;;
    force=*)         force="${kv#*=}" ;;
    quick=*)         quick="${kv#*=}" ;;
    *) echo "[run_nb] unknown arg '$kv'"; exit 2 ;;
  esac
done

RUN_DIR="$RUNS_ROOT/$STAGE/$dataset/$graph_builder/$seed"
UNIT="$STAGE/$dataset/$graph_builder/$seed"

if [ -f "$RUN_DIR/DONE" ] && [ "$force" != "1" ]; then
  echo "[run_nb] skip  $UNIT (DONE)"
  exit 0
fi

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/DONE" "$RUN_DIR/FAILED"
LOG="$RUN_DIR/log.txt"
echo "[run_nb] start $UNIT  $(date '+%F %T')" | tee -a "$LOG"

cd "$REPO_ROOT"
papermill "$NB" "$RUN_DIR/executed.ipynb" \
  -k "$KERNEL" --cwd "$REPO_ROOT" --log-output --progress-bar \
  -p dataset "$dataset" \
  -p graph_builder "$graph_builder" \
  -p seed "$seed" \
  -p run_dir "$RUN_DIR" \
  -p force "$force" \
  -p quick "$quick" \
  >> "$LOG" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ] && [ -f "$RUN_DIR/DONE" ]; then
  echo "[run_nb] done  $UNIT  $(date '+%F %T')" | tee -a "$LOG"
  exit 0
fi

{
  echo "$(date '+%F %T') exit=$STATUS"
  echo "--- last 60 log lines ---"
  tail -n 60 "$LOG"
} > "$RUN_DIR/FAILED"
echo "[run_nb] FAIL  $UNIT (exit $STATUS) see $RUN_DIR/FAILED" | tee -a "$LOG"
exit 1
