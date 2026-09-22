#!/usr/bin/env bash
# Run run_all.sh (or any stage) detached from the SSH session, for nodes without SLURM.
# Uses tmux if available (attachable), otherwise nohup.
#
#   scripts/run_detached.sh                    # run_all.sh
#   scripts/run_detached.sh 05_gnn             # run_stage.sh 05_gnn
#   tmux attach -t cgb                          # watch a tmux run
#   tail -f runs/detached/*.log                 # watch a nohup run
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
mkdir -p "$RUNS_ROOT/detached"
STAMP="$(date '+%Y%m%d_%H%M%S')"

if [ $# -gt 0 ]; then
  CMD="bash scripts/run_stage.sh $*"
  NAME="cgb_$1"
else
  CMD="bash scripts/run_all.sh"
  NAME="cgb"
fi
LOG="$RUNS_ROOT/detached/${NAME}_$STAMP.log"
FULL="cd '$REPO_ROOT' && source scripts/env.sh && $CMD"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$NAME" "bash -lc \"$FULL 2>&1 | tee '$LOG'\""
  echo "[detached] tmux session '$NAME' started. Attach: tmux attach -t $NAME   Log: $LOG"
else
  nohup bash -lc "$FULL" > "$LOG" 2>&1 &
  echo "[detached] nohup pid $! started. Log: $LOG"
fi
