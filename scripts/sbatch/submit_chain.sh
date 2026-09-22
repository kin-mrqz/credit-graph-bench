#!/usr/bin/env bash
# Submit all stages as a SLURM dependency chain: each starts only after the previous succeeded
# (afterok). A failed stage stops the chain; fix, then resubmit from that stage:
#   bash scripts/sbatch/submit_chain.sh 05_gnn
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "${RUNS_ROOT:-runs}/slurm"
START_FROM="${1:-}"

mapfile -t STAGES < <(python -m src.plan_units --stages)
prev=""
started=0; [ -z "$START_FROM" ] && started=1
for st in "${STAGES[@]}"; do
  if [ $started -eq 0 ]; then [ "$st" = "$START_FROM" ] && started=1 || continue; fi
  f="scripts/sbatch/$st.sbatch"
  [ -f "$f" ] || { echo "missing $f"; exit 2; }
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable "$f")
  else
    jid=$(sbatch --parsable --dependency=afterok:"$prev" "$f")
  fi
  echo "submitted $st as job $jid${prev:+ (after $prev)}"
  prev="$jid"
done
echo "watch:  squeue -u \$USER ;  tail -f ${RUNS_ROOT:-runs}/slurm/*.out"
