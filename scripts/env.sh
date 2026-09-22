#!/usr/bin/env bash
# Source this on the GPU node before running anything:  source scripts/env.sh
# Sets project paths and activates the environment created by scripts/setup_env.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# Data and outputs. Override before sourcing if you use scratch space, e.g.
#   export DATA_ROOT=/scratch/$USER/cgb_data RUNS_ROOT=/scratch/$USER/cgb_runs
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data_raw}"
export RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
mkdir -p "$DATA_ROOT" "$RUNS_ROOT"

# Environment: conda env name or a venv directory. setup_env.sh writes .env_kind.
ENV_NAME="${ENV_NAME:-credit-graph-bench}"
if [ -f "$REPO_ROOT/.env_kind" ]; then
  ENV_KIND="$(cat "$REPO_ROOT/.env_kind")"
else
  ENV_KIND="venv"
fi

if [ "$ENV_KIND" = "conda" ]; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME"
else
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[env] REPO_ROOT=$REPO_ROOT"
echo "[env] DATA_ROOT=$DATA_ROOT"
echo "[env] RUNS_ROOT=$RUNS_ROOT"
echo "[env] python=$(command -v python)"
