#!/usr/bin/env bash
# Sourced by every .sbatch file after the #SBATCH header.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "${RUNS_ROOT:-runs}/slurm"
# shellcheck disable=SC1091
source scripts/env.sh
echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) stage=$STAGE start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[sbatch] no GPU visible"
bash scripts/run_stage.sh "$STAGE"
STATUS=$?
echo "[sbatch] stage=$STAGE exit=$STATUS end=$(date '+%F %T')"
exit $STATUS
