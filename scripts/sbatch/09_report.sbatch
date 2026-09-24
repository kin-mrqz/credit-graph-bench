#!/usr/bin/env bash
#SBATCH --job-name=cgb_09_report
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
export STAGE=09_report
export FORCE=1   # the report is cheap and should always reflect the latest runs/
source scripts/sbatch/_common.sh
STATUS=$?
if [ "$STATUS" -eq 0 ]; then
  jupyter nbconvert --to html "$RUNS_ROOT/09_report/all/all/0/executed.ipynb" \
    --output-dir report --output gate_a_report && echo "[sbatch] report/gate_a_report.html"
  STATUS=$?
fi
exit "$STATUS"
