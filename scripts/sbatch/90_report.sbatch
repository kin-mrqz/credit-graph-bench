#!/usr/bin/env bash
#SBATCH --job-name=cgb_90_report
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
export STAGE=90_report
export FORCE=1   # the report is cheap and should always reflect the latest runs/
source scripts/sbatch/_common.sh
jupyter nbconvert --to html "$RUNS_ROOT/90_report/all/all/0/executed.ipynb" \
  --output-dir report --output gate_a_report && echo "[sbatch] report/gate_a_report.html"
