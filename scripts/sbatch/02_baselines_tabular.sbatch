#!/usr/bin/env bash
#SBATCH --job-name=cgb_02_baselines_tabular
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# 4 datasets x 5 seeds x (LR, XGBoost on GPU, MLP).
export STAGE=02_baselines_tabular
source scripts/sbatch/_common.sh
