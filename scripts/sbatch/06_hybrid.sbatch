#!/usr/bin/env bash
#SBATCH --job-name=cgb_06_hybrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# Reads 05 embeddings; XGBoost on GPU.
export STAGE=06_hybrid
source scripts/sbatch/_common.sh
