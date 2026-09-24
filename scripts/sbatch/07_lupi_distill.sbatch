#!/usr/bin/env bash
#SBATCH --job-name=cgb_07_lupi_distill
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# Teachers per graph builder (GNN on clean graphs) and a 5x3 (lambda, T) student grid per k.
# Grid points are checkpointed individually; resubmit to continue a killed sweep.
export STAGE=07_lupi_distill
source scripts/sbatch/_common.sh
