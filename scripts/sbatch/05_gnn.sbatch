#!/usr/bin/env bash
#SBATCH --job-name=cgb_05_gnn
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# Heaviest stage: DGraph-Fin (3.7M nodes) x 2 archs x 5 seeds plus Home Credit hetero and
# Lending Club. Epoch checkpoints mean a time-limit kill loses at most one epoch per unit;
# resubmit the same file to continue.
export STAGE=05_gnn
source scripts/sbatch/_common.sh
