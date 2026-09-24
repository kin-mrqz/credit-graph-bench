#!/usr/bin/env bash
#SBATCH --job-name=cgb_03_build_graphs
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# kNN on 1.1M Lending Club rows is the slow part; it is CPU-bound (sklearn, n_jobs=-1).
export STAGE=03_build_graphs
source scripts/sbatch/_common.sh
