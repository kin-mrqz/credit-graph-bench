#!/usr/bin/env bash
#SBATCH --job-name=cgb_08_calibration
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# CPU-only work on saved predictions; the GPU request is only so the job lands on the same partition.
export STAGE=08_calibration_stability_business
source scripts/sbatch/_common.sh
