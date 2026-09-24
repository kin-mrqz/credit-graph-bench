#!/usr/bin/env bash
#SBATCH --job-name=cgb_01_load_and_split
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%j.out
# Memory: Lending Club CSV (1.7 GB) and Home Credit child tables (27M rows) are read with pandas.
export STAGE=01_load_and_split
source scripts/sbatch/_common.sh
