#!/bin/bash
#SBATCH --job-name=pop_ot_ablation
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source ~/.bashrc
conda activate pop

# Backward-compatible launcher for OT ablation family.
bash experiments/ablation_ot_matching/run.sh "$@"
