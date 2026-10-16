#!/bin/bash
#SBATCH --job-name=pop_fusion_comparison
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

# Backward-compatible launcher for fusion/condition-mode ablation family.
bash experiments/ablation_fusion_condition_mode/run.sh "$@"
