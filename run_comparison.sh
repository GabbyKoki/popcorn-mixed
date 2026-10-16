#!/bin/bash
#SBATCH --job-name=pop_comparison
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source ~/.bashrc
conda activate pop

# Backward-compatible launcher for prediction mode comparison.
bash experiments/comparison_prediction_mode/run.sh "$@"
