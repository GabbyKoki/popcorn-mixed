#!/bin/bash
#SBATCH --job-name=pop_tune
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source ~/.bashrc
conda activate pop

# Backward-compatible launcher for HPO family (Optuna).
bash experiments/hpo/run_optuna.sh "$@"
