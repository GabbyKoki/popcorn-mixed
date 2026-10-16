#!/bin/bash
#SBATCH --job-name=pop_train
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

# Backward-compatible launcher; delegates to the new family runner.
bash experiments/core_training/run.sh data=all "$@"
