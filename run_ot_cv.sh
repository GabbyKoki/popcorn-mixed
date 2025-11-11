#!/bin/bash
#SBATCH --job-name=pop_ot_ablation
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out



for fold in 0 1 2 3 4; do python train.py data=all data.use_matched_control=false logger.name="ot_random_fold${fold}" data.fold_idx=${fold}; done