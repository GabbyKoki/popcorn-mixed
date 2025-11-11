#!/bin/bash
#SBATCH --job-name=pop_train
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out


source ~/.bashrc
conda activate pop

python train.py -m data=all data.fold_idx=0,1,2,3,4