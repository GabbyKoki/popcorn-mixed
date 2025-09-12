#!/bin/bash
#SBATCH --job-name=pop_comparison
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
# 运行 Absolute vs Residual 模式比较实验

echo "Absolute vs Residual Mode Comparison..."

source ~/.bashrc
conda activate pop

# 运行比较脚本
python compare_modes.py \
    logger.name="transformer_mil_comparison" 

echo "Comparison experiment completed! Results saved in comparison_results/ directory"



