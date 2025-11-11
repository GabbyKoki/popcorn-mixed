#!/bin/bash
#SBATCH --job-name=pop_ot_ablation
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

# OT matching ablation study: testing with/without OT matching
# Uses default fusion type (add)

source ~/.bashrc
conda activate pop

echo "Starting OT matching ablation study..."
echo "======================================"
echo "Testing matching strategies: with OT matching, without OT matching (random sampling)"
echo "Using default fusion type: add"
echo ""

exp_num=0
total_exps=2

# 1. With OT matching
exp_num=$((exp_num + 1))
echo "Experiment $exp_num/$total_exps: use_matched_control=True (OT matching)"
python train.py data=all data.use_matched_control=true logger.name="ot_matched"

# 2. Without OT matching (random sampling)
exp_num=$((exp_num + 1))
echo ""
echo "Experiment $exp_num/$total_exps: use_matched_control=False (random sampling)"
python train.py data=all data.use_matched_control=false logger.name="ot_random"

echo ""
echo "======================================"
echo "OT matching ablation study completed! ($total_exps experiments)"
echo "Check W&B for results comparison."




