#!/bin/bash
#SBATCH --job-name=pop_fusion_comparison
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

# Unified comparison study: Pure vs Mixed conditions with different fusion methods
# All experiments use OT matching (use_matched_control=true)

source ~/.bashrc
conda activate pop

echo "Starting fusion comparison study (Pure vs Mixed)..."
echo "=================================================="
echo "Comparing fusion methods: add, gated, cross_attn"
echo "Comparing condition modes: pure, mixed"
echo "All experiments use OT matching (use_matched_control=true)"
echo ""

exp_num=0
total_exps=6

# Pure mode experiments (3 fusion methods)
echo ""
echo "=== PURE MODE EXPERIMENTS ==="
exp_num=$((exp_num + 1))
echo ""
echo "Experiment $exp_num/$total_exps: condition_mode=pure, fusion_type=cross_attn"
python train.py data=all data.condition_mode=pure model.fusion_type=cross_attn data.use_matched_control=true logger.name="pure_cross_attn_ot"

exp_num=$((exp_num + 1))
echo "Experiment $exp_num/$total_exps: condition_mode=pure, fusion_type=add"
python train.py data=all data.condition_mode=pure model.fusion_type=add data.use_matched_control=true logger.name="pure_add_ot"

exp_num=$((exp_num + 1))
echo ""
echo "Experiment $exp_num/$total_exps: condition_mode=pure, fusion_type=gated"
python train.py data=all data.condition_mode=pure model.fusion_type=gated data.use_matched_control=true logger.name="pure_gated_ot"

# Mixed mode experiments (3 fusion methods)
echo ""
echo "=== MIXED MODE EXPERIMENTS ==="
exp_num=$((exp_num + 1))
echo "Experiment $exp_num/$total_exps: condition_mode=mixed, fusion_type=add"
python train.py data=all data.condition_mode=mixed model.fusion_type=add data.use_matched_control=true logger.name="mixed_add_ot"

exp_num=$((exp_num + 1))
echo ""
echo "Experiment $exp_num/$total_exps: condition_mode=mixed, fusion_type=gated"
python train.py data=all data.condition_mode=mixed model.fusion_type=gated data.use_matched_control=true logger.name="mixed_gated_ot"

exp_num=$((exp_num + 1))
echo ""
echo "Experiment $exp_num/$total_exps: condition_mode=mixed, fusion_type=cross_attn"
python train.py data=all data.condition_mode=mixed model.fusion_type=cross_attn data.use_matched_control=true logger.name="mixed_cross_attn_ot"

echo ""
echo "=================================================="
echo "Fusion comparison study completed! ($total_exps experiments)"
echo "Results:"
echo "  - Pure mode: add, gated, cross_attn (3 experiments)"
echo "  - Mixed mode: add, gated, cross_attn (3 experiments)"
echo "Check W&B for results comparison."

