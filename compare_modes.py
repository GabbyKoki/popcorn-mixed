#!/usr/bin/env python3
"""
比较脚本：评估 absolute vs residual 预测模式的性能
"""

import os
import pandas as pd
import hydra
import torch
import wandb
import numpy as np
import scanpy as sc
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from pert_bags import BagCreator
from bag_sampler import BootstrapBagSampler
from dataset import PerturbationDataset, MixedConditionDataset, EvalPerturbationDataset, collate_batch
from model import TransformerMILLightning


def run_experiment(cfg: DictConfig, prediction_mode: str, run_name: str):
    """运行单个实验（absolute或residual模式）"""
    
    # 设置随机种子
    seed_everything(cfg.training.seed)
    
    # 加载数据
    adata_path = os.path.join(cfg.paths.processed_data_dir, cfg.data.h5ad_filename)
    adata = sc.read_h5ad(adata_path)
    class_list = sorted(adata.obs[cfg.data.cluster_col].unique())
    
    # 加载perturbation embeddings
    pert_embeddings = adata.uns[cfg.data.pert_embedding_key]
    
    # 获取train/val/test条件
    split_col = cfg.data.split_col
    train_conditions = adata.obs[adata.obs[split_col] == 'train'][cfg.data.condition_col].unique()
    val_conditions = adata.obs[adata.obs[split_col] == 'val'][cfg.data.condition_col].unique()
    test_conditions = adata.obs[adata.obs[split_col] == 'test'][cfg.data.condition_col].unique()
    
    # 移除control condition
    train_conditions = train_conditions[train_conditions != cfg.data.control_condition]
    val_conditions = val_conditions[val_conditions != cfg.data.control_condition]
    test_conditions = test_conditions[test_conditions != cfg.data.control_condition]
    
    # 创建perturbation-control匹配
    bag_creator = BagCreator(
        adata=adata,
        condition_col=cfg.data.condition_col,
        cluster_col=cfg.data.cluster_col,
        control_condition=cfg.data.control_condition,
        matching_index_path=cfg.paths.matching_index_path
    )
    
    if cfg.data.compute_matching:
        bag_creator.compute_matching_index(list(train_conditions) + list(val_conditions) + list(test_conditions))
    else:
        bag_creator.load_matching_index()
    
    # 确定训练模式
    ood_mode = cfg.training.get('ood_mode', True)
    if ood_mode:
        training_sampling_conditions = list(train_conditions)
    else:
        training_sampling_conditions = list(train_conditions) + list(val_conditions) + list(test_conditions)
    
    # 创建数据集
    sampler = BootstrapBagSampler(ratio=cfg.training.sample_ratio, n_bags=cfg.training.n_bags)
    
    train_dataset = MixedConditionDataset(
        bag_creator=bag_creator,
        sampling_conditions=training_sampling_conditions,
        class_list=class_list,
        pert_embeddings=pert_embeddings,
        sampler=sampler,
        seed=cfg.training.seed,
        cell_embed_key=cfg.data.cell_embed_key
    )
    
    val_dataset = EvalPerturbationDataset(
        perturbation_conditions=list(val_conditions),
        adata=adata,
        condition_col=cfg.data.condition_col,
        cluster_col=cfg.data.cluster_col,
        class_list=class_list,
        pert_embeddings=pert_embeddings,
        control_condition=cfg.data.control_condition,
        seed=cfg.training.seed + 1,
        cell_embed_key=cfg.data.cell_embed_key
    )
    
    test_dataset = EvalPerturbationDataset(
        perturbation_conditions=list(test_conditions),
        adata=adata,
        condition_col=cfg.data.condition_col,
        cluster_col=cfg.data.cluster_col,
        class_list=class_list,
        pert_embeddings=pert_embeddings,
        control_condition=cfg.data.control_condition,
        seed=cfg.training.seed + 2,
        cell_embed_key=cfg.data.cell_embed_key
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_batch)
    
    # 创建模型（使用指定的prediction_mode）
    model_cfg = cfg.model.copy()
    model_cfg.prediction_mode = prediction_mode
    
    model = TransformerMILLightning(
        input_dim=train_dataset.input_dim,
        gene_input_dim=train_dataset.gene_input_dim,
        output_dim=len(class_list),
        **model_cfg
    )
    
    # WandB logger
    wandb_logger = WandbLogger(
        project=f"{cfg.logger.project}_comparison",
        name=run_name,
        log_model=False
    )
    
    # 回调函数
    early_stopping_callback = EarlyStopping(
        monitor="val/prop_mse",
        mode="min",
        patience=cfg.training.patience,
        verbose=True
    )
    
    callbacks = [early_stopping_callback]
    if cfg.training.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/prop_mse",
            mode="min",
            save_top_k=1,
            verbose=True,
            filename=f"best-{prediction_mode}-{{epoch:02d}}-{{val/prop_mse:.4f}}"
        )
        callbacks.insert(0, checkpoint_callback)
    
    # 训练器
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=cfg.training.enable_checkpointing,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=cfg.training.log_every_n_steps
    )
    
    # 训练
    trainer.fit(model, train_loader, val_loader)
    
    # 测试
    result = trainer.test(model, test_loader)
    
    # 收集预测结果
    trained_model = trainer.model
    trained_model.eval()
    device = trained_model.device
    
    predictions = []
    ground_truths = []
    conditions = []
    control_props_list = []
    
    with torch.no_grad():
        dataset_index = 0
        for batch in test_loader:
            bags, attn_mask, target_proportions, _cell_labels, pert_emd, control_props = batch
            bags = bags.to(device)
            attn_mask = attn_mask.to(device)
            pert_emd = pert_emd.to(device)
            control_props = control_props.to(device)
            
            pred_prop, _ = trained_model(bags, pert_emd, attn_mask, control_props)
            pred_prop = pred_prop.detach().cpu().numpy()
            target_np = target_proportions.detach().cpu().numpy()
            control_np = control_props.detach().cpu().numpy()
            
            batch_size = pred_prop.shape[0]
            for j in range(batch_size):
                condition_idx = dataset_index + j
                cond_name = test_dataset.perturbation_conditions[condition_idx]
                predictions.append(pred_prop[j])
                ground_truths.append(target_np[j])
                conditions.append(cond_name)
                control_props_list.append(control_np[j])
            dataset_index += batch_size
    
    wandb.finish()
    
    # 返回结果
    return {
        'predictions': np.array(predictions),
        'ground_truths': np.array(ground_truths),
        'conditions': conditions,
        'control_props': np.array(control_props_list),
        'test_metrics': result[0],
        'val_metrics': {
            'val_mse': trainer.callback_metrics.get("val/prop_mse", float('inf')),
            'val_r2': trainer.callback_metrics.get("val/prop_r2", -float('inf')),
            'val_pcc': trainer.callback_metrics.get("val/prop_pcc", -float('inf'))
        }
    }


def compare_results(absolute_results, residual_results, class_list, save_dir="comparison_results"):
    """比较两种模式的结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 整体性能比较
    print("=== 整体性能比较 ===")
    print(f"Absolute Mode - Test MSE: {absolute_results['test_metrics']['test/prop_mse']:.4f}")
    print(f"Residual Mode - Test MSE: {residual_results['test_metrics']['test/prop_mse']:.4f}")
    print(f"Absolute Mode - Test R2: {absolute_results['test_metrics']['test/prop_r2']:.4f}")
    print(f"Residual Mode - Test R2: {residual_results['test_metrics']['test/prop_r2']:.4f}")
    print(f"Absolute Mode - Test PCC: {absolute_results['test_metrics']['test/prop_pcc']:.4f}")
    print(f"Residual Mode - Test PCC: {residual_results['test_metrics']['test/prop_pcc']:.4f}")
    
    # 2. 计算delta proportions
    abs_deltas = absolute_results['predictions'] - absolute_results['control_props']
    res_deltas = residual_results['predictions'] - residual_results['control_props']
    true_deltas = absolute_results['ground_truths'] - absolute_results['control_props']
    
    # 3. Delta proportion预测性能比较
    abs_delta_mse = np.mean((abs_deltas - true_deltas) ** 2)
    res_delta_mse = np.mean((res_deltas - true_deltas) ** 2)
    
    print("\n=== Delta Proportion预测性能 ===")
    print(f"Absolute Mode - Delta MSE: {abs_delta_mse:.4f}")
    print(f"Residual Mode - Delta MSE: {res_delta_mse:.4f}")
    
    # 4. 每个条件的详细比较
    comparison_data = []
    for i, condition in enumerate(absolute_results['conditions']):
        abs_pred = absolute_results['predictions'][i]
        res_pred = residual_results['predictions'][i]
        gt = absolute_results['ground_truths'][i]
        ctrl = absolute_results['control_props'][i]
        
        abs_mse = np.mean((abs_pred - gt) ** 2)
        res_mse = np.mean((res_pred - gt) ** 2)
        
        abs_delta_mse_cond = np.mean((abs_pred - ctrl - (gt - ctrl)) ** 2)
        res_delta_mse_cond = np.mean((res_pred - ctrl - (gt - ctrl)) ** 2)
        
        comparison_data.append({
            'condition': condition,
            'absolute_mse': abs_mse,
            'residual_mse': res_mse,
            'absolute_delta_mse': abs_delta_mse_cond,
            'residual_delta_mse': res_delta_mse_cond,
            'mse_improvement': abs_mse - res_mse,
            'delta_mse_improvement': abs_delta_mse_cond - res_delta_mse_cond
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(os.path.join(save_dir, "condition_comparison.csv"), index=False)
    
    # 5. 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 整体MSE比较
    axes[0, 0].bar(['Absolute', 'Residual'], 
                   [absolute_results['test_metrics']['test/prop_mse'], 
                    residual_results['test_metrics']['test/prop_mse']])
    axes[0, 0].set_title('Overall MSE Comparison')
    axes[0, 0].set_ylabel('MSE')
    
    # Delta MSE比较
    axes[0, 1].bar(['Absolute', 'Residual'], [abs_delta_mse, res_delta_mse])
    axes[0, 1].set_title('Delta Proportion MSE Comparison')
    axes[0, 1].set_ylabel('Delta MSE')
    
    # 每个条件的MSE改进
    axes[1, 0].hist(df_comparison['mse_improvement'], bins=20, alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', label='No improvement')
    axes[1, 0].set_title('MSE Improvement per Condition (Abs - Res)')
    axes[1, 0].set_xlabel('MSE Improvement')
    axes[1, 0].legend()
    
    # 每个条件的Delta MSE改进
    axes[1, 1].hist(df_comparison['delta_mse_improvement'], bins=20, alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', label='No improvement')
    axes[1, 1].set_title('Delta MSE Improvement per Condition (Abs - Res)')
    axes[1, 1].set_xlabel('Delta MSE Improvement')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 统计显著性检验
    abs_mses = df_comparison['absolute_mse'].values
    res_mses = df_comparison['residual_mse'].values
    abs_delta_mses = df_comparison['absolute_delta_mse'].values
    res_delta_mses = df_comparison['residual_delta_mse'].values
    
    # Paired t-test
    mse_tstat, mse_pval = stats.ttest_rel(abs_mses, res_mses)
    delta_tstat, delta_pval = stats.ttest_rel(abs_delta_mses, res_delta_mses)
    
    print(f"\n=== 统计显著性检验 ===")
    print(f"MSE Paired t-test: t={mse_tstat:.4f}, p={mse_pval:.4f}")
    print(f"Delta MSE Paired t-test: t={delta_tstat:.4f}, p={delta_pval:.4f}")
    
    # 7. 保存总结报告
    summary = {
        'absolute_test_mse': float(absolute_results['test_metrics']['test/prop_mse']),
        'residual_test_mse': float(residual_results['test_metrics']['test/prop_mse']),
        'absolute_test_r2': float(absolute_results['test_metrics']['test/prop_r2']),
        'residual_test_r2': float(residual_results['test_metrics']['test/prop_r2']),
        'absolute_delta_mse': float(abs_delta_mse),
        'residual_delta_mse': float(res_delta_mse),
        'mse_improvement_mean': float(df_comparison['mse_improvement'].mean()),
        'delta_mse_improvement_mean': float(df_comparison['delta_mse_improvement'].mean()),
        'mse_ttest_pval': float(mse_pval),
        'delta_mse_ttest_pval': float(delta_pval),
        'conditions_improved_mse': int((df_comparison['mse_improvement'] > 0).sum()),
        'conditions_improved_delta': int((df_comparison['delta_mse_improvement'] > 0).sum()),
        'total_conditions': len(df_comparison)
    }
    
    import json
    with open(os.path.join(save_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """主函数：比较absolute和residual两种模式"""
    
    print("开始比较 Absolute vs Residual 预测模式...")
    print(OmegaConf.to_yaml(cfg))
    
    # 运行absolute模式实验
    print("\n=== 运行 Absolute Mode 实验 ===")
    absolute_results = run_experiment(cfg, "absolute", f"{cfg.logger.name}_absolute")
    
    # 运行residual模式实验  
    print("\n=== 运行 Residual Mode 实验 ===")
    residual_results = run_experiment(cfg, "residual", f"{cfg.logger.name}_residual")
    
    # 比较结果
    print("\n=== 比较结果 ===")
    class_list = sorted(cfg.data.get('class_list', []))  # 如果配置中没有，将从数据中获取
    if not class_list:
        # 从数据中获取class_list
        adata_path = os.path.join(cfg.paths.processed_data_dir, cfg.data.h5ad_filename)
        adata = sc.read_h5ad(adata_path)
        class_list = sorted(adata.obs[cfg.data.cluster_col].unique())
    
    summary = compare_results(absolute_results, residual_results, class_list)
    
    print("\n=== 比较完成 ===")
    print("结果已保存到 comparison_results/ 目录")
    
    return summary


if __name__ == "__main__":
    main()



