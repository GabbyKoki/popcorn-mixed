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
import plotly.graph_objects as go

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from pert_bags import BagCreator
from bag_sampler import BootstrapBagSampler
from dataset import PerturbationDataset, MixedConditionDataset, EvalPerturbationDataset, collate_batch
from model import TransformerMILLightning


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    
    seed_everything(cfg.training.seed)
    # Load AnnData
    adata_path = os.path.join(cfg.paths.processed_data_dir, cfg.data.h5ad_filename)
    adata = sc.read_h5ad(adata_path)
    class_list = sorted(adata.obs[cfg.data.cluster_col].unique())

    # Load perturbation embeddings
    pert_embeddings = adata.uns[cfg.data.pert_embedding_key]

    # Get train/val/test conditions from split column (exclude control)
    split_col = cfg.data.split_col
    train_conditions = adata.obs[adata.obs[split_col] == 'train'][cfg.data.condition_col].unique()
    val_conditions = adata.obs[adata.obs[split_col] == 'val'][cfg.data.condition_col].unique()
    test_conditions = adata.obs[adata.obs[split_col] == 'test'][cfg.data.condition_col].unique()
    
    # Remove control condition from all splits
    train_conditions = train_conditions[train_conditions != cfg.data.control_condition]
    val_conditions = val_conditions[val_conditions != cfg.data.control_condition]
    test_conditions = test_conditions[test_conditions != cfg.data.control_condition]
    
    print(f"Train conditions: {train_conditions}")
    print(f"Val conditions: {val_conditions}")
    print(f"Test conditions: {test_conditions}")

    # Create perturbation-control matching for ALL conditions
    bag_creator = BagCreator(
        adata=adata,
        condition_col=cfg.data.condition_col,
        cluster_col=cfg.data.cluster_col,
        control_condition=cfg.data.control_condition,
        matching_index_path=cfg.paths.matching_index_path
    )
    if cfg.data.compute_matching:
        # Always compute matching for all conditions
        bag_creator.compute_matching_index(list(train_conditions) + list(val_conditions) + list(test_conditions))
    else:
        bag_creator.load_matching_index()

    # Determine training mode: OOD or IID
    ood_mode = cfg.training.get('ood_mode', True)  # Default to OOD mode
    
    if ood_mode:
        # OOD mode: training only samples from train conditions
        training_sampling_conditions = list(train_conditions)
        print("Using OOD mode: training samples only from train conditions")
    else:
        # IID mode: training samples from all conditions
        training_sampling_conditions = list(train_conditions) + list(val_conditions) + list(test_conditions)
        print("Using IID mode: training samples from all conditions")
    
    print(f"Training will sample from {len(training_sampling_conditions)} conditions: {training_sampling_conditions}")

    # Create sampler for training
    sampler = BootstrapBagSampler(ratio=cfg.training.sample_ratio, n_bags=cfg.training.n_bags)
    
    # Create training dataset with condition-based sampling
    train_dataset = MixedConditionDataset(
        bag_creator=bag_creator,
        sampling_conditions=training_sampling_conditions,
        class_list=class_list,
        pert_embeddings=pert_embeddings,
        sampler=sampler,
        seed=cfg.training.seed,
        cell_embed_key=cfg.data.cell_embed_key
    )

    # Create validation and test datasets using EvalPerturbationDataset
    # These use randomly sampled control cells, not matched ones
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

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_batch)

    # Model
    model = TransformerMILLightning(
        input_dim=train_dataset.input_dim,
        gene_input_dim=train_dataset.gene_input_dim,
        output_dim=len(class_list),
        **cfg.model
    )

    # WandB logger
    wandb_logger = WandbLogger(project=cfg.logger.project, name=cfg.logger.name,log_model=cfg.logger.log_model)
    # Push full Hydra config to Weights & Biases config for complete traceability
    try:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    except Exception as e:
        print(f"Warning: failed to push config to wandb: {e}")
    
    

    # Create callbacks. Only add ModelCheckpoint when checkpointing is enabled
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
            filename="best-{epoch:02d}-{val/prop_mse:.4f}"
        )
        callbacks.insert(0, checkpoint_callback)
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=cfg.training.enable_checkpointing,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=cfg.training.log_every_n_steps
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    result = trainer.test(model, test_loader)

    # ---- Collect per-condition predictions and log W&B summary table and stacked bar charts ----
    trained_model = trainer.model
    trained_model.eval()
    device = trained_model.device

    rows = []  # list of [condition, gt_list, pred_list]
    all_gt = []
    all_pred = []
    conds = []

    with torch.no_grad():
        dataset_index = 0
        for batch in test_loader:
            bags, attn_mask, target_proportions, _cell_labels, pert_emd = batch
            bags = bags.to(device)
            attn_mask = attn_mask.to(device)
            pert_emd = pert_emd.to(device)
            pred_prop, _ = trained_model(bags, pert_emd, attn_mask)
            pred_prop = pred_prop.detach().cpu().numpy()
            target_np = target_proportions.detach().cpu().numpy()

            batch_size = pred_prop.shape[0]
            for j in range(batch_size):
                # Get condition name directly from the dataset's perturbation_conditions
                condition_idx = dataset_index + j
                cond_name = test_dataset.perturbation_conditions[condition_idx]
                gt_list = target_np[j].tolist()
                pred_list = pred_prop[j].tolist()
                rows.append([cond_name, gt_list, pred_list])
                conds.append(cond_name)
                all_gt.append(gt_list)
                all_pred.append(pred_list)
            dataset_index += batch_size

    # Optionally order rows by the original test_conditions order
    test_order = list(test_conditions)
    order_index = {c: i for i, c in enumerate(test_order)}
    rows.sort(key=lambda r: order_index.get(r[0], len(test_order)))
    conds = [r[0] for r in rows]
    all_gt = [r[1] for r in rows]
    all_pred = [r[2] for r in rows]

    # Create W&B table (with per-condition interactive Plotly figure)
    table = wandb.Table(columns=["ood_condition", "ood_ground_truth", "ood_prediction", "interactive_plotly"])
    for r in rows:
        cond_name = r[0]
        gt_vals = r[1]
        pred_vals = r[2]
        fig_row = go.Figure()
        fig_row.add_bar(name="GT", x=list(class_list), y=gt_vals)
        fig_row.add_bar(name="Pred", x=list(class_list), y=pred_vals)
        fig_row.update_layout(barmode='group', title=f"{cond_name}", yaxis_title="Proportion", xaxis_tickangle=-45)
        table.add_data(cond_name, gt_vals, pred_vals, wandb.Plotly(fig_row))

    # Create stacked bar charts (GT and Prediction)
    gt_matrix = np.array(all_gt)
    pred_matrix = np.array(all_pred)

    # Ground truth stacked bar
    fig_gt, ax_gt = plt.subplots(figsize=(max(8, len(conds) * 0.6), 4))
    cum = np.zeros(len(conds))
    for cls_idx, cls_name in enumerate(class_list):
        values = gt_matrix[:, cls_idx] if gt_matrix.size else np.zeros(len(conds))
        ax_gt.bar(conds, values, bottom=cum, label=str(cls_name))
        cum += values
    ax_gt.set_ylabel("Proportion")
    ax_gt.set_title("Ground Truth Proportions (OOD)")
    ax_gt.set_xticklabels(conds, rotation=45, ha='right')
    ax_gt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_gt.tight_layout()

    # Prediction stacked bar
    fig_pred, ax_pred = plt.subplots(figsize=(max(8, len(conds) * 0.6), 4))
    cum = np.zeros(len(conds))
    for cls_idx, cls_name in enumerate(class_list):
        values = pred_matrix[:, cls_idx] if pred_matrix.size else np.zeros(len(conds))
        ax_pred.bar(conds, values, bottom=cum, label=str(cls_name))
        cum += values
    ax_pred.set_ylabel("Proportion")
    ax_pred.set_title("Predicted Proportions (OOD)")
    ax_pred.set_xticklabels(conds, rotation=45, ha='right')
    ax_pred.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_pred.tight_layout()

    # Save into W&B summary
    wandb_logger.experiment.summary["validation_table"] = table
    wandb_logger.experiment.summary["ood_conditions"] = conds
    wandb_logger.experiment.summary["gt_stacked_bars"] = wandb.Image(fig_gt)
    wandb_logger.experiment.summary["pred_stacked_bars"] = wandb.Image(fig_pred)

    wandb.log({"test_metric": result})
    wandb.finish()

    # Return validation objective for Optuna sweeper (minimize)
    val_r2 = trainer.callback_metrics.get("val/prop_r2")
    try:
        return float(val_r2.detach().cpu().item()) if val_r2 is not None else float('inf')
    except Exception:
        return float('inf')


if __name__ == "__main__":
    main()