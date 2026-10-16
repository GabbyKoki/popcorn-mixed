import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import hydra
import torch
import wandb
import numpy as np
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
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
from dataset import (
    PerturbationDataset, MixedConditionDataset, PureConditionDataset,
    EvalPerturbationDataset, collate_batch, collate_pure_batch,
)
from model import TransformerMILLightning
from subclusters import build_control_subclusters, get_subcluster_metadata


NATURE_STACKED_PALETTE = [
    "#0F4D92",
    "#3775BA",
    "#42949E",
    "#7C6CCF",
    "#9A4D8E",
    "#B64342",
    "#E28E2C",
    "#7BAA5B",
    "#7884B4",
    "#B89BD9",
    "#D9544D",
    "#5B7FCA",
    "#AADCA9",
    "#E9A6A1",
    "#D8D8D8",
    "#606060",
    "#C45AD6",
    "#33B5A5",
    "#F0C0CC",
    "#B4C0E4",
]


def _apply_publication_style(font_size=8, axes_linewidth=1.0):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False


def _safe_spearman(pred_vals, target_vals):
    pred = np.asarray(pred_vals, dtype=float)
    target = np.asarray(target_vals, dtype=float)
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return float("nan")
    pred_rank = np.argsort(np.argsort(pred))
    target_rank = np.argsort(np.argsort(target))
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def _safe_pcc(pred_vals, target_vals):
    pred = np.asarray(pred_vals, dtype=float)
    target = np.asarray(target_vals, dtype=float)
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def _safe_r2(pred_vals, target_vals):
    pred = np.asarray(pred_vals, dtype=float)
    target = np.asarray(target_vals, dtype=float)
    denom = np.sum((target - target.mean()) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(1.0 - (np.sum((target - pred) ** 2) / denom))


def _metric_dict(pred_vals, target_vals):
    pred = np.asarray(pred_vals, dtype=float)
    target = np.asarray(target_vals, dtype=float)
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "pcc": _safe_pcc(pred, target),
        "r2": _safe_r2(pred, target),
        "spearman": _safe_spearman(pred, target),
    }


def _make_condition_stacked_bar_figure(class_list, condition_labels, proportion_matrix, title):
    _apply_publication_style(font_size=8, axes_linewidth=1.0)
    proportions = np.asarray(proportion_matrix, dtype=float)
    n_conditions = len(condition_labels)
    fig_width = max(8.5, n_conditions * 0.38)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    x = np.arange(n_conditions)
    bottoms = np.zeros(n_conditions, dtype=float)
    colors = [NATURE_STACKED_PALETTE[idx % len(NATURE_STACKED_PALETTE)] for idx in range(len(class_list))]
    for idx, label in enumerate(class_list):
        heights = proportions[:, idx]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.82,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.25,
            label=str(label),
        )
        bottoms += heights

    ax.set_xlim(-0.6, n_conditions - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title(title, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in condition_labels], rotation=60, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7, width=0.8, length=3)
    ax.tick_params(axis="x", width=0.8, length=2)
    ax.grid(False)
    ax.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=6,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    return fig


def _make_grouped_bar_figure(labels, gt_prop, pred_prop, condition, y_max=None):
    gt = np.asarray(gt_prop, dtype=float)
    pred = np.asarray(pred_prop, dtype=float)
    fig_width = max(10, len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = np.arange(len(labels))
    width = 0.42
    ax.bar(x - width / 2, gt, width=width, label="GT", color="#4c78a8")
    ax.bar(x + width / 2, pred, width=width, label="Pred", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Proportion")
    ax.set_title(str(condition))
    if y_max is not None:
        ax.set_ylim(0.0, float(y_max))
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _compute_full_population_gt(adata, conditions, condition_col, cluster_col, class_list,
                                 control_condition, subcluster_col=None, subcluster_labels=None,
                                 subcluster_sizes=None):
    """Compute ground-truth proportions from the full population (no sampling)."""
    ctrl_mask = adata.obs[condition_col] == control_condition
    ctrl_labels = adata.obs.loc[ctrl_mask, cluster_col]
    ctrl_type_prop = ctrl_labels.value_counts(normalize=True).reindex(class_list, fill_value=0).values.astype(np.float64)

    gt = {}
    for cond in conditions:
        pert_mask = adata.obs[condition_col] == cond
        labels = adata.obs.loc[pert_mask, cluster_col]
        type_prop = labels.value_counts(normalize=True).reindex(class_list, fill_value=0).values.astype(np.float64)
        entry = {"type_prop": type_prop, "ctrl_type_prop": ctrl_type_prop}

        if subcluster_col and subcluster_labels and subcluster_sizes:
            sub_index = {label: idx for idx, label in enumerate(subcluster_labels)}
            for src_name, src_mask in [("sub_prop", pert_mask), ("ctrl_sub_prop", ctrl_mask)]:
                sub_labels_arr = adata.obs.loc[src_mask, subcluster_col].values
                type_labels_arr = adata.obs.loc[src_mask, cluster_col].values
                props = np.zeros(len(subcluster_labels), dtype=np.float64)
                for type_idx, type_label in enumerate(class_list):
                    tmask = type_labels_arr == type_label
                    if tmask.sum() == 0:
                        continue
                    subs = sub_labels_arr[tmask]
                    block_start = sum(subcluster_sizes[:type_idx])
                    for sl in subs:
                        si = sub_index.get(sl, -1)
                        if si >= 0:
                            props[si] += 1
                    block_end = block_start + subcluster_sizes[type_idx]
                    block_total = props[block_start:block_end].sum()
                    if block_total > 0:
                        props[block_start:block_end] /= block_total
                entry[src_name] = props
        gt[cond] = entry
    return gt


def _prepare_cell_encoder_view(adata, cfg: DictConfig, split_col: str):
    variant = str(cfg.data.get("encoder_variant", "") or "").strip()
    if not variant:
        return adata, cfg.data.cell_embed_key

    train_mask = np.asarray(adata.obs[split_col] == "train")
    if variant == "scgpt512":
        print("Cell encoder variant: scgpt512 -> adata.obsm['X_scgpt_CP_512']")
        return adata, "X_scgpt_CP_512"

    if variant == "pca100_global":
        print("Cell encoder variant: pca100_global -> adata.obsm['X_pca']")
        return adata, "X_pca"

    if variant == "scvi100":
        print("Cell encoder variant: scvi100 -> adata.obsm['X_scvi_100']")
        return adata, "X_scvi_100"

    if variant == "geneformer_v1":
        print("Cell encoder variant: geneformer_v1 -> adata.obsm['X_geneformer_v1_256']")
        return adata, "X_geneformer_v1_256"

    if variant == "pca100_train":
        n_comps = int(cfg.data.get("encoder_pca_n_comps", 100))
        derived_key = f"X_pca_train_{n_comps}"
        if derived_key not in adata.obsm:
            print(f"Cell encoder variant: pca100_train -> fitting TruncatedSVD(n_components={n_comps}) on train split only")
            svd = TruncatedSVD(n_components=n_comps, random_state=cfg.training.seed)
            svd.fit(adata.X[train_mask])
            adata = adata.copy()
            adata.obsm[derived_key] = svd.transform(adata.X).astype(np.float32, copy=False)
        return adata, derived_key

    if variant == "hvg5000_train":
        n_top_genes = int(cfg.data.get("encoder_hvg_top_genes", 5000))
        print(f"Cell encoder variant: hvg5000_train -> selecting top {n_top_genes} HVGs from train split only")
        train_adata = adata[train_mask].copy()
        sc.pp.highly_variable_genes(train_adata, n_top_genes=n_top_genes, subset=False)
        keep_mask = np.asarray(train_adata.var["highly_variable"].values, dtype=bool)
        selected_genes = train_adata.var_names[keep_mask]
        adata = adata[:, selected_genes].copy()
        return adata, "X"

    raise ValueError(f"Unknown data.encoder_variant: {variant}")


@hydra.main(config_path="conf", config_name="tcell", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.training.seed)
    runtime_output_dir = Path(HydraConfig.get().runtime.output_dir)
    adata_path = os.path.join(cfg.data.processed_data_dir, cfg.data.h5ad_filename)
    adata = sc.read_h5ad(adata_path)
    split_col = cfg.data.split_col
    adata, effective_cell_embed_key = _prepare_cell_encoder_view(adata, cfg, split_col)
    cfg.data.cell_embed_key = effective_cell_embed_key
    class_list = sorted(adata.obs[cfg.data.cluster_col].unique())
    pert_embeddings = adata.uns[cfg.data.pert_embedding_key]

    # ---- Subclusters ----
    subcluster_cfg = cfg.data.get("subcluster", {})
    use_subclusters = bool(subcluster_cfg.get("enabled", False))
    subcluster_kwargs = {}
    subcluster_sizes = None
    subcluster_labels = []
    subcluster_meta = None
    if use_subclusters:
        subcluster_col = subcluster_cfg.get("col", "control_subcluster")
        if subcluster_cfg.get("compute", False) or subcluster_col not in adata.obs:
            build_control_subclusters(
                adata=adata,
                type_col=cfg.data.cluster_col,
                condition_col=cfg.data.condition_col,
                control_condition=cfg.data.control_condition,
                embedding_key=subcluster_cfg.get("embedding_key", cfg.data.cell_embed_key),
                subcluster_col=subcluster_col,
                n_neighbors=subcluster_cfg.get("n_neighbors", 15),
                resolution=subcluster_cfg.get("resolution", 1.0),
                random_state=cfg.training.seed,
                output_json=subcluster_cfg.get("metadata_path", None),
            )
        subcluster_meta = get_subcluster_metadata(adata, cfg.data.cluster_col, subcluster_col, class_list)
        subcluster_sizes = subcluster_meta["subcluster_sizes"]
        subcluster_labels = subcluster_meta["subclusters"]
        subcluster_kwargs = {
            "subcluster_col": subcluster_col,
            "subcluster_list": subcluster_labels,
            "subcluster_to_type": subcluster_meta["subcluster_to_type"],
        }
        print(f"Using {sum(subcluster_sizes)} control-defined subclusters across {len(class_list)} types")

    # ---- Splits ----
    train_conditions = adata.obs[adata.obs[split_col] == 'train'][cfg.data.condition_col].unique()
    val_conditions = adata.obs[adata.obs[split_col] == 'val'][cfg.data.condition_col].unique()
    test_conditions = adata.obs[adata.obs[split_col] == 'test'][cfg.data.condition_col].unique()
    train_conditions = train_conditions[train_conditions != cfg.data.control_condition]
    val_conditions = val_conditions[val_conditions != cfg.data.control_condition]
    test_conditions = test_conditions[test_conditions != cfg.data.control_condition]
    print(f"Train: {len(train_conditions)} | Val: {len(val_conditions)} | Test: {len(test_conditions)}")

    # ---- OT Matching ----
    bag_creator = BagCreator(
        adata=adata,
        condition_col=cfg.data.condition_col,
        cluster_col=cfg.data.cluster_col,
        control_condition=cfg.data.control_condition,
        matching_index_path=cfg.data.matching_index_path,
    )
    all_conds = list(train_conditions) + list(val_conditions) + list(test_conditions)
    if cfg.data.compute_matching:
        bag_creator.compute_matching_index(all_conds)
    else:
        bag_creator.load_matching_index()

    # ---- Datasets ----
    ood_mode = cfg.training.get('ood_mode', True)
    training_sampling_conditions = list(train_conditions) if ood_mode else all_conds
    print(f"Training samples from {len(training_sampling_conditions)} conditions (ood_mode={ood_mode})")

    sampler = BootstrapBagSampler(ratio=cfg.training.sample_ratio, n_bags=cfg.training.n_bags)
    use_matched_control = cfg.data.get('use_matched_control', True)
    condition_mode = cfg.data.get('condition_mode', 'mixed')

    if condition_mode == 'pure':
        train_dataset = PureConditionDataset(
            bag_creator=bag_creator, sampling_conditions=training_sampling_conditions,
            class_list=class_list, pert_embeddings=pert_embeddings, sampler=sampler,
            seed=cfg.training.seed, cell_embed_key=effective_cell_embed_key,
            use_matched_control=use_matched_control, **subcluster_kwargs,
        )
        train_collate_fn = collate_pure_batch
    else:
        train_dataset = MixedConditionDataset(
            bag_creator=bag_creator, sampling_conditions=training_sampling_conditions,
            class_list=class_list, pert_embeddings=pert_embeddings, sampler=sampler,
            seed=cfg.training.seed, cell_embed_key=effective_cell_embed_key,
            use_matched_control=use_matched_control, **subcluster_kwargs,
        )
        train_collate_fn = collate_batch

    # Val: legacy single-shot (for early stopping during training)
    val_dataset = EvalPerturbationDataset(
        perturbation_conditions=list(val_conditions), adata=adata,
        condition_col=cfg.data.condition_col, cluster_col=cfg.data.cluster_col,
        class_list=class_list, pert_embeddings=pert_embeddings,
        control_condition=cfg.data.control_condition, seed=cfg.training.seed + 1,
        cell_embed_key=effective_cell_embed_key, **subcluster_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_batch)

    # ---- Model ----
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if use_subclusters:
        model_cfg["subcluster_sizes"] = subcluster_sizes
    model = TransformerMILLightning(
        input_dim=train_dataset.input_dim,
        gene_input_dim=train_dataset.gene_input_dim,
        output_dim=len(class_list),
        **model_cfg,
    )

    # ---- Logger ----
    logger_name = cfg.logger.name.replace("_fix.h5ad", "") if cfg.logger.name else cfg.data.h5ad_filename.replace("_fix.h5ad", "")
    wandb_save_dir = runtime_output_dir / "wandb"
    wandb_save_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        project=cfg.logger.project, name=logger_name, group=cfg.logger.group,
        save_dir=str(wandb_save_dir), log_model=cfg.logger.log_model,
    )
    try:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    except Exception as e:
        print(f"Warning: failed to push config to wandb: {e}")

    # ---- Train ----
    callbacks = [EarlyStopping(monitor="val/prop_mse", mode="min", patience=cfg.training.patience, verbose=True)]
    if cfg.training.enable_checkpointing:
        callbacks.insert(0, ModelCheckpoint(monitor="val/prop_mse", mode="min", save_top_k=1, verbose=True))

    trainer = Trainer(
        max_epochs=cfg.training.epochs, logger=wandb_logger, callbacks=callbacks,
        enable_checkpointing=cfg.training.enable_checkpointing, deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    trainer.fit(model, train_loader, val_loader)

    # ==================================================================
    # Bootstrap-aggregated test evaluation
    # ==================================================================
    trained_model = trainer.model
    trained_model.eval()
    device = trained_model.device

    eval_n_bags = cfg.training.get("eval_n_bags", 20)
    eval_sampler = BootstrapBagSampler(ratio=cfg.training.sample_ratio, n_bags=eval_n_bags)

    test_bootstrap_dataset = PureConditionDataset(
        bag_creator=bag_creator, sampling_conditions=list(test_conditions),
        class_list=class_list, pert_embeddings=pert_embeddings, sampler=eval_sampler,
        seed=cfg.training.seed + 100, cell_embed_key=effective_cell_embed_key,
        use_matched_control=use_matched_control, **subcluster_kwargs,
    )
    test_bootstrap_loader = DataLoader(
        test_bootstrap_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, collate_fn=collate_pure_batch,
    )

    print(f"\n=== Bootstrap test evaluation: {eval_n_bags} bags/condition, "
          f"{len(test_conditions)} conditions ===")

    cond_type_preds = defaultdict(list)
    cond_sub_preds = defaultdict(list)

    with torch.no_grad():
        for batch in test_bootstrap_loader:
            conditions = batch[-1]  # pure batch always has conditions last
            bags = batch[0].to(device)
            attn_mask = batch[1].to(device)
            pert_emd = batch[4].to(device)

            outputs = trained_model(bags, pert_emd, attn_mask)
            pred_prop = outputs[0].detach().cpu().numpy()
            pred_sub = outputs[2].detach().cpu().numpy() if use_subclusters and len(outputs) > 2 else None

            for j in range(pred_prop.shape[0]):
                cond = conditions[j]
                cond_type_preds[cond].append(pred_prop[j])
                if pred_sub is not None:
                    cond_sub_preds[cond].append(pred_sub[j])

    # Full-population ground truth
    gt_data = _compute_full_population_gt(
        adata, test_conditions, cfg.data.condition_col, cfg.data.cluster_col,
        class_list, cfg.data.control_condition,
        subcluster_col=subcluster_kwargs.get("subcluster_col") if use_subclusters else None,
        subcluster_labels=subcluster_labels if use_subclusters else None,
        subcluster_sizes=subcluster_sizes if use_subclusters else None,
    )

    # Build per-condition result rows
    rows = []
    for cond in test_conditions:
        if cond not in cond_type_preds or cond not in gt_data:
            continue
        pred_avg = np.mean(cond_type_preds[cond], axis=0)
        pred_std = np.std(cond_type_preds[cond], axis=0)
        gt_entry = gt_data[cond]
        gt_prop = gt_entry["type_prop"]
        ctrl_prop = gt_entry["ctrl_type_prop"]

        abs_metrics = _metric_dict(pred_avg, gt_prop)
        delta_metrics = _metric_dict(pred_avg - ctrl_prop, gt_prop - ctrl_prop)

        row = {
            "ood_condition": cond,
            "n_bootstrap_bags": len(cond_type_preds[cond]),
            "class_labels_order": list(map(str, class_list)),
            "ood_ground_truth": gt_prop.tolist(),
            "ood_prediction": pred_avg.tolist(),
            "ood_prediction_std": pred_std.tolist(),
            "control_proportion": ctrl_prop.tolist(),
            "delta_ground_truth": (gt_prop - ctrl_prop).tolist(),
            "delta_prediction": (pred_avg - ctrl_prop).tolist(),
            "abs_mse": abs_metrics["mse"], "abs_pcc": abs_metrics["pcc"],
            "abs_r2": abs_metrics["r2"], "abs_spearman": abs_metrics["spearman"],
            "delta_mse": delta_metrics["mse"], "delta_pcc": delta_metrics["pcc"],
            "delta_r2": delta_metrics["r2"], "delta_spearman": delta_metrics["spearman"],
        }

        if use_subclusters and cond in cond_sub_preds and "sub_prop" in gt_entry:
            sub_avg = np.mean(cond_sub_preds[cond], axis=0)
            gt_sub = gt_entry["sub_prop"]
            ctrl_sub = gt_entry["ctrl_sub_prop"]
            row.update({
                "subcluster_labels_order": list(map(str, subcluster_labels)),
                "subcluster_ground_truth": gt_sub.tolist(),
                "subcluster_prediction": sub_avg.tolist(),
                "subcluster_control": ctrl_sub.tolist(),
                "subcluster_mse": float(np.mean((sub_avg - gt_sub) ** 2)),
                "subcluster_delta_pcc": _safe_pcc(sub_avg - ctrl_sub, gt_sub - ctrl_sub),
            })
        rows.append(row)

    # ---- Save results ----
    output_dir = runtime_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_label_payload = {
        "class_labels_order": list(map(str, class_list)),
        "cluster_col": cfg.data.cluster_col,
        "condition_col": cfg.data.condition_col,
        "control_condition": cfg.data.control_condition,
        "cell_embed_key": effective_cell_embed_key,
        "encoder_variant": cfg.data.get("encoder_variant", ""),
    }
    if use_subclusters:
        class_label_payload["subcluster_labels_order"] = list(map(str, subcluster_labels))
        class_label_payload["subcluster_sizes"] = subcluster_sizes
    (output_dir / "class_labels_order.json").write_text(json.dumps(class_label_payload, indent=2))
    (output_dir / "ood_predictions_with_labels.json").write_text(json.dumps(rows, indent=2))

    rows_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not rows_df.empty:
        rows_df.to_csv(output_dir / "ood_predictions_with_labels.csv", index=False)

    aggregate_metrics = {}
    if rows:
        for key in ["abs_mse", "abs_pcc", "abs_r2", "abs_spearman",
                     "delta_mse", "delta_pcc", "delta_r2", "delta_spearman"]:
            vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
            aggregate_metrics[key] = float(np.mean(vals)) if vals else float("nan")
        if use_subclusters:
            for key in ["subcluster_mse", "subcluster_delta_pcc"]:
                vals = [r[key] for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))]
                aggregate_metrics[key] = float(np.mean(vals)) if vals else float("nan")
    (output_dir / "ood_metric_summary.json").write_text(json.dumps(aggregate_metrics, indent=2))

    print("\n=== Aggregate Test Metrics (bootstrap-aggregated) ===")
    for k, v in aggregate_metrics.items():
        print(f"  {k}: {v:.6f}")

    # ---- Wandb logging ----
    conds = [r["ood_condition"] for r in rows]
    table = wandb.Table(columns=[
        "ood_condition", "n_bags", "abs_mse", "abs_pcc", "abs_r2",
        "delta_mse", "delta_pcc", "delta_r2", "interactive_plotly",
    ])
    media_dir = output_dir / "wandb_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    stacked_media = []
    subcluster_media = []
    for r in rows:
        fig_row = go.Figure()
        fig_row.add_bar(name="GT", x=list(class_list), y=r["ood_ground_truth"])
        fig_row.add_bar(name="Pred", x=list(class_list), y=r["ood_prediction"])
        fig_row.update_layout(barmode='group', title=r["ood_condition"], yaxis_title="Proportion", xaxis_tickangle=-45)
        safe_cond = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(r["ood_condition"]))
        if "subcluster_ground_truth" in r and "subcluster_prediction" in r and "subcluster_labels_order" in r:
            fig_sub = _make_grouped_bar_figure(
                labels=r["subcluster_labels_order"],
                gt_prop=r["subcluster_ground_truth"],
                pred_prop=r["subcluster_prediction"],
                condition=f'{r["ood_condition"]} subclusters',
                y_max=max(
                    np.max(np.asarray(r["subcluster_ground_truth"], dtype=float)),
                    np.max(np.asarray(r["subcluster_prediction"], dtype=float)),
                ) * 1.1 + 1e-8,
            )
            sub_png_path = media_dir / f"{safe_cond}_subcluster_grouped.png"
            fig_sub.savefig(sub_png_path, dpi=180, bbox_inches="tight")
            plt.close(fig_sub)
            subcluster_media.append(wandb.Image(str(sub_png_path), caption=f'{r["ood_condition"]} subclusters'))
        table.add_data(
            r["ood_condition"], r["n_bootstrap_bags"],
            r["abs_mse"], r["abs_pcc"], r["abs_r2"],
            r["delta_mse"], r["delta_pcc"], r["delta_r2"],
            wandb.Plotly(fig_row),
        )

    if rows:
        condition_labels = [r["ood_condition"] for r in rows]
        gt_matrix = [r["ood_ground_truth"] for r in rows]
        pred_matrix = [r["ood_prediction"] for r in rows]

        gt_fig = _make_condition_stacked_bar_figure(
            class_list=class_list,
            condition_labels=condition_labels,
            proportion_matrix=gt_matrix,
            title="Ground Truth Proportions Across OOD Conditions",
        )
        gt_png_path = media_dir / "all_conditions_ground_truth_stacked.png"
        gt_fig.savefig(gt_png_path, dpi=240, bbox_inches="tight")
        gt_fig.savefig(media_dir / "all_conditions_ground_truth_stacked.svg", bbox_inches="tight")
        plt.close(gt_fig)

        pred_fig = _make_condition_stacked_bar_figure(
            class_list=class_list,
            condition_labels=condition_labels,
            proportion_matrix=pred_matrix,
            title="Predicted Proportions Across OOD Conditions",
        )
        pred_png_path = media_dir / "all_conditions_predicted_stacked.png"
        pred_fig.savefig(pred_png_path, dpi=240, bbox_inches="tight")
        pred_fig.savefig(media_dir / "all_conditions_predicted_stacked.svg", bbox_inches="tight")
        plt.close(pred_fig)

        stacked_media = [
            wandb.Image(str(gt_png_path), caption="Ground truth stacked bars across all OOD conditions"),
            wandb.Image(str(pred_png_path), caption="Predicted stacked bars across all OOD conditions"),
        ]

    wandb_logger.experiment.summary["validation_table"] = table
    wandb_logger.experiment.summary["class_labels_order"] = list(map(str, class_list))
    test_metric_logs = {}
    for key, value in aggregate_metrics.items():
        wandb_logger.experiment.summary[key] = value
        wandb_logger.experiment.summary[f"test/{key}"] = value
        test_metric_logs[f"test/{key}"] = value
    if test_metric_logs:
        wandb_logger.experiment.log(test_metric_logs)
    if stacked_media:
        wandb_logger.experiment.log({"test/stacked_bar_plots": stacked_media})
    if subcluster_media:
        wandb_logger.experiment.log({"test/subcluster_bar_plots": subcluster_media})
    wandb.finish()

    val_r2 = trainer.callback_metrics.get("val/prop_r2")
    try:
        return float(val_r2.detach().cpu().item()) if val_r2 is not None else float('inf')
    except Exception:
        return float('inf')


if __name__ == "__main__":
    main()
