import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, r2_score, pearson_corrcoef
import numpy as np


def _safe_pearson(pred_vec, target_vec):
    if torch.std(pred_vec) == 0 or torch.std(target_vec) == 0:
        return torch.tensor(float("nan"), device=pred_vec.device)
    return pearson_corrcoef(pred_vec, target_vec)


def _safe_r2(pred_vec, target_vec):
    try:
        return r2_score(pred_vec, target_vec)
    except Exception:
        return torch.tensor(float("nan"), device=pred_vec.device)


def _safe_spearman(pred_vec, target_vec):
    pred_np = pred_vec.detach().cpu().numpy().astype(float)
    target_np = target_vec.detach().cpu().numpy().astype(float)
    if np.std(pred_np) == 0.0 or np.std(target_np) == 0.0:
        return torch.tensor(float("nan"), device=pred_vec.device)
    pred_rank = np.argsort(np.argsort(pred_np))
    target_rank = np.argsort(np.argsort(target_np))
    corr = np.corrcoef(pred_rank, target_rank)[0, 1]
    return torch.tensor(float(corr), device=pred_vec.device)


def _bag_metric_dict(pred_vec, target_vec):
    return {
        "mse": F.mse_loss(pred_vec, target_vec),
        "r2": _safe_r2(pred_vec, target_vec),
        "pcc": _safe_pearson(pred_vec, target_vec),
        "spearman": _safe_spearman(pred_vec, target_vec),
    }


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, cell_emb, pert_emb):
        gate_in = torch.cat([cell_emb, pert_emb], dim=-1)
        gate = self.sigmoid(self.fc(gate_in))
        fused = cell_emb + gate * pert_emb
        return self.layernorm(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cell_emb, pert_emb, key_padding_mask=None):
        attn_output, _ = self.attn(query=cell_emb, key=pert_emb, value=pert_emb, key_padding_mask=key_padding_mask)
        out = self.layernorm(cell_emb + self.proj(attn_output))
        return out


def _mmd_rbf(x, y, sigma=None):
    if sigma is None:
        with torch.no_grad():
            dists = torch.cdist(x, y, p=2)
            sigma = dists.median().clamp_min(1e-5)
    gamma = 1.0 / (2.0 * sigma ** 2)
    xx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2))
    yy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2))
    xy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2))
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def _kl_loss(pred, target, eps=1e-8):
    target_safe = target.clamp(min=eps)
    pred_safe = pred.clamp(min=eps)
    return (target_safe * (target_safe.log() - pred_safe.log())).sum(dim=-1).mean()


class TransformerMILLightning(LightningModule):
    def __init__(
        self,
        input_dim,
        gene_input_dim,
        hidden_dim,
        output_dim,
        n_heads=4,
        dim_feedforward=None,
        lr=1e-3,
        cell_loss_weight=0.5,
        num_layers=1,
        dropout=0.1,
        fusion_type="add",
        prop_loss_type="mse",
        mmd_sigma=None,
        subcluster_sizes=None,
        subcluster_prop_loss_weight=0.0,
        subcluster_cell_loss_weight=0.0,
        type_prop_loss_weight=1.0,
        **unused_kwargs,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 2

        self.save_hyperparameters()

        assert fusion_type in ("add", "gated", "cross_attn"), f"Unknown fusion_type: {fusion_type}"
        self.fusion_type = fusion_type
        if fusion_type == "gated":
            self.fusion = GatedFusion(hidden_dim)
        elif fusion_type == "cross_attn":
            self.fusion = CrossAttentionFusion(hidden_dim, n_heads=n_heads, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cell_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.subcluster_sizes = list(subcluster_sizes or [])
        self.use_subclusters = bool(self.subcluster_sizes and sum(self.subcluster_sizes) > 0)
        if self.use_subclusters:
            self.subcluster_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, sum(self.subcluster_sizes))
            )
            subcluster_type_idx = []
            for type_idx, size in enumerate(self.subcluster_sizes):
                subcluster_type_idx.extend([type_idx] * int(size))
            self.register_buffer("subcluster_type_idx", torch.tensor(subcluster_type_idx, dtype=torch.long), persistent=False)
        else:
            self.subcluster_head = None
            self.register_buffer("subcluster_type_idx", torch.empty(0, dtype=torch.long), persistent=False)

        assert prop_loss_type in ("mse", "kl", "mmd"), f"Unknown prop_loss_type: {prop_loss_type}"
        self.prop_loss_type = prop_loss_type
        self.prop_loss_fn = nn.MSELoss()
        self.cell_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _block_softmax(self, subcluster_logits):
        probs = torch.zeros_like(subcluster_logits)
        start = 0
        for size in self.subcluster_sizes:
            end = start + int(size)
            if end > start:
                probs[..., start:end] = torch.softmax(subcluster_logits[..., start:end], dim=-1)
            start = end
        return probs

    def _aggregate_subcluster_props(self, type_probs, subcluster_probs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        type_mass = (type_probs * mask).sum(dim=1).clamp_min(1e-8)
        joint = subcluster_probs * type_probs[..., self.subcluster_type_idx]
        numer = (joint * mask).sum(dim=1)
        denom = type_mass[:, self.subcluster_type_idx]
        return numer / denom.clamp_min(1e-8)

    def forward(self, cells, pert_emd, attention_mask, *args, **kwargs):
        _, n_tokens, _ = cells.size()
        cell_emb = self.encoder(cells)

        if pert_emd.dim() == 2:
            pert_emd = pert_emd.unsqueeze(1).expand(-1, n_tokens, -1)
        pert_emb = self.gene_encoder(pert_emd)

        if self.fusion_type == "add":
            combined_emb = cell_emb + pert_emb
        elif self.fusion_type == "gated":
            combined_emb = self.fusion(cell_emb, pert_emb)
        elif self.fusion_type == "cross_attn":
            combined_emb = self.fusion(cell_emb, pert_emb, key_padding_mask=~attention_mask)

        encoded = self.transformer(combined_emb, src_key_padding_mask=~attention_mask)
        cell_logits = self.cell_classifier(encoded)
        cell_probs = torch.softmax(cell_logits, dim=-1)
        valid_probs = cell_probs * attention_mask.unsqueeze(-1).float()
        proportions = valid_probs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        if self.use_subclusters:
            subcluster_logits = self.subcluster_head(encoded)
            subcluster_probs = self._block_softmax(subcluster_logits)
            subcluster_props = self._aggregate_subcluster_props(cell_probs, subcluster_probs, attention_mask)
            return proportions, cell_logits, subcluster_props, subcluster_logits
        return proportions, cell_logits

    def _compute_prop_loss(self, prop_pred, labels):
        if self.prop_loss_type == "kl":
            return _kl_loss(prop_pred, labels)
        elif self.prop_loss_type == "mmd":
            return _mmd_rbf(prop_pred, labels, sigma=self.hparams.mmd_sigma)
        else:
            return self.prop_loss_fn(prop_pred, labels)

    def _compute_subcluster_prop_loss(self, sub_pred, sub_target, type_weights):
        losses = []
        start = 0
        for type_idx, size in enumerate(self.subcluster_sizes):
            end = start + int(size)
            if end > start:
                block_loss = F.mse_loss(sub_pred[:, start:end], sub_target[:, start:end], reduction="none").mean(dim=1)
                losses.append(block_loss * type_weights[:, type_idx])
            start = end
        if not losses:
            return torch.zeros((), device=sub_pred.device)
        stacked = torch.stack(losses, dim=1)
        denom = type_weights.sum(dim=1).clamp_min(1e-8)
        return (stacked.sum(dim=1) / denom).mean()

    def _unpack_batch(self, batch):
        has_conditions = isinstance(batch[-1], (tuple, list))
        if len(batch) in (9, 10):
            if has_conditions:
                bags, attn_mask, labels, cell_labels, pert_emd, control_props, sub_props, cell_sublabels, control_sub_props, _conditions = batch
            else:
                bags, attn_mask, labels, cell_labels, pert_emd, control_props, sub_props, cell_sublabels, control_sub_props = batch
            return {
                "bags": bags, "attn_mask": attn_mask, "labels": labels,
                "cell_labels": cell_labels, "pert_emd": pert_emd,
                "control_props": control_props, "sub_props": sub_props,
                "cell_sublabels": cell_sublabels, "control_sub_props": control_sub_props,
            }
        if has_conditions:
            bags, attn_mask, labels, cell_labels, pert_emd, control_props, _conditions = batch
        else:
            bags, attn_mask, labels, cell_labels, pert_emd, control_props = batch
        return {
            "bags": bags, "attn_mask": attn_mask, "labels": labels,
            "cell_labels": cell_labels, "pert_emd": pert_emd,
            "control_props": control_props,
            "sub_props": None, "cell_sublabels": None, "control_sub_props": None,
        }

    def _shared_step(self, batch, stage):
        parts = self._unpack_batch(batch)
        outputs = self(parts["bags"], parts["pert_emd"], parts["attn_mask"])
        if self.use_subclusters:
            prop_pred, cell_logits, sub_pred, sub_logits = outputs
        else:
            prop_pred, cell_logits = outputs
            sub_pred = sub_logits = None
        prop_loss = self._compute_prop_loss(prop_pred, parts["labels"])
        cell_loss = self.cell_loss_fn(cell_logits.view(-1, cell_logits.size(-1)), parts["cell_labels"].view(-1))
        loss = self.hparams.type_prop_loss_weight * (1 - self.hparams.cell_loss_weight) * prop_loss
        loss = loss + self.hparams.cell_loss_weight * cell_loss
        logs = {f"{stage}/loss": loss, f"{stage}/prop_loss": prop_loss, f"{stage}/cell_loss": cell_loss}
        if self.use_subclusters and parts["sub_props"] is not None:
            sub_prop_loss = self._compute_subcluster_prop_loss(sub_pred, parts["sub_props"], parts["labels"])
            sub_cell_loss = self.cell_loss_fn(sub_logits.view(-1, sub_logits.size(-1)), parts["cell_sublabels"].view(-1))
            loss = loss + self.hparams.subcluster_prop_loss_weight * sub_prop_loss
            loss = loss + self.hparams.subcluster_cell_loss_weight * sub_cell_loss
            logs[f"{stage}/loss"] = loss
            logs[f"{stage}/subcluster_prop_loss"] = sub_prop_loss
            logs[f"{stage}/subcluster_cell_loss"] = sub_cell_loss
        return loss, logs, prop_pred, sub_pred, parts

    def training_step(self, batch, batch_idx):
        loss, logs, _, _, parts = self._shared_step(batch, "train")
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=parts["bags"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, logs, prop_pred, sub_pred, parts = self._shared_step(batch, "val")
        target_proportions = parts["labels"]
        control_props = parts["control_props"]
        abs_metrics, delta_metrics = [], []
        for i in range(prop_pred.size(0)):
            abs_metrics.append(_bag_metric_dict(prop_pred[i], target_proportions[i]))
            delta_metrics.append(_bag_metric_dict(prop_pred[i] - control_props[i], target_proportions[i] - control_props[i]))
        metric_logs = {
            "val/prop_mse": torch.stack([m['mse'] for m in abs_metrics]).mean(),
            "val/prop_r2": torch.stack([m['r2'] for m in abs_metrics]).mean(),
            "val/prop_pcc": torch.stack([m['pcc'] for m in abs_metrics]).mean(),
            "val/prop_spearman": torch.nanmean(torch.stack([m['spearman'] for m in abs_metrics])),
            "val/delta_mse": torch.stack([m['mse'] for m in delta_metrics]).mean(),
            "val/delta_r2": torch.stack([m['r2'] for m in delta_metrics]).mean(),
            "val/delta_pcc": torch.stack([m['pcc'] for m in delta_metrics]).mean(),
            "val/delta_spearman": torch.nanmean(torch.stack([m['spearman'] for m in delta_metrics])),
        }
        metric_logs.update({k: v for k, v in logs.items() if k != "val/loss"})
        if sub_pred is not None and parts["sub_props"] is not None:
            metric_logs["val/subcluster_mse"] = F.mse_loss(sub_pred, parts["sub_props"])
            metric_logs["val/subcluster_delta_pcc"] = _safe_pearson(
                (sub_pred - parts["control_sub_props"]).flatten(),
                (parts["sub_props"] - parts["control_sub_props"]).flatten(),
            )
        self.log_dict(metric_logs, prog_bar=True, on_epoch=True, logger=True, batch_size=parts["bags"].size(0))

    def test_step(self, batch, batch_idx):
        _, logs, prop_pred, sub_pred, parts = self._shared_step(batch, "test")
        target_proportions = parts["labels"]
        control_props = parts["control_props"]
        abs_metrics, delta_metrics = [], []
        for i in range(prop_pred.size(0)):
            abs_metrics.append(_bag_metric_dict(prop_pred[i], target_proportions[i]))
            delta_metrics.append(_bag_metric_dict(prop_pred[i] - control_props[i], target_proportions[i] - control_props[i]))
        metric_logs = {
            "test/prop_mse": torch.stack([m['mse'] for m in abs_metrics]).mean(),
            "test/prop_r2": torch.stack([m['r2'] for m in abs_metrics]).mean(),
            "test/prop_pcc": torch.stack([m['pcc'] for m in abs_metrics]).mean(),
            "test/prop_spearman": torch.nanmean(torch.stack([m['spearman'] for m in abs_metrics])),
            "test/delta_mse": torch.stack([m['mse'] for m in delta_metrics]).mean(),
            "test/delta_r2": torch.stack([m['r2'] for m in delta_metrics]).mean(),
            "test/delta_pcc": torch.stack([m['pcc'] for m in delta_metrics]).mean(),
            "test/delta_spearman": torch.nanmean(torch.stack([m['spearman'] for m in delta_metrics])),
        }
        metric_logs.update({k: v for k, v in logs.items() if k != "test/loss"})
        if sub_pred is not None and parts["sub_props"] is not None:
            metric_logs["test/subcluster_mse"] = F.mse_loss(sub_pred, parts["sub_props"])
            metric_logs["test/subcluster_delta_pcc"] = _safe_pearson(
                (sub_pred - parts["control_sub_props"]).flatten(),
                (parts["sub_props"] - parts["control_sub_props"]).flatten(),
            )
        self.log_dict(metric_logs, prog_bar=True, on_epoch=True, logger=True, batch_size=parts["bags"].size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
