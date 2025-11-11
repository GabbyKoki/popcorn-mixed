import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, r2_score, pearson_corrcoef


class GatedFusion(nn.Module):
    """
    Gated fusion: fused = cell_emb + gate * pert_emb
    where gate = sigmoid(Linear([cell_emb, pert_emb]))
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, cell_emb, pert_emb):
        # Input: [B, N, D], Output: [B, N, D]
        gate_in = torch.cat([cell_emb, pert_emb], dim=-1)
        gate = self.sigmoid(self.fc(gate_in))
        fused = cell_emb + gate * pert_emb
        return self.layernorm(fused)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: cells attend to perturbation embeddings.
    Q=cell_emb, K=pert_emb, V=pert_emb
    """
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cell_emb, pert_emb, key_padding_mask=None):
        # Input: [B, N, D], Output: [B, N, D]
        attn_output, _ = self.attn(query=cell_emb, key=pert_emb, value=pert_emb, key_padding_mask=key_padding_mask)
        out = self.layernorm(cell_emb + self.proj(attn_output))
        return out


class TransformerMILLightning(LightningModule):
    def __init__(self, input_dim, gene_input_dim, hidden_dim, output_dim, n_heads=4, dim_feedforward=None, lr=1e-3, cell_loss_weight=0.5, num_layers=1, dropout=0.1, fusion_type="add"):
        """
        TransformerMIL model for predicting cell type proportions under perturbation conditions.
        
        Args:
            fusion_type: str, "add", "gated", or "cross_attn"
                - "add": Simple element-wise addition (default, backward compatible)
                - "gated": Gated fusion with learnable gate
                - "cross_attn": Cross-attention fusion
        """
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

        # Initialize fusion module
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

        self.prop_loss_fn = nn.MSELoss()
        self.cell_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, cells, pert_emd, attention_mask, *args, **kwargs):
        """
        Forward pass.
        
        Args:
            cells: [B, N, input_dim] - cell embeddings
            pert_emd: [B, N, gene_input_dim] or [B, gene_input_dim] - perturbation embeddings
            attention_mask: [B, N] - boolean mask (True = valid token)
            *args, **kwargs: ignored (backward compatibility)
        
        Returns:
            proportions: [B, output_dim] - predicted cell type proportions
            cell_logits: [B, N, output_dim] - per-cell classification logits
        """
        B, N, _ = cells.size()
        
        # Encode cells and perturbations
        cell_emb = self.encoder(cells)  # [B, N, hidden_dim]
        
        # Handle both per-cell and per-bag perturbation embeddings
        if pert_emd.dim() == 2:
            pert_emd = pert_emd.unsqueeze(1).expand(-1, N, -1)
        pert_emb = self.gene_encoder(pert_emd)  # [B, N, hidden_dim]
        
        # Fusion: combine cell and perturbation embeddings
        if self.fusion_type == "add":
            combined_emb = cell_emb + pert_emb
        elif self.fusion_type == "gated":
            combined_emb = self.fusion(cell_emb, pert_emb)
        elif self.fusion_type == "cross_attn":
            key_padding_mask = ~attention_mask
            combined_emb = self.fusion(cell_emb, pert_emb, key_padding_mask=key_padding_mask)
        
        # Transformer encoding
        transformer_mask = ~attention_mask
        encoded = self.transformer(combined_emb, src_key_padding_mask=transformer_mask)
        
        # Cell classification and proportion calculation
        cell_logits = self.cell_classifier(encoded)  # [B, N, output_dim]
        cell_probs = torch.softmax(cell_logits, dim=-1)
        
        # Average over valid cells to get bag-level proportions
        valid_probs = cell_probs * attention_mask.unsqueeze(-1).float()
        proportions = valid_probs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return proportions, cell_logits

    def training_step(self, batch, batch_idx):
        # Handle both pure (7 elements) and mixed (6 elements) batch formats
        if len(batch) == 7:
            bags, attn_mask, labels, cell_labels, pert_emd, control_props, conditions = batch
        else:
            bags, attn_mask, labels, cell_labels, pert_emd, control_props = batch
        
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask)

        # Compute losses
        prop_loss = self.prop_loss_fn(prop_pred, labels)
        flat_logits = cell_logits.view(-1, cell_logits.size(-1))
        flat_labels = cell_labels.view(-1)
        cell_loss = self.cell_loss_fn(flat_logits, flat_labels)

        # Combined loss
        loss = (1 - self.hparams.cell_loss_weight) * prop_loss + self.hparams.cell_loss_weight * cell_loss
        
        self.log_dict({
            "train/loss": loss,
            "train/prop_loss": prop_loss,
            "train/cell_loss": cell_loss,
        }, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        bags, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask)

        # Compute metrics per bag (each bag = one perturbation condition)
        batch_size = prop_pred.size(0)
        bag_metrics = []
        
        for i in range(batch_size):
            pred_vec = prop_pred[i]  # [C]
            target_vec = target_proportions[i]  # [C]
            
            bag_mse = F.mse_loss(pred_vec, target_vec)
            bag_r2 = r2_score(pred_vec, target_vec)
            bag_pcc = pearson_corrcoef(pred_vec, target_vec)
            
            bag_metrics.append({'mse': bag_mse, 'r2': bag_r2, 'pcc': bag_pcc})
        
        # Average metrics across bags
        avg_mse = torch.stack([m['mse'] for m in bag_metrics]).mean()
        avg_r2 = torch.stack([m['r2'] for m in bag_metrics]).mean()
        avg_pcc = torch.stack([m['pcc'] for m in bag_metrics]).mean()

        self.log_dict({
            "val/prop_mse": avg_mse,
            "val/prop_r2": avg_r2,
            "val/prop_pcc": avg_pcc,
        }, prog_bar=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        bags, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask)

        # Compute metrics per bag (each bag = one perturbation condition)
        batch_size = prop_pred.size(0)
        bag_metrics = []
        
        for i in range(batch_size):
            pred_vec = prop_pred[i]  # [C]
            target_vec = target_proportions[i]  # [C]
            
            bag_mse = F.mse_loss(pred_vec, target_vec)
            bag_r2 = r2_score(pred_vec, target_vec)
            bag_pcc = pearson_corrcoef(pred_vec, target_vec)
            
            bag_metrics.append({'mse': bag_mse, 'r2': bag_r2, 'pcc': bag_pcc})
        
        # Average metrics across bags
        avg_mse = torch.stack([m['mse'] for m in bag_metrics]).mean()
        avg_r2 = torch.stack([m['r2'] for m in bag_metrics]).mean()
        avg_pcc = torch.stack([m['pcc'] for m in bag_metrics]).mean()

        self.log_dict({
            "test/prop_mse": avg_mse,
            "test/prop_r2": avg_r2,
            "test/prop_pcc": avg_pcc,
        }, prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)