import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, r2_score, pearson_corrcoef


class EmbeddingMeanTransformerMILLightning(LightningModule):
    def __init__(self, input_dim, gene_input_dim, hidden_dim, output_dim, n_heads=4, dim_feedforward=None, lr=1e-3, cell_loss_weight=0.5, num_layers=1, dropout=0.1, prediction_mode="absolute"):
        """
        Embedding Mean MIL model based on TransformerMIL structure.
        
        Key differences from the original model:
        - Uses Transformer to encode cells into embeddings
        - Performs mean aggregation at the embedding level (not probability level)
        - Predicts proportions directly from the averaged embedding
        - Maintains cell-level classification for auxiliary supervision
        - Uses both cell loss and proportion loss
        
        Architecture:
        1. Cell embedding + perturbation embedding → combined embedding
        2. Transformer encoder for context modeling
        3. Mean aggregation of encoded embeddings → bag embedding
        4. Direct proportion prediction from bag embedding
        5. Cell-level classification on individual embeddings (for auxiliary loss)
        
        Args:
            prediction_mode: str, "absolute" or "residual"
                - "absolute": Directly predict proportions under perturbation conditions (p_pert)
                - "residual": Predict change amount Δp, then add to p_control to get p_pert
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

        # Use provided dim_feedforward or default to hidden_dim * 2
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 2
            
        self.save_hyperparameters()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cell classifier (same as original model, for auxiliary supervision)
        self.cell_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Proportion predictor that works on bag embedding (averaged cell embeddings)
        self.proportion_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Ensure proportions sum to 1
        )

        self.prop_loss_fn = nn.MSELoss()
        self.cell_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, cells, pert_emd, attention_mask, control_prop=None):
        B, N, _ = cells.size()  # B: batch size, N: max sequence length
        
        # Encode cells
        cell_emb = self.encoder(cells)  # [B, N, hidden_dim]
        
        # Encode perturbation embeddings for each cell
        pert_emb = self.gene_encoder(pert_emd)  # [B, N, hidden_dim]
        
        # Combine cell embeddings with perturbation embeddings
        combined_emb = cell_emb + pert_emb  # Element-wise addition
        
        # Apply transformer to the combined embeddings
        transformer_mask = ~attention_mask
        encoded = self.transformer(combined_emb, src_key_padding_mask=transformer_mask)
        
        # Mean aggregation at embedding level (only valid embeddings)
        # Apply attention mask to only consider valid (non-padded) embeddings
        valid_embeddings = encoded * attention_mask.unsqueeze(-1).float()  # [B, N, hidden_dim]
        bag_embedding = valid_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # [B, hidden_dim]
        
        # Predict proportions directly from bag embedding
        proportions = self.proportion_predictor(bag_embedding)  # [B, output_dim]
        
        # Cell classification (for auxiliary supervision)
        cell_logits = self.cell_classifier(encoded)  # [B, N, output_dim]
        
        # Handle absolute vs residual prediction modes
        if self.hparams.prediction_mode == "absolute":
            output_prop = proportions
        elif self.hparams.prediction_mode == "residual":
            if control_prop is None:
                raise ValueError("control_prop must be provided in residual mode")
            output_prop = control_prop + proportions  # proportions as delta
        else:
            raise ValueError(f"Unknown prediction_mode: {self.hparams.prediction_mode}")
        
        return output_prop, cell_logits

    def training_step(self, batch, batch_idx):
        bags, attn_mask, labels, cell_labels, pert_emd, control_props = batch
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask, control_props)

        # Proportion loss
        prop_loss = self.prop_loss_fn(prop_pred, labels)
        
        # Cell loss (auxiliary supervision)
        flat_logits = cell_logits.view(-1, cell_logits.size(-1))
        flat_labels = cell_labels.view(-1)
        cell_loss = self.cell_loss_fn(flat_logits, flat_labels)

        # Combined loss
        loss = (1-self.hparams.cell_loss_weight) * prop_loss + self.hparams.cell_loss_weight * cell_loss
        
        self.log_dict({
            "train/loss": loss,
            "train/prop_loss": prop_loss,
            "train/cell_loss": cell_loss,
        }, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        bags, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask, control_props)

        # Proportion metrics per bag
        batch_size = prop_pred.size(0)
        bag_metrics = []
        
        for i in range(batch_size):
            pred_vec = prop_pred[i]  # [C] - predicted proportions for this bag
            target_vec = target_proportions[i]  # [C] - true proportions for this bag
            
            # Compute per-bag metrics
            bag_mse = F.mse_loss(pred_vec, target_vec)
            
            # R2 score for this bag (vector comparison)
            bag_r2 = r2_score(pred_vec, target_vec)
            
            # Pearson correlation for this bag (vector comparison)
            bag_pcc = pearson_corrcoef(pred_vec, target_vec)
            
            bag_metrics.append({
                'mse': bag_mse,
                'r2': bag_r2,
                'pcc': bag_pcc
            })
        
        # Aggregate metrics across bags
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
        prop_pred, cell_logits = self(bags, pert_emd, attn_mask, control_props)

        # Proportion metrics per bag
        batch_size = prop_pred.size(0)
        bag_metrics = []
        
        for i in range(batch_size):
            pred_vec = prop_pred[i]  # [C] - predicted proportions for this bag
            target_vec = target_proportions[i]  # [C] - true proportions for this bag
            
            # Compute per-bag metrics
            bag_mse = F.mse_loss(pred_vec, target_vec)
            
            # R2 score for this bag (vector comparison)
            bag_r2 = r2_score(pred_vec, target_vec)
            
            # Pearson correlation for this bag (vector comparison)
            bag_pcc = pearson_corrcoef(pred_vec, target_vec)
            
            bag_metrics.append({
                'mse': bag_mse,
                'r2': bag_r2,
                'pcc': bag_pcc
            })
        
        # Aggregate metrics across bags
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
