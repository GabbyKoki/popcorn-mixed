import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, r2_score, pearson_corrcoef


class CLSTokenTransformerMILLightning(LightningModule):
    def __init__(self, input_dim, gene_input_dim, hidden_dim, output_dim, n_heads=4, dim_feedforward=None, lr=1e-3, cell_loss_weight=0.5, num_layers=1, dropout=0.1, prediction_mode="absolute"):
        """
        CLS Token MIL model based on TransformerMIL structure.
        
        Key differences from the original model:
        - Adds a learnable CLS token at the beginning of the sequence
        - Uses Transformer self-attention to let CLS token aggregate information from all cells
        - Directly predicts proportions from the CLS token representation
        - Maintains cell-level classification for auxiliary supervision
        - Uses both cell loss and proportion loss
        
        Architecture:
        1. Cell embedding + perturbation embedding → combined embedding
        2. Prepend learnable CLS token to the sequence
        3. Transformer encoder with self-attention (CLS token attends to all cells)
        4. Cell-level classification on cell tokens (for auxiliary loss)
        5. Direct proportion prediction from CLS token representation
        
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

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
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

        # Proportion predictor that works on CLS token
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
        
        # Prepend CLS token to the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        sequence_with_cls = torch.cat([cls_tokens, combined_emb], dim=1)  # [B, N+1, hidden_dim]
        
        # Create attention mask for the sequence with CLS token
        # CLS token is always valid (True), so prepend True to attention_mask
        cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [B, N+1]
        
        # Apply transformer to the sequence (CLS + cells)
        transformer_mask = ~extended_attention_mask
        encoded_with_cls = self.transformer(sequence_with_cls, src_key_padding_mask=transformer_mask)
        
        # Split CLS token and cell representations
        cls_representation = encoded_with_cls[:, 0, :]  # [B, hidden_dim] - CLS token
        cell_representations = encoded_with_cls[:, 1:, :]  # [B, N, hidden_dim] - cell tokens
        
        # Predict proportions directly from CLS token
        proportions = self.proportion_predictor(cls_representation)  # [B, output_dim]
        
        # Cell classification (for auxiliary supervision)
        cell_logits = self.cell_classifier(cell_representations)  # [B, N, output_dim]
        
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
        bags, _pert_cells, attn_mask, labels, cell_labels, pert_emd, control_props = batch
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
        bags, _pert_cells, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
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
        bags, _pert_cells, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
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
