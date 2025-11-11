import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, r2_score, pearson_corrcoef


class AttentionMILLightning(LightningModule):
    def __init__(self, input_dim, gene_input_dim, hidden_dim, output_dim, n_heads=4, dim_feedforward=None, lr=1e-3, num_layers=1, dropout=0.1, prediction_mode="absolute"):
        """
        Classic Attention-MIL model following Ilse et al. (2018) approach.
        
        Key differences from the original model:
        - No cell-level classification
        - Uses classic attention-MIL: MLP scoring + weighted sum aggregation
        - Each instance (cell) gets an attention score, then weighted sum to bag representation
        - Only proportion loss is used (no cell loss)
        
        Architecture:
        1. Cell embedding + perturbation embedding → combined embedding
        2. Transformer encoder for context modeling
        3. Instance-level attention scoring (MLP: hidden_dim → hidden_dim → 1)
        4. Softmax attention weights + weighted sum aggregation
        5. Direct proportion prediction from bag representation
        
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

        # Classic attention-MIL: instance-level attention scoring
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Direct proportion predictor
        self.proportion_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Ensure proportions sum to 1
        )

        self.prop_loss_fn = nn.MSELoss()

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
        
        # Classic attention-MIL aggregation
        # Compute attention scores for each cell
        attention_scores = self.attention(encoded)  # [B, N, 1]
        attention_scores = attention_scores.squeeze(-1)  # [B, N]
        
        # Apply attention mask (set padded positions to very negative values)
        attention_scores = attention_scores.masked_fill(transformer_mask, -1e9)
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, N]
        
        # Weighted sum of cell embeddings to get bag representation
        bag_representation = torch.sum(
            attention_weights.unsqueeze(-1) * encoded,  # [B, N, 1] * [B, N, hidden_dim]
            dim=1  # Sum over sequence dimension
        )  # [B, hidden_dim]
        
        # Predict proportions directly from bag representation
        proportions = self.proportion_predictor(bag_representation)  # [B, output_dim]
        
        # Handle absolute vs residual prediction modes
        if self.hparams.prediction_mode == "absolute":
            output_prop = proportions
        elif self.hparams.prediction_mode == "residual":
            if control_prop is None:
                raise ValueError("control_prop must be provided in residual mode")
            output_prop = control_prop + proportions  # proportions as delta
        else:
            raise ValueError(f"Unknown prediction_mode: {self.hparams.prediction_mode}")
        
        return output_prop, attention_weights

    def training_step(self, batch, batch_idx):
        bags, attn_mask, labels, cell_labels, pert_emd, control_props = batch
        prop_pred, attention_weights = self(bags, pert_emd, attn_mask, control_props)

        # Only proportion loss (no cell loss in attention-only model)
        prop_loss = self.prop_loss_fn(prop_pred, labels)

        self.log_dict({
            "train/loss": prop_loss,
            "train/prop_loss": prop_loss,
        }, prog_bar=False, on_step=True, on_epoch=True)

        return prop_loss

    def validation_step(self, batch, batch_idx):
        bags, attn_mask, target_proportions, cell_labels, pert_emd, control_props = batch
        prop_pred, attention_weights = self(bags, pert_emd, attn_mask, control_props)

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
        prop_pred, attention_weights = self(bags, pert_emd, attn_mask, control_props)

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
