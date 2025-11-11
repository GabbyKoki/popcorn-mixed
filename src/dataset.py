import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from pert_bags import PertBag
from bag_sampler import BootstrapBagSampler


def collate_batch(batch):
    """Collate function for mixed condition (per-cell perturbation embeddings)."""
    bags, labels, cell_labels, pert_emd, control_props = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)
    padded_cell_labels = pad_sequence(cell_labels, batch_first=True, padding_value=-1)
    padded_pert_emd = pad_sequence(pert_emd, batch_first=True)  # Each cell has its own pert embedding
    attention_mask = torch.tensor([[1] * len(b) + [0] * (padded_bags.size(1) - len(b)) for b in bags])
    labels = torch.stack(labels)
    control_props = torch.stack(control_props)
    return padded_bags, attention_mask.bool(), labels, padded_cell_labels, padded_pert_emd, control_props


def collate_pure_batch(batch):
    """Collate function for pure condition (single perturbation embedding per bag)."""
    bags, labels, cell_labels, pert_emd, control_props, conditions = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)
    padded_cell_labels = pad_sequence(cell_labels, batch_first=True, padding_value=-1)
    # pert_emd is already a single vector per bag: [embedding_dim]
    pert_emd = torch.stack(pert_emd)  # [B, embedding_dim]
    attention_mask = torch.tensor([[1] * len(b) + [0] * (padded_bags.size(1) - len(b)) for b in bags])
    labels = torch.stack(labels)
    control_props = torch.stack(control_props)
    return padded_bags, attention_mask.bool(), labels, padded_cell_labels, pert_emd, control_props, conditions


class PerturbationDataset(Dataset): # not used
    """Dataset that performs runtime sampling with mixed perturbations."""
    
    def __init__(self, pert_bags, adata, cluster_col, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X"):
        """
        Dataset that performs sampling on-demand in __getitem__ with mixed perturbation support.
        
        Args:
            pert_bags: List of PertBag objects
            adata: AnnData object containing cell data
            cluster_col: Column name for cluster information
            class_list: List of all possible cluster classes
            pert_embeddings: Dictionary mapping perturbation names to embeddings
            sampler: BootstrapBagSampler instance for sampling
            seed: Random seed for reproducible sampling
            cell_embed_key: Key for cell embeddings in adata (default "X")
        """
        self.pert_bags = pert_bags
        self.adata = adata
        self.cluster_col = cluster_col
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        
        # Set up cell embeddings
        self.X = adata.X if cell_embed_key == "X" else adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        
        # Create RNG for consistent sampling per epoch
        self.rng = np.random.RandomState(seed)
        
        # Calculate total number of samples across all bags
        self.total_samples = sum(self.sampler.n_bags for _ in self.pert_bags)
        
        # Create mapping from global index to (bag_idx, sample_idx)
        self.index_mapping = []
        for bag_idx, _ in enumerate(self.pert_bags):
            for sample_idx in range(self.sampler.n_bags):
                self.index_mapping.append((bag_idx, sample_idx))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        bag_idx, sample_idx = self.index_mapping[idx]
        pert_bag = self.pert_bags[bag_idx]
        
        # Get indices for this perturbation bag
        pert_idx, ctrl_idx = pert_bag.get_matched_indices()
        pair_idx = np.arange(len(pert_idx))
        
        # Create a deterministic RNG state based on bag_idx and sample_idx
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + bag_idx * 1000 + sample_idx)
        
        # Sample pairs for this specific sample
        sampled_pair_idx = self.sampler.sample(pair_idx, rng=sample_rng)[0]  # Get first sample
        
        # Get sampled indices
        pert_s = pert_idx[sampled_pair_idx]
        ctrl_s = ctrl_idx[sampled_pair_idx]
        
        # Get target labels and calculate proportions
        all_labels = pert_bag.get_perturbation_cluster_labels()
        target_labels = all_labels[sampled_pair_idx]
        
        # Calculate proportions using consistent class_list
        label_series = pd.Series(target_labels, dtype="category")
        label_series = label_series.cat.set_categories(self.class_list)
        proportions = label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Calculate control proportions (for residual mode)
        ctrl_labels = self.adata.obs.loc[ctrl_s][self.cluster_col].values
        ctrl_label_series = pd.Series(ctrl_labels, dtype="category")
        ctrl_label_series = ctrl_label_series.cat.set_categories(self.class_list)
        control_proportions = ctrl_label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Get control cell embeddings
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_s)], dtype=torch.float32)
        
        # Get cluster indices for each cell
        cluster_idx = [self.class_list.index(c) for c in target_labels]
        cluster_idx = torch.tensor(cluster_idx, dtype=torch.long)
        
        # Create perturbation embeddings for each cell (supporting mixed perturbations)
        pert_emb_list = []
        for i, cell_idx in enumerate(pert_s):
            # For now, all cells in the same bag have the same perturbation
            # But this structure supports mixed perturbations in the future
            gene = pert_bag.perturbation_name.split("+")[0]
            if gene == "ctrl":
                pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
            else:
                pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
            pert_emb_list.append(pert_emb)
        
        pert_emb_tensor = torch.stack(pert_emb_list)  # Shape: [n_cells, embedding_dim]
        
        return X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32)


class MixedConditionDataset(Dataset):
    """Dataset that samples from specified conditions with runtime sampling."""
    
    def __init__(self, bag_creator, sampling_conditions, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X", use_matched_control=True):
        """
        Dataset that samples from specified conditions (supports both OOD and IID modes).
        
        Args:
            bag_creator: BagCreator instance with computed matching index
            sampling_conditions: List of conditions to sample from during training
            class_list: List of all possible cluster classes
            pert_embeddings: Dictionary mapping perturbation names to embeddings
            sampler: BootstrapBagSampler instance for sampling
            seed: Random seed for reproducible sampling
            cell_embed_key: Key for cell embeddings in adata (default "X")
            use_matched_control: If True, use OT-matched control cells; if False, randomly sample control cells
        """
        self.bag_creator = bag_creator
        self.adata = bag_creator.adata
        self.sampling_conditions = sampling_conditions
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        self.use_matched_control = use_matched_control
        
        # Set up cell embeddings
        self.X = self.adata.X if cell_embed_key == "X" else self.adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        
        # Get all matched data for the specified sampling conditions
        self.matched_data = self._get_filtered_matched_data()
        
        # Get all control cell indices for random sampling
        self.all_control_idx = self.adata.obs[self.adata.obs[self.bag_creator.condition_col] == self.bag_creator.control_condition].index
        
        # Create RNG for consistent sampling per epoch
        self.rng = np.random.RandomState(seed)
        
        # Calculate total number of samples
        self.total_samples = self.sampler.n_bags
        
    def _get_filtered_matched_data(self):
        """Get matched data filtered by sampling conditions."""
        all_data = self.bag_creator.get_all_matched_data()
        
        # Filter by sampling conditions
        mask = np.array([cond in self.sampling_conditions for cond in all_data['conditions']])
        
        return {
            'pert_cells': all_data['pert_cells'][mask],
            'ctrl_cells': all_data['ctrl_cells'][mask],
            'conditions': np.array(all_data['conditions'])[mask].tolist(),
            'target_labels': np.array(all_data['target_labels'])[mask]
        }

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Create a deterministic RNG state based on idx
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        
        # Sample from all available perturbation cells
        n_total = len(self.matched_data['pert_cells'])
        pair_indices = np.arange(n_total)
        
        # Sample pairs for this specific sample
        sampled_pair_idx = self.sampler.sample(pair_indices, rng=sample_rng)[0]  # Get first sample
        
        # Get sampled perturbation cells
        pert_cells = self.matched_data['pert_cells'][sampled_pair_idx]
        conditions = np.array(self.matched_data['conditions'])[sampled_pair_idx]
        target_labels = self.matched_data['target_labels'][sampled_pair_idx]
        
        # Get control cells: either matched or randomly sampled
        if self.use_matched_control:
            ctrl_cells = self.matched_data['ctrl_cells'][sampled_pair_idx]
        else:
            # Randomly sample control cells for each perturbation cell
            n_cells = len(pert_cells)
            ctrl_cells = pd.Index(sample_rng.choice(self.all_control_idx, size=n_cells, replace=True))
        
        # Calculate proportions using consistent class_list
        label_series = pd.Series(target_labels, dtype="category")
        label_series = label_series.cat.set_categories(self.class_list)
        proportions = label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Calculate control proportions (for residual mode)
        ctrl_labels = self.adata.obs.loc[ctrl_cells][self.bag_creator.cluster_col].values
        ctrl_label_series = pd.Series(ctrl_labels, dtype="category")
        ctrl_label_series = ctrl_label_series.cat.set_categories(self.class_list)
        control_proportions = ctrl_label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Get control cell embeddings
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_cells)], dtype=torch.float32)
        
        # Get cluster indices for each cell
        cluster_idx = [self.class_list.index(c) for c in target_labels]
        cluster_idx = torch.tensor(cluster_idx, dtype=torch.long)
        
        # Create perturbation embeddings for each cell
        pert_emb_list = []
        for condition in conditions:
            gene = condition.split("+")[0]
            if gene == "ctrl":
                pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
            else:
                pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
            pert_emb_list.append(pert_emb)
        
        pert_emb_tensor = torch.stack(pert_emb_list)  # Shape: [n_cells, embedding_dim]
        
        return X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32)


class PureConditionDataset(Dataset):
    """
    Dataset for pure condition: each bag contains cells from a single perturbation condition.
    All cells in a bag share the same perturbation embedding.
    Suitable for cross-attention fusion.
    """
    
    def __init__(self, bag_creator, sampling_conditions, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X", use_matched_control=True):
        """
        Dataset that creates bags with single perturbation condition (pure mode).
        
        Args:
            bag_creator: BagCreator instance with computed matching index
            sampling_conditions: List of conditions to sample from during training
            class_list: List of all possible cluster classes
            pert_embeddings: Dictionary mapping perturbation names to embeddings
            sampler: BootstrapBagSampler instance for sampling
            seed: Random seed for reproducible sampling
            cell_embed_key: Key for cell embeddings in adata (default "X")
            use_matched_control: If True, use OT-matched control cells; if False, randomly sample control cells
        """
        self.bag_creator = bag_creator
        self.adata = bag_creator.adata
        self.sampling_conditions = sampling_conditions
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        self.use_matched_control = use_matched_control
        
        # Set up cell embeddings
        self.X = self.adata.X if cell_embed_key == "X" else self.adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        
        # Get all control cell indices for random sampling
        self.all_control_idx = self.adata.obs[self.adata.obs[self.bag_creator.condition_col] == self.bag_creator.control_condition].index
        
        # Create condition-to-bag mapping (each condition becomes a bag)
        # Group matched data by condition
        self.condition_bags = {}
        all_data = self.bag_creator.get_all_matched_data()
        
        for cond in self.sampling_conditions:
            # Filter data for this condition
            cond_mask = np.array([c == cond for c in all_data['conditions']])
            if cond_mask.sum() > 0:
                self.condition_bags[cond] = {
                    'pert_cells': all_data['pert_cells'][cond_mask],
                    'ctrl_cells': all_data['ctrl_cells'][cond_mask],
                    'target_labels': np.array(all_data['target_labels'])[cond_mask]
                }
        
        # Create bag list and index mapping
        self.condition_list = list(self.condition_bags.keys())
        self.total_samples = sum(self.sampler.n_bags for _ in self.condition_list)
        
        # Create mapping from global index to (condition_idx, sample_idx)
        self.index_mapping = []
        for cond_idx, cond in enumerate(self.condition_list):
            for sample_idx in range(self.sampler.n_bags):
                self.index_mapping.append((cond_idx, sample_idx, cond))
        
        # Create RNG for consistent sampling per epoch
        self.rng = np.random.RandomState(seed)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        cond_idx, sample_idx, condition = self.index_mapping[idx]
        bag_data = self.condition_bags[condition]
        
        # Get all perturbation cells for this condition
        pert_cells = bag_data['pert_cells']
        n_total = len(pert_cells)
        pair_indices = np.arange(n_total)
        
        # Create a deterministic RNG state based on condition and sample_idx
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + cond_idx * 1000 + sample_idx)
        
        # Sample pairs for this specific sample
        sampled_pair_idx = self.sampler.sample(pair_indices, rng=sample_rng)[0]  # Get first sample
        
        # Get sampled perturbation cells
        pert_s = pert_cells[sampled_pair_idx]
        target_labels = bag_data['target_labels'][sampled_pair_idx]
        
        # Get control cells: either matched or randomly sampled
        if self.use_matched_control:
            ctrl_s = bag_data['ctrl_cells'][sampled_pair_idx]
        else:
            # Randomly sample control cells
            n_cells = len(pert_s)
            ctrl_s = pd.Index(sample_rng.choice(self.all_control_idx, size=n_cells, replace=True))
        
        # Calculate proportions using consistent class_list
        label_series = pd.Series(target_labels, dtype="category")
        label_series = label_series.cat.set_categories(self.class_list)
        proportions = label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Calculate control proportions
        ctrl_labels = self.adata.obs.loc[ctrl_s][self.bag_creator.cluster_col].values
        ctrl_label_series = pd.Series(ctrl_labels, dtype="category")
        ctrl_label_series = ctrl_label_series.cat.set_categories(self.class_list)
        control_proportions = ctrl_label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Get control cell embeddings
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_s)], dtype=torch.float32)
        
        # Get cluster indices for each cell
        cluster_idx = [self.class_list.index(c) for c in target_labels]
        cluster_idx = torch.tensor(cluster_idx, dtype=torch.long)
        
        # Get perturbation embedding for this condition (single embedding for entire bag)
        # Extract gene name from condition (handle "GENE1+GENE2" format by taking first)
        gene = condition.split("+")[0]
        if gene == "ctrl" or gene == self.bag_creator.control_condition:
            pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
        else:
            pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
        
        # Return single pert_emb (not per-cell) for pure condition
        # Shape: [embedding_dim] instead of [n_cells, embedding_dim]
        return X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb, torch.tensor(control_proportions, dtype=torch.float32), condition


class EvalPerturbationDataset(Dataset):
    """Dataset for evaluation that uses randomly sampled control cells."""
    
    def __init__(self, perturbation_conditions, adata, condition_col, cluster_col, class_list, pert_embeddings, control_condition="ctrl", seed=42, cell_embed_key="X"):
        """
        Dataset for evaluation that uses randomly sampled control cells.
        
        Args:
            perturbation_conditions: List of perturbation condition names
            adata: AnnData object containing cell data
            condition_col: Column name for condition information
            cluster_col: Column name for cluster information
            class_list: List of all possible cluster classes
            pert_embeddings: Dictionary mapping perturbation names to embeddings
            control_condition: Name of control condition (default "ctrl")
            seed: Random seed for reproducible sampling
            cell_embed_key: Key for cell embeddings in adata (default "X")
        """
        self.perturbation_conditions = perturbation_conditions
        self.adata = adata
        self.condition_col = condition_col
        self.cluster_col = cluster_col
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.control_condition = control_condition
        self.cell_embed_key = cell_embed_key
        
        # Set up cell embeddings
        self.X = adata.X if cell_embed_key == "X" else adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        
        # Get all control cell indices
        self.all_control_idx = adata.obs[adata.obs[condition_col] == control_condition].index
        
        # Create RNG for consistent sampling
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.perturbation_conditions)

    def __getitem__(self, idx):
        condition = self.perturbation_conditions[idx]
        
        # Get perturbation cells for this condition (to calculate target proportions)
        pert_cells = self.adata.obs[self.adata.obs[self.condition_col] == condition].index
        n_pert = len(pert_cells)
        
        # Sample control cells randomly from all available control cells
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        sampled_ctrl = sample_rng.choice(self.all_control_idx, size=n_pert, replace=False)
        
        # Calculate target proportions based on actual perturbation cells
        pert_clusters = self.adata.obs.loc[pert_cells][self.cluster_col]
        label_series = pd.Series(pert_clusters.values, dtype="category")
        label_series = label_series.cat.set_categories(self.class_list)
        proportions = label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Calculate control proportions (for residual mode)
        ctrl_clusters = self.adata.obs.loc[sampled_ctrl][self.cluster_col].values
        ctrl_label_series = pd.Series(ctrl_clusters, dtype="category")
        ctrl_label_series = ctrl_label_series.cat.set_categories(self.class_list)
        control_proportions = ctrl_label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values
        
        # Get control cell embeddings (input)
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(sampled_ctrl)], dtype=torch.float32)
        
        # Use control cell cluster labels as dummy labels (will be ignored in loss)
        cluster_idx = [self.class_list.index(c) for c in ctrl_clusters]
        cluster_idx = torch.tensor(cluster_idx, dtype=torch.long)
        
        # Get perturbation embedding for this condition (single embedding for entire bag - pure mode)
        # Extract gene name from condition (handle "GENE1+GENE2" format by taking first)
        gene = condition.split("+")[0]
        if gene == "ctrl" or gene == self.control_condition:
            pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
        else:
            pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
        
        # Return per-cell format for compatibility with existing collate_batch
        # But we can use pure format if needed
        # For now, expand to per-cell format: [embedding_dim] -> [n_cells, embedding_dim]
        pert_emb_tensor = pert_emb.unsqueeze(0).expand(n_pert, -1)  # [n_cells, embedding_dim]
        
        return X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32)