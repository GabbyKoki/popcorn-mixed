import os
import pickle
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

try:
    import cinemaot as co
except ImportError:
    print("Warning: cinemaot not available.")
    co = None


class PertBag:
    def __init__(self, perturbation_cells, control_cells, perturbation_name, adata, cluster_col="cluster_name"):
        self.perturbation_cells = perturbation_cells
        self.control_cells = control_cells
        self.perturbation_name = perturbation_name
        self.adata = adata
        self.cluster_col = cluster_col
        self._calculate_proportions()

    def _calculate_proportions(self):
        clusters = self.adata.obs.loc[self.perturbation_cells][self.cluster_col]
        self.proportions = clusters.value_counts(normalize=True).to_dict()

    def get_perturbation_proportions(self):
        return np.array(list(self.proportions.values()))

    def get_perturbation_cluster_labels(self):
        return self.adata.obs.loc[self.perturbation_cells][self.cluster_col].values

    def get_matched_indices(self):
        return self.perturbation_cells, self.control_cells


class BagCreator:
    """Simplified bag creator for creating basic perturbation bags."""
    def __init__(self, adata, condition_col="condition", cluster_col="cluster_name", control_condition="ctrl", matching_index_path=None):
        self.adata = adata
        self.condition_col = condition_col
        self.cluster_col = cluster_col
        self.control_condition = control_condition
        self.matching_index_path = matching_index_path
        self.control_indices = adata.obs[adata.obs[condition_col] == control_condition].index
        if issparse(adata.X):
            self.adata.X = adata.X.toarray()
        self.matching_index = {}
        self.all_pert_to_ctrl_mapping = {}

    def compute_matching_index(self, perturbation_conditions: List[str]) -> Dict[str, Dict[str, pd.Index]]:
        """Compute matching between perturbed and control cells."""
        if co is None:
            raise ImportError("cinemaot required")
        adata_subset = self.adata[self.adata.obs[self.condition_col].isin(perturbation_conditions + [self.control_condition])].copy()
        sc.pp.pca(adata_subset)
        
        for cond in perturbation_conditions:
            cf, ot, _ = co.cinemaot.cinemaot_unweighted(
                adata_subset, obs_label=self.condition_col,
                ref_label=cond, expr_label=self.control_condition
            )
            pert_mask = adata_subset.obs[self.condition_col] == cond
            ctrl_mask = adata_subset.obs[self.condition_col] == self.control_condition
            pert_indices = adata_subset.obs[pert_mask].index
            ctrl_indices = adata_subset.obs[ctrl_mask].index
            matched_ctrl = ctrl_indices[ot.argmax(axis=1)]
            self.matching_index[cond] = {"perturbation": pert_indices, "control": matched_ctrl}
            
            # Create individual cell mapping
            for pert_idx, ctrl_idx in zip(pert_indices, matched_ctrl):
                self.all_pert_to_ctrl_mapping[pert_idx] = ctrl_idx
                
        if self.matching_index_path:
            with open(self.matching_index_path, "wb") as f:
                pickle.dump(self.matching_index, f)
        return self.matching_index

    def load_matching_index(self) -> Dict[str, Dict[str, pd.Index]]:
        """Load matching index from file."""
        if not self.matching_index_path:
            raise ValueError("matching_index_path not specified")
        
        if not os.path.exists(self.matching_index_path):
            raise FileNotFoundError(f"Matching index file not found: {self.matching_index_path}")
        
        with open(self.matching_index_path, "rb") as f:
            self.matching_index = pickle.load(f)
            
        # Rebuild individual cell mapping
        self.all_pert_to_ctrl_mapping = {}
        for cond in self.matching_index:
            pert_indices = self.matching_index[cond]["perturbation"]
            ctrl_indices = self.matching_index[cond]["control"]
            for pert_idx, ctrl_idx in zip(pert_indices, ctrl_indices):
                self.all_pert_to_ctrl_mapping[pert_idx] = ctrl_idx
                
        return self.matching_index

    def create_bags(self, perturbation_conditions: List[str]) -> List[PertBag]:
        """Create PertBag objects for given perturbation conditions."""
        return [
            PertBag(self.matching_index[cond]["perturbation"], self.matching_index[cond]["control"], cond, self.adata, cluster_col=self.cluster_col)
            for cond in perturbation_conditions
        ]
    
    def get_all_matched_data(self):
        """Get all matched perturbation-control pairs across all conditions."""
        all_pert_cells = []
        all_ctrl_cells = []
        all_conditions = []
        all_target_labels = []
        
        for cond, data in self.matching_index.items():
            pert_cells = data["perturbation"]
            ctrl_cells = data["control"]
            
            # Get cluster labels for perturbation cells
            cluster_labels = self.adata.obs.loc[pert_cells][self.cluster_col].values
            
            all_pert_cells.extend(pert_cells)
            all_ctrl_cells.extend(ctrl_cells)
            all_conditions.extend([cond] * len(pert_cells))
            all_target_labels.extend(cluster_labels)
        
        return {
            'pert_cells': pd.Index(all_pert_cells),
            'ctrl_cells': pd.Index(all_ctrl_cells),
            'conditions': all_conditions,
            'target_labels': all_target_labels
        }