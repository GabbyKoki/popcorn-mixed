import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from pert_bags import PertBag
from bag_sampler import BootstrapBagSampler


def collate_batch(batch):
    has_subclusters = len(batch[0]) == 8
    if has_subclusters:
        bags, labels, cell_labels, pert_emd, control_props, subcluster_props, cell_sublabels, control_subcluster_props = zip(*batch)
    else:
        bags, labels, cell_labels, pert_emd, control_props = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)
    padded_cell_labels = pad_sequence(cell_labels, batch_first=True, padding_value=-1)
    padded_pert_emd = pad_sequence(pert_emd, batch_first=True)
    attention_mask = torch.tensor([[1] * len(b) + [0] * (padded_bags.size(1) - len(b)) for b in bags])
    labels = torch.stack(labels)
    control_props = torch.stack(control_props)
    if has_subclusters:
        padded_cell_sublabels = pad_sequence(cell_sublabels, batch_first=True, padding_value=-1)
        return (
            padded_bags, attention_mask.bool(), labels, padded_cell_labels,
            padded_pert_emd, control_props,
            torch.stack(subcluster_props), padded_cell_sublabels, torch.stack(control_subcluster_props),
        )
    return padded_bags, attention_mask.bool(), labels, padded_cell_labels, padded_pert_emd, control_props


def collate_pure_batch(batch):
    has_subclusters = len(batch[0]) == 9
    if has_subclusters:
        bags, labels, cell_labels, pert_emd, control_props, subcluster_props, cell_sublabels, control_subcluster_props, conditions = zip(*batch)
    else:
        bags, labels, cell_labels, pert_emd, control_props, conditions = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True)
    padded_cell_labels = pad_sequence(cell_labels, batch_first=True, padding_value=-1)
    pert_emd = torch.stack(pert_emd)
    attention_mask = torch.tensor([[1] * len(b) + [0] * (padded_bags.size(1) - len(b)) for b in bags])
    labels = torch.stack(labels)
    control_props = torch.stack(control_props)
    if has_subclusters:
        padded_cell_sublabels = pad_sequence(cell_sublabels, batch_first=True, padding_value=-1)
        return (
            padded_bags, attention_mask.bool(), labels, padded_cell_labels,
            pert_emd, control_props,
            torch.stack(subcluster_props), padded_cell_sublabels, torch.stack(control_subcluster_props),
            conditions,
        )
    return padded_bags, attention_mask.bool(), labels, padded_cell_labels, pert_emd, control_props, conditions


class SubclusterMixin:
    def _init_subclusters(self, subcluster_col=None, subcluster_list=None, subcluster_to_type=None):
        self.subcluster_col = subcluster_col
        self.subcluster_list = list(subcluster_list or [])
        self.subcluster_to_type = dict(subcluster_to_type or {})
        self.use_subclusters = bool(self.subcluster_col and self.subcluster_list)
        self.subcluster_index = {label: idx for idx, label in enumerate(self.subcluster_list)}
        self.subcluster_type_idx = np.array(
            [self.class_list.index(self.subcluster_to_type[label]) for label in self.subcluster_list],
            dtype=np.int64,
        ) if self.use_subclusters else np.array([], dtype=np.int64)

    def _type_proportions(self, labels):
        label_series = pd.Series(labels, dtype="category").cat.set_categories(self.class_list)
        return label_series.value_counts(normalize=True).reindex(self.class_list, fill_value=0.0).values

    def _subcluster_targets(self, cell_ids, type_labels):
        if not self.use_subclusters:
            return None, None
        sub_labels = self.adata.obs.loc[cell_ids][self.subcluster_col].values
        cell_sub_idx = torch.tensor([self.subcluster_index.get(label, -1) for label in sub_labels], dtype=torch.long)
        props = np.zeros(len(self.subcluster_list), dtype=np.float32)
        for type_idx, type_label in enumerate(self.class_list):
            type_mask = np.asarray(type_labels) == type_label
            denom = int(type_mask.sum())
            if denom == 0:
                continue
            valid_subs = [self.subcluster_index.get(label, -1) for label in np.asarray(sub_labels)[type_mask]]
            valid_subs = [idx for idx in valid_subs if idx >= 0 and self.subcluster_type_idx[idx] == type_idx]
            if valid_subs:
                counts = np.bincount(valid_subs, minlength=len(self.subcluster_list)).astype(np.float32)
                props += counts / max(float(len(valid_subs)), 1.0)
        return cell_sub_idx, torch.tensor(props, dtype=torch.float32)


class PerturbationDataset(SubclusterMixin, Dataset):
    def __init__(self, pert_bags, adata, cluster_col, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X", subcluster_col=None, subcluster_list=None, subcluster_to_type=None):
        self.pert_bags = pert_bags
        self.adata = adata
        self.cluster_col = cluster_col
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        self.X = adata.X if cell_embed_key == "X" else adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        self._init_subclusters(subcluster_col, subcluster_list, subcluster_to_type)
        self.rng = np.random.RandomState(seed)
        self.total_samples = sum(self.sampler.n_bags for _ in self.pert_bags)
        self.index_mapping = []
        for bag_idx, _ in enumerate(self.pert_bags):
            for sample_idx in range(self.sampler.n_bags):
                self.index_mapping.append((bag_idx, sample_idx))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        bag_idx, sample_idx = self.index_mapping[idx]
        pert_bag = self.pert_bags[bag_idx]
        pert_idx, ctrl_idx = pert_bag.get_matched_indices()
        pair_idx = np.arange(len(pert_idx))
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + bag_idx * 1000 + sample_idx)
        sampled_pair_idx = self.sampler.sample(pair_idx, rng=sample_rng)[0]
        pert_s = pert_idx[sampled_pair_idx]
        ctrl_s = ctrl_idx[sampled_pair_idx]
        all_labels = pert_bag.get_perturbation_cluster_labels()
        target_labels = all_labels[sampled_pair_idx]
        proportions = self._type_proportions(target_labels)
        ctrl_labels = self.adata.obs.loc[ctrl_s][self.cluster_col].values
        control_proportions = self._type_proportions(ctrl_labels)
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_s)], dtype=torch.float32)
        cluster_idx = torch.tensor([self.class_list.index(c) for c in target_labels], dtype=torch.long)
        pert_emb_list = []
        for _cell_idx in pert_s:
            gene = pert_bag.perturbation_name.split("+")[0]
            if gene == "ctrl":
                pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
            else:
                pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
            pert_emb_list.append(pert_emb)
        pert_emb_tensor = torch.stack(pert_emb_list)
        item = (X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32))
        if self.use_subclusters:
            cell_sublabels, subcluster_props = self._subcluster_targets(pert_s, target_labels)
            _, control_subcluster_props = self._subcluster_targets(ctrl_s, ctrl_labels)
            item = item + (subcluster_props, cell_sublabels, control_subcluster_props)
        return item


class MixedConditionDataset(SubclusterMixin, Dataset):
    def __init__(self, bag_creator, sampling_conditions, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X", use_matched_control=True, subcluster_col=None, subcluster_list=None, subcluster_to_type=None):
        self.bag_creator = bag_creator
        self.adata = bag_creator.adata
        self.sampling_conditions = sampling_conditions
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        self.use_matched_control = use_matched_control
        self.X = self.adata.X if cell_embed_key == "X" else self.adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        self._init_subclusters(subcluster_col, subcluster_list, subcluster_to_type)
        self.matched_data = self._get_filtered_matched_data()
        self.all_control_idx = self.adata.obs[self.adata.obs[self.bag_creator.condition_col] == self.bag_creator.control_condition].index
        self.rng = np.random.RandomState(seed)
        self.total_samples = self.sampler.n_bags

    def _get_filtered_matched_data(self):
        all_data = self.bag_creator.get_all_matched_data()
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
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        n_total = len(self.matched_data['pert_cells'])
        sampled_pair_idx = self.sampler.sample(np.arange(n_total), rng=sample_rng)[0]
        pert_cells = self.matched_data['pert_cells'][sampled_pair_idx]
        conditions = np.array(self.matched_data['conditions'])[sampled_pair_idx]
        target_labels = self.matched_data['target_labels'][sampled_pair_idx]
        if self.use_matched_control:
            ctrl_cells = self.matched_data['ctrl_cells'][sampled_pair_idx]
        else:
            ctrl_cells = pd.Index(sample_rng.choice(self.all_control_idx, size=len(pert_cells), replace=True))
        proportions = self._type_proportions(target_labels)
        ctrl_labels = self.adata.obs.loc[ctrl_cells][self.bag_creator.cluster_col].values
        control_proportions = self._type_proportions(ctrl_labels)
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_cells)], dtype=torch.float32)
        cluster_idx = torch.tensor([self.class_list.index(c) for c in target_labels], dtype=torch.long)
        pert_emb_list = []
        for condition in conditions:
            gene = condition.split("+")[0]
            if gene == "ctrl":
                pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
            else:
                pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
            pert_emb_list.append(pert_emb)
        pert_emb_tensor = torch.stack(pert_emb_list)
        item = (X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32))
        if self.use_subclusters:
            cell_sublabels, subcluster_props = self._subcluster_targets(pert_cells, target_labels)
            _, control_subcluster_props = self._subcluster_targets(ctrl_cells, ctrl_labels)
            item = item + (subcluster_props, cell_sublabels, control_subcluster_props)
        return item


class PureConditionDataset(SubclusterMixin, Dataset):
    def __init__(self, bag_creator, sampling_conditions, class_list, pert_embeddings, sampler, seed=42, cell_embed_key="X", use_matched_control=True, subcluster_col=None, subcluster_list=None, subcluster_to_type=None):
        self.bag_creator = bag_creator
        self.adata = bag_creator.adata
        self.sampling_conditions = sampling_conditions
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.sampler = sampler
        self.cell_embed_key = cell_embed_key
        self.use_matched_control = use_matched_control
        self.X = self.adata.X if cell_embed_key == "X" else self.adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        self._init_subclusters(subcluster_col, subcluster_list, subcluster_to_type)
        self.all_control_idx = self.adata.obs[self.adata.obs[self.bag_creator.condition_col] == self.bag_creator.control_condition].index
        self.condition_bags = {}
        all_data = self.bag_creator.get_all_matched_data()
        for cond in self.sampling_conditions:
            cond_mask = np.array([c == cond for c in all_data['conditions']])
            if cond_mask.sum() > 0:
                self.condition_bags[cond] = {
                    'pert_cells': all_data['pert_cells'][cond_mask],
                    'ctrl_cells': all_data['ctrl_cells'][cond_mask],
                    'target_labels': np.array(all_data['target_labels'])[cond_mask]
                }
        self.condition_list = list(self.condition_bags.keys())
        self.total_samples = sum(self.sampler.n_bags for _ in self.condition_list)
        self.index_mapping = []
        for cond_idx, cond in enumerate(self.condition_list):
            for sample_idx in range(self.sampler.n_bags):
                self.index_mapping.append((cond_idx, sample_idx, cond))
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cond_idx, sample_idx, condition = self.index_mapping[idx]
        bag_data = self.condition_bags[condition]
        pair_indices = np.arange(len(bag_data['pert_cells']))
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + cond_idx * 1000 + sample_idx)
        sampled_pair_idx = self.sampler.sample(pair_indices, rng=sample_rng)[0]
        pert_s = bag_data['pert_cells'][sampled_pair_idx]
        target_labels = bag_data['target_labels'][sampled_pair_idx]
        if self.use_matched_control:
            ctrl_s = bag_data['ctrl_cells'][sampled_pair_idx]
        else:
            ctrl_s = pd.Index(sample_rng.choice(self.all_control_idx, size=len(pert_s), replace=True))
        proportions = self._type_proportions(target_labels)
        ctrl_labels = self.adata.obs.loc[ctrl_s][self.bag_creator.cluster_col].values
        control_proportions = self._type_proportions(ctrl_labels)
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(ctrl_s)], dtype=torch.float32)
        cluster_idx = torch.tensor([self.class_list.index(c) for c in target_labels], dtype=torch.long)
        gene = condition.split("+")[0]
        if gene == "ctrl" or gene == self.bag_creator.control_condition:
            pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
        else:
            pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
        item = (X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb, torch.tensor(control_proportions, dtype=torch.float32))
        if self.use_subclusters:
            cell_sublabels, subcluster_props = self._subcluster_targets(pert_s, target_labels)
            _, control_subcluster_props = self._subcluster_targets(ctrl_s, ctrl_labels)
            item = item + (subcluster_props, cell_sublabels, control_subcluster_props)
        return item + (condition,)


class EvalPerturbationDataset(SubclusterMixin, Dataset):
    """Legacy single-shot eval dataset. Prefer bootstrap eval via PureConditionDataset."""
    def __init__(self, perturbation_conditions, adata, condition_col, cluster_col, class_list, pert_embeddings, control_condition="ctrl", seed=42, cell_embed_key="X", subcluster_col=None, subcluster_list=None, subcluster_to_type=None):
        self.perturbation_conditions = perturbation_conditions
        self.adata = adata
        self.condition_col = condition_col
        self.cluster_col = cluster_col
        self.class_list = class_list
        self.pert_embeddings = pert_embeddings
        self.control_condition = control_condition
        self.cell_embed_key = cell_embed_key
        self.X = adata.X if cell_embed_key == "X" else adata.obsm[cell_embed_key]
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
        self.input_dim = self.X.shape[1]
        self.gene_input_dim = len(next(iter(pert_embeddings.values())))
        self._init_subclusters(subcluster_col, subcluster_list, subcluster_to_type)
        self.all_control_idx = adata.obs[adata.obs[condition_col] == control_condition].index
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.perturbation_conditions)

    def __getitem__(self, idx):
        condition = self.perturbation_conditions[idx]
        pert_cells = self.adata.obs[self.adata.obs[self.condition_col] == condition].index
        n_pert = len(pert_cells)
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        sampled_ctrl = sample_rng.choice(self.all_control_idx, size=n_pert, replace=False)
        pert_clusters = self.adata.obs.loc[pert_cells][self.cluster_col]
        proportions = self._type_proportions(pert_clusters.values)
        ctrl_clusters = self.adata.obs.loc[sampled_ctrl][self.cluster_col].values
        control_proportions = self._type_proportions(ctrl_clusters)
        X_ctrl = torch.tensor(self.X[self.adata.obs.index.get_indexer(sampled_ctrl)], dtype=torch.float32)
        cluster_idx = torch.tensor([self.class_list.index(c) for c in ctrl_clusters], dtype=torch.long)
        gene = condition.split("+")[0]
        if gene == "ctrl" or gene == self.control_condition:
            pert_emb = torch.zeros(len(next(iter(self.pert_embeddings.values()))), dtype=torch.float32)
        else:
            pert_emb = torch.tensor(self.pert_embeddings[gene], dtype=torch.float32)
        pert_emb_tensor = pert_emb.unsqueeze(0).expand(n_pert, -1)
        item = (X_ctrl, torch.tensor(proportions, dtype=torch.float32), cluster_idx, pert_emb_tensor, torch.tensor(control_proportions, dtype=torch.float32))
        if self.use_subclusters:
            cell_sublabels, subcluster_props = self._subcluster_targets(pert_cells, pert_clusters.values)
            _, control_subcluster_props = self._subcluster_targets(sampled_ctrl, ctrl_clusters)
            item = item + (subcluster_props, cell_sublabels, control_subcluster_props)
        return item
