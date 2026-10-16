import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist


def _embedding_matrix(adata, embedding_key):
    if embedding_key == "X":
        matrix = adata.X
    else:
        matrix = adata.obsm[embedding_key]
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _nearest_centroid_labels(x, centroids, labels):
    nearest = cdist(x, centroids).argmin(axis=1)
    return np.asarray([labels[idx] for idx in nearest], dtype=object)


def build_control_subclusters(
    adata,
    type_col,
    condition_col,
    control_condition,
    embedding_key,
    subcluster_col="control_subcluster",
    n_neighbors=15,
    resolution=1.0,
    random_state=0,
    output_json=None,
):
    """Define within-type control subclusters and assign all cells to nearest centroids."""
    x_all = _embedding_matrix(adata, embedding_key)
    control_mask = adata.obs[condition_col].values == control_condition
    assignments = np.full(adata.n_obs, "", dtype=object)
    metadata = {
        "subcluster_col": subcluster_col,
        "type_col": type_col,
        "condition_col": condition_col,
        "control_condition": control_condition,
        "embedding_key": embedding_key,
        "subclusters": [],
        "subcluster_to_type": {},
        "subcluster_sizes": [],
    }

    for type_value in sorted(adata.obs[type_col].dropna().unique()):
        type_mask = adata.obs[type_col].values == type_value
        control_type_pos = np.flatnonzero(control_mask & type_mask)
        all_type_pos = np.flatnonzero(type_mask)
        if len(control_type_pos) == 0:
            continue

        control_subset = adata[control_type_pos].copy()
        if embedding_key == "X":
            use_rep = None
        else:
            control_subset.obsm[embedding_key] = x_all[control_type_pos]
            use_rep = embedding_key
        n_neighbors_eff = min(int(n_neighbors), max(len(control_type_pos) - 1, 1))
        if len(control_type_pos) < 3:
            leiden_labels = np.zeros(len(control_type_pos), dtype=object)
        else:
            sc.pp.neighbors(control_subset, n_neighbors=n_neighbors_eff, use_rep=use_rep)
            sc.tl.leiden(control_subset, resolution=resolution, random_state=random_state, key_added="_tmp_subcluster")
            leiden_labels = control_subset.obs["_tmp_subcluster"].astype(str).values

        labels = np.asarray([f"{type_value}__s{label}" for label in leiden_labels], dtype=object)
        unique_labels = sorted(np.unique(labels))
        centroids = []
        for label in unique_labels:
            centroids.append(x_all[control_type_pos][labels == label].mean(axis=0))
            metadata["subclusters"].append(label)
            metadata["subcluster_to_type"][label] = type_value
        metadata["subcluster_sizes"].append(len(unique_labels))

        assigned = _nearest_centroid_labels(x_all[all_type_pos], np.vstack(centroids), unique_labels)
        assignments[all_type_pos] = assigned

    missing = assignments == ""
    if missing.any():
        assignments[missing] = "__unassigned__"
    adata.obs[subcluster_col] = assignments

    if output_json:
        Path(output_json).write_text(json.dumps(metadata, indent=2))
    return metadata


def get_subcluster_metadata(adata, type_col, subcluster_col, class_list):
    if subcluster_col not in adata.obs:
        raise KeyError(f"{subcluster_col!r} is not present in adata.obs")
    rows = adata.obs[[type_col, subcluster_col]].dropna().drop_duplicates()
    rows = rows[rows[subcluster_col] != "__unassigned__"]
    subclusters = []
    subcluster_to_type = {}
    subcluster_sizes = []
    for type_value in class_list:
        labels = sorted(rows.loc[rows[type_col] == type_value, subcluster_col].astype(str).unique())
        subcluster_sizes.append(len(labels))
        for label in labels:
            subclusters.append(label)
            subcluster_to_type[label] = type_value
    return {
        "subclusters": subclusters,
        "subcluster_to_type": subcluster_to_type,
        "subcluster_sizes": subcluster_sizes,
    }
