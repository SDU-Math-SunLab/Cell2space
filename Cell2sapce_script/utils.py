import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from anndata import AnnData

def preprocess(sc_adata, st_adata, select_hvg: str = 'intersection', n_features: int = 350):
    """Exactly matches your HVG selection logic."""
    assert sc_adata.shape[1] >= n_features, 'Too few genes in scRNA-seq data!'
    assert st_adata.shape[1] >= n_features, 'Too few genes in ST data!'

    sc.pp.highly_variable_genes(sc_adata, flavor="seurat_v3", n_top_genes=n_features)
    sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_features)

    sc_hvg = sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index
    st_hvg = st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index

    if select_hvg == 'intersection':
        inter_gene = set(sc_hvg).intersection(set(st_hvg))
    elif select_hvg == 'union':
        sc_gene = set(sc_adata.var_names)
        st_gene = set(st_adata.var_names)
        common_gene = set(sc_gene).intersection(set(st_gene))
        inter_gene = set(sc_hvg).union(set(st_hvg))
        inter_gene = inter_gene.intersection(common_gene)

    print(f"Number of common genes selected: {len(inter_gene)}")
    sc_adata = sc_adata[:, list(inter_gene)].copy()
    st_adata = st_adata[:, list(inter_gene)].copy()

    return sc_adata, st_adata

def cal_dist(coord, normalize: bool = True):
    """Calculates Euclidean distances using your original nested loop."""
    dist = []
    num_cells = len(coord)
    for i in tqdm(range(num_cells), desc="Calculating distances"):
        xi, yi = coord[i, :]
        for j in range(num_cells):
            if i >= j:
                continue
            xj, yj = coord[j, :]
            dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist.append(dist_tmp)

    if normalize:
        max_dist = max(dist)
        dist = np.array(dist) / max_dist
    return dist

def create_distance_matrix(coord):
    """Creates an inverse distance matrix (1/dist)."""
    dist = cal_dist(coord)
    num_cells = len(coord)
    distance_matrix = np.zeros((num_cells, num_cells))

    idx = 0
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            inv_dist = 1.0 / (dist[idx] + 1e-10)
            distance_matrix[i, j] = inv_dist
            distance_matrix[j, i] = inv_dist
            idx += 1
    return distance_matrix