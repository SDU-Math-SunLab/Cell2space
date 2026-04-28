# Cell2space

**Cell2space: Stepwise multi-scale reconstruction of cell spatial organization from single-cell RNA sequencing data.**

Cell2space is a deep learning framework designed to reconstruct cell spatial organization from scRNA-seq data using spatial transcriptomics (ST) references. Instead of directly predicting absolute coordinates, Cell2space learns a transferable **spatial affinity function** from ST data and applies it to query cells.

## Overview

<div align="center">
  <img src="overview.png" width="800">
</div>

Cell2space follows a stepwise inference strategy:

1. **ST-query affinity inference**: predict affinities between query cells and ST reference spots.
2. **Domain assignment**: assign each query cell to a spatial domain based on ST-domain affinity scores.
3. **Cell-cell spatial affinity inference**: reconstruct query-cell neighborhood relationships using the learned affinity function.

The core workflow performs inference only by default. Benchmark metrics such as domain accuracy and neighborhood cosine similarity can be computed separately when query ground-truth labels and spatial coordinates are available.

## Installation

```bash
git clone https://github.com/SDU-Math-SunLab/Cell2space.git
cd Cell2space
pip install -r requirements.txt
```

Main dependencies include:

```text
torch
scanpy
anndata
numpy
pandas
scikit-learn
tqdm
harmony-pytorch
```

## Input Data

Cell2space takes two `.h5ad` files as input:

| Input | Description |
| --- | --- |
| Query data | scRNA-seq cells or held-out ST spots to be mapped |
| ST reference data | ST spots with spatial domain labels |

Required fields:

| Field | Location | Example |
| --- | --- | --- |
| Gene names | `adata.var_names` | shared genes between query and ST |
| ST domain labels | `adata_st.obs[domain_col]` | `"Region"` |
| Spatial coordinates | `adata.obs` or `adata.obsm["spatial"]` | `("array_row", "array_col")` |

Query domain labels and query spatial coordinates are optional for inference. They are only required for benchmark evaluation or visualization.

## Two Running Modes

Cell2space provides two versions through the same API:

| Mode | Use case | Main SC-SC output |
| --- | --- | --- |
| `mode="full"` | Small or medium datasets; dense matrix inspection | Dense `n_query x n_query` matrix |
| `mode="lite"` | Larger datasets or limited memory | Sparse top-k neighbor indices and scores |

## Quick Start

```python
import sys
from pathlib import Path

CELL2SPACE_DIR = Path("/path/to/Cell2space/Cell2sapce_script")
if str(CELL2SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(CELL2SPACE_DIR))

from cell2space_runner import run_cell2space

result = run_cell2space(
    mode="full",  # use "lite" for the memory-lite version
    sc_path="/path/to/query.h5ad",
    st_path="/path/to/reference_st.h5ad",
    out_dir="/path/to/output",
    domain_col="Region",
    sc_spatial_cols=("array_row", "array_col"),
    st_spatial_cols=("array_row", "array_col"),
    device="cuda",
    seed=42,
    select_hvg="union",
    n_features=1000,
    use_harmony=True,
    k_neighbors=3,
    total_pos=15000,
    total_neg=45000,
    epochs=300,
    batch_size=256,
    filter_mode="hard",
    soft_alpha=0.1,
    layer_relations="auto",
    save_outputs=True,
    save_dense_outputs=True,
)
```

## Memory-Lite Example

```python
result = run_cell2space(
    mode="lite",
    sc_path="/path/to/query.h5ad",
    st_path="/path/to/reference_st.h5ad",
    out_dir="/path/to/output_lite",
    domain_col="Region",
    sc_spatial_cols=("array_row", "array_col"),
    st_spatial_cols=("array_row", "array_col"),
    device="cuda",
    seed=42,
    select_hvg="union",
    n_features=1000,
    use_harmony=True,
    k_neighbors=3,
    total_pos=15000,
    total_neg=45000,
    epochs=300,
    batch_size=256,
    filter_mode="hard",
    soft_alpha=0.1,
    layer_relations="auto",
    pred_batch_size=512,
    rel_batch_size=256,
    candidate_neighbors=256,
    rel_topk=30,
    save_outputs=True,
    save_dense_outputs=False,
)
```

## Main Outputs

The returned `result` dictionary contains:

| Key | Description |
| --- | --- |
| `sc_st_matrix` | ST spot-by-query cell affinity matrix. Optional in lite mode. |
| `sc_domain_scores` | Query cell-by-domain score matrix |
| `sc_domain_predictions` | Predicted spatial domain for each query cell |
| `sc_sc_matrix` | Dense query cell-by-query cell affinity matrix in full mode |
| `sc_sc_filtered_matrix` | Domain-prior filtered dense SC-SC matrix in full mode |
| `sc_sc_topk_indices` | Sparse top-k inferred neighbor indices in lite mode |
| `sc_sc_topk_values` | Sparse top-k inferred neighbor affinity scores in lite mode |

Backward-compatible keys are also kept:

```text
prob_matrix          -> sc_st_matrix
norm_scores          -> sc_domain_scores
predicted_domains    -> sc_domain_predictions
cell_rel_matrix      -> sc_sc_matrix
filtered_rel         -> sc_sc_filtered_matrix
topk_rel_indices     -> sc_sc_topk_indices
topk_rel_values      -> sc_sc_topk_values
```

Print predicted domains:

```python
import numpy as np
import pandas as pd

pred_domains = np.asarray(result["sc_domain_predictions"]).astype(str)
print(pd.Series(pred_domains).value_counts().sort_index())
```

Save per-cell predictions:

```python
pred_df = pd.DataFrame({
    "cell_id": result["adata_sc"].obs_names,
    "predicted_domain": result["sc_domain_predictions"],
})
pred_df.to_csv("/path/to/output/predicted_domains_per_cell.csv", index=False)
```

## Optional Benchmark Evaluation

Evaluation is separated from inference. Use it only when query ground-truth labels and spatial coordinates are available.

```python
from cell2space_runner import evaluate_cell2space_result

metrics = evaluate_cell2space_result(
    result,
    evaluate_k=(1, 3, 5, 10, 15),
)

print("Domain accuracy:", metrics.get("domain_accuracy"))
print("Cosine k=10:", metrics.get("cosine_k10"))
```

## Domain Relation Prior

Cell2space can optionally filter inferred SC-SC affinities using domain adjacency information.

| Option | Meaning |
| --- | --- |
| `layer_relations="auto"` | Use DLPFC layer adjacency if DLPFC labels are detected; otherwise use same-domain relations |
| `layer_relations="dlpfc"` | Always use the DLPFC layer adjacency prior |
| `layer_relations="identity"` | Allow only same-domain relationships |
| `layer_relations="none"` | Disable relation filtering |
| `layer_relations={...}` | Use a custom relation dictionary |

DLPFC preset:

```python
{
    "Layer1": ["Layer1", "Layer2"],
    "Layer2": ["Layer1", "Layer2", "Layer3"],
    "Layer3": ["Layer2", "Layer3", "Layer4"],
    "Layer4": ["Layer3", "Layer4", "Layer5"],
    "Layer5": ["Layer4", "Layer5", "Layer6"],
    "Layer6": ["Layer5", "Layer6", "WM"],
    "WM": ["Layer6", "WM"],
}
```

For non-DLPFC datasets, a conservative choice is:

```python
layer_relations="identity"
```

To disable filtering:

```python
layer_relations="none"
filter_mode="none"
```

## Important Parameters

| Parameter | Description | Default/example |
| --- | --- | --- |
| `mode` | Running version | `"full"` or `"lite"` |
| `domain_col` | ST domain label column | `"Region"` |
| `sc_spatial_cols` | Query spatial coordinate columns | `("array_row", "array_col")` |
| `st_spatial_cols` | ST spatial coordinate columns | `("array_row", "array_col")` |
| `select_hvg` | HVG selection strategy | `"union"` |
| `n_features` | Number of HVGs selected per dataset | `350` or `1000` |
| `use_harmony` | Whether to integrate query and ST data with Harmony | `True` |
| `k_neighbors` | ST nearest-neighbor count for positive pair sampling | `3` |
| `total_pos` | Number of positive training pairs | `15000` |
| `total_neg` | Number of negative training pairs | `45000` |
| `epochs` | MLP training epochs | `300` |
| `batch_size` | MLP training batch size | `256` |
| `filter_mode` | Domain-prior filtering strategy | `"hard"`, `"soft"`, or `"none"` |
| `soft_alpha` | Weight for disallowed pairs in soft filtering | `0.1` |
| `candidate_neighbors` | Lite mode candidate neighbor pool size | `256` |
| `rel_topk` | Lite mode retained top-k SC-SC neighbors | `30` |
| `evaluate_k` | Neighbor sizes used for optional evaluation | `(1, 3, 5, 10, 15)` |

## Saved Files

When `save_outputs=True`, Cell2space writes outputs to `out_dir`.

Common files:

| File | Description |
| --- | --- |
| `query_with_predictions.h5ad` | Query AnnData with predicted domain labels |
| `reference_processed.h5ad` | Processed ST reference AnnData |
| `sc_domain_predictions.npy` | Predicted query domain labels |
| `sc_domain_scores.npy` | Query-by-domain score matrix |
| `summary.json` | Configuration and output summary |

Full mode additionally saves:

```text
sc_st_matrix.npy
sc_sc_matrix.npy
sc_sc_filtered_matrix.npy
```

Lite mode additionally saves:

```text
sc_sc_topk_indices.npy
sc_sc_topk_values.npy
domain_order.npy
```

## Tutorials

Example scripts are provided in:

```text
TUTORIAL.ipynb

```

## Citation

If you use Cell2space in your research, please cite:

```text
Cell2space: Stepwise multi-scale reconstruction of cell spatial organization
from single-cell RNA sequencing data.
Jieyi Pan, Qiyuan Guan, and Duanchen Sun.
```
