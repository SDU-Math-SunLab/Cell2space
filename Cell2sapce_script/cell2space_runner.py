import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from main import evaluate_full_result, run_pipeline
from pipeline import SpatialAffinityFunction
from utils import preprocess


DLPFC_LAYER_RELATIONS = {
    "Layer1": ["Layer1", "Layer2"],
    "Layer2": ["Layer1", "Layer2", "Layer3"],
    "Layer3": ["Layer2", "Layer3", "Layer4"],
    "Layer4": ["Layer3", "Layer4", "Layer5"],
    "Layer5": ["Layer4", "Layer5", "Layer6"],
    "Layer6": ["Layer5", "Layer6", "WM"],
    "WM": ["Layer6", "WM"],
}


@dataclass
class Cell2spaceConfig:
    sc_path: str
    st_path: str
    out_dir: str
    mode: str = "full"
    domain_col: str = "Region"
    sc_spatial_cols: tuple = ("array_row", "array_col")
    st_spatial_cols: tuple = ("array_row", "array_col")
    device: str = "cuda"
    seed: int = 42
    select_hvg: str = "union"
    n_features: int = 350
    use_harmony: bool = True
    k_neighbors: int = 3
    total_pos: int = 15000
    total_neg: int = 45000
    epochs: int = 300
    batch_size: int = 256
    filter_mode: str = "hard"
    soft_alpha: float = 0.1
    layer_relations: object = "auto"
    pred_batch_size: int = 512
    rel_batch_size: int = 256
    candidate_neighbors: int = 256
    rel_topk: int = 30
    evaluate_k: tuple = (1, 3, 5, 10, 15)
    evaluate: bool = False
    save_outputs: bool = True
    save_dense_outputs: bool = True


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _normalize_mode(mode):
    mode = str(mode).lower().replace("-", "_")
    if mode in {"full", "normal", "standard"}:
        return "full"
    if mode in {"lite", "memlite", "memory_lite", "memorylite"}:
        return "lite"
    raise ValueError("mode must be one of: 'full', 'normal', 'lite', or 'memory_lite'")


def _reference_domain_labels(st_path, domain_col):
    adata = None
    try:
        adata = ad.read_h5ad(st_path, backed="r")
        if domain_col not in adata.obs.columns:
            return []
        return adata.obs[domain_col].dropna().astype(str).unique().tolist()
    finally:
        if adata is not None and getattr(adata, "file", None) is not None:
            adata.file.close()


def resolve_layer_relations(layer_relations="auto", domain_labels=None):
    """
    Resolve domain adjacency priors.

    Accepted values:
    - "auto": use DLPFC layer adjacency if DLPFC labels are detected; otherwise use same-domain only.
    - "dlpfc": always use DLPFC layer adjacency.
    - "identity": only allow same-domain neighbors.
    - "none": disable relation filtering by returning an empty dictionary.
    - dict: use the provided custom relations.
    """
    if isinstance(layer_relations, dict):
        return layer_relations

    labels = sorted({str(x) for x in (domain_labels or [])})
    preset = "auto" if layer_relations is None else str(layer_relations).lower()

    if preset == "none":
        return {}
    if preset == "dlpfc":
        return DLPFC_LAYER_RELATIONS.copy()
    if preset == "identity":
        return {label: [label] for label in labels}
    if preset != "auto":
        raise ValueError("layer_relations must be 'auto', 'dlpfc', 'identity', 'none', or a dictionary")

    dlpfc_labels = set(DLPFC_LAYER_RELATIONS)
    if labels and set(labels).issubset(dlpfc_labels):
        return DLPFC_LAYER_RELATIONS.copy()

    return {label: [label] for label in labels}


def _prepare_adata(adata_obj, domain_col, spatial_cols, name, require_domain=True, require_spatial=True):
    if domain_col not in adata_obj.obs.columns:
        if not require_domain:
            adata_obj = adata_obj.copy()
            adata_obj.obs_names_make_unique()
            adata_obj.var_names_make_unique()
            if spatial_cols is not None and all(col in adata_obj.obs.columns for col in spatial_cols):
                adata_obj.obsm["spatial"] = np.asarray(adata_obj.obs[list(spatial_cols)], dtype=float)
            return adata_obj
        raise KeyError(
            f"{name}: domain column '{domain_col}' not found in adata.obs. "
            f"Available columns include: {list(adata_obj.obs.columns[:20])}"
        )

    if require_domain:
        adata_obj = adata_obj[~adata_obj.obs[domain_col].isna()].copy()
        adata_obj.obs["domain"] = adata_obj.obs[domain_col].astype(str)
    else:
        adata_obj = adata_obj.copy()
        if not adata_obj.obs[domain_col].isna().any():
            adata_obj.obs["domain"] = adata_obj.obs[domain_col].astype(str)

    if spatial_cols is not None:
        missing_cols = [col for col in spatial_cols if col not in adata_obj.obs.columns]
        if missing_cols:
            if not require_spatial:
                adata_obj.obs_names_make_unique()
                adata_obj.var_names_make_unique()
                return adata_obj
            raise KeyError(
                f"{name}: spatial columns {missing_cols} not found in adata.obs. "
                f"Requested spatial columns: {spatial_cols}"
            )
        adata_obj.obsm["spatial"] = np.asarray(adata_obj.obs[list(spatial_cols)], dtype=float)
    elif require_spatial and "spatial" not in adata_obj.obsm:
        raise KeyError(
            f"{name}: no spatial information found. Provide spatial_cols or set adata.obsm['spatial'] first."
        )

    adata_obj.obs_names_make_unique()
    adata_obj.var_names_make_unique()
    return adata_obj


def _json_ready_config(config):
    out = asdict(config)
    out["sc_spatial_cols"] = None if config.sc_spatial_cols is None else list(config.sc_spatial_cols)
    out["st_spatial_cols"] = None if config.st_spatial_cols is None else list(config.st_spatial_cols)
    out["evaluate_k"] = list(config.evaluate_k)
    if isinstance(config.layer_relations, dict):
        out["layer_relations"] = config.layer_relations
    return out


def _optional_metric_summary(result):
    summary = {}
    for key in ("domain_accuracy", "strict_accuracy", "cosine_similarity"):
        if key in result and result[key] is not None:
            summary[key] = float(result[key])
    for key, value in result.items():
        if key.startswith("cosine_k") and value is not None:
            summary[key] = float(value)
    return summary


def _evaluate_sparse_neighbor_cosine(adata_sc, topk_indices, evaluate_k=(1, 3, 5, 10, 15), default_k=3):
    if "spatial" not in adata_sc.obsm:
        return {}

    evaluate_k = tuple(sorted(set(evaluate_k)))
    max_k = max(evaluate_k)
    if topk_indices.shape[1] < max_k:
        raise ValueError(
            f"Sparse SC-SC output only contains {topk_indices.shape[1]} neighbors, "
            f"but evaluation requested k={max_k}. Increase rel_topk or reduce evaluate_k."
        )

    sc_expr = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else np.asarray(adata_sc.X)
    spatial = np.asarray(adata_sc.obsm["spatial"], dtype=float)
    n_neighbors = min(max_k + 1, len(spatial))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(spatial)
    _, true_idx = nbrs.kneighbors(spatial)
    true_idx = true_idx[:, 1:]

    metrics = {}
    for k in evaluate_k:
        sims = []
        for i in range(len(sc_expr)):
            pred_idx = topk_indices[i, :k]
            true_neighbors = true_idx[i, :k]
            pred_mean = np.mean(sc_expr[pred_idx], axis=0).reshape(1, -1)
            true_mean = np.mean(sc_expr[true_neighbors], axis=0).reshape(1, -1)
            sims.append(cosine_similarity(pred_mean, true_mean)[0, 0])
        metrics[f"cosine_k{k}"] = float(np.mean(sims))

    default_key = f"cosine_k{default_k}"
    metrics["cosine_similarity"] = metrics.get(default_key, metrics[f"cosine_k{max_k}"])
    return metrics


def evaluate_cell2space_result(result, evaluate_k=(1, 3, 5, 10, 15), default_k=None):
    """
    Evaluate a Cell2space inference result when ground-truth query labels/spatial
    coordinates are available.

    This function is intentionally separate from inference, so non-benchmark
    datasets can be processed without requiring query labels or coordinates.
    """
    config = result.get("config", {})
    if default_k is None:
        default_k = config.get("k_neighbors", 3) if isinstance(config, dict) else 3

    mode = str(result.get("mode", "full")).lower()
    if mode == "full" and "filtered_rel" in result:
        metrics = evaluate_full_result(result, evaluate_k=evaluate_k, default_k=default_k)
    else:
        adata_sc = result["adata_sc"]
        predicted_domains = np.asarray(result.get("predicted_domains", result.get("sc_domain_predictions")))
        metrics = {}
        if "domain" in adata_sc.obs.columns and predicted_domains.size:
            true_domains = adata_sc.obs["domain"].astype(str).values
            metrics["domain_accuracy"] = float(np.mean(predicted_domains.astype(str) == true_domains))
            metrics["strict_accuracy"] = metrics["domain_accuracy"]
        if "topk_rel_indices" in result:
            metrics.update(
                _evaluate_sparse_neighbor_cosine(
                    adata_sc=adata_sc,
                    topk_indices=result["topk_rel_indices"],
                    evaluate_k=evaluate_k,
                    default_k=default_k,
                )
            )

    result.update(metrics)
    return metrics


def _save_full_outputs(result, out_dir, config, layer_relations):
    out_dir.mkdir(exist_ok=True, parents=True)
    adata_sc = result["adata_sc"].copy()
    adata_sc.obs["predicted_domain"] = np.asarray(result["predicted_domains"]).astype(str)

    adata_sc.write_h5ad(out_dir / "query_with_predictions.h5ad")
    result["adata_st"].write_h5ad(out_dir / "reference_processed.h5ad")

    np.save(out_dir / "sc_domain_predictions.npy", result["sc_domain_predictions"])
    np.save(out_dir / "sc_domain_scores.npy", result["sc_domain_scores"])
    np.save(out_dir / "sc_sc_filtered_matrix.npy", result["sc_sc_filtered_matrix"])

    # Backward-compatible filenames used by earlier notebooks.
    np.save(out_dir / "predicted_domains.npy", result["predicted_domains"])
    np.save(out_dir / "norm_scores.npy", result["norm_scores"])
    np.save(out_dir / "filtered_rel.npy", result["filtered_rel"])

    if config.save_dense_outputs:
        np.save(out_dir / "sc_st_matrix.npy", result["sc_st_matrix"])
        np.save(out_dir / "sc_sc_matrix.npy", result["sc_sc_matrix"])
        np.save(out_dir / "prob_matrix.npy", result["prob_matrix"])
        np.save(out_dir / "cell_rel_matrix.npy", result["cell_rel_matrix"])

    summary = {
        "mode": "full",
        "outputs": {
            "sc_st_matrix": "Dense ST spot-by-query SC affinity matrix. Saved when save_dense_outputs=True.",
            "sc_domain_predictions": "Predicted query domain labels.",
            "sc_domain_scores": "Query-by-domain score matrix.",
            "sc_sc_matrix": "Dense query-by-query affinity matrix. Saved when save_dense_outputs=True.",
            "sc_sc_filtered_matrix": "Domain-prior filtered query-by-query affinity matrix.",
        },
        "layer_relations": layer_relations,
        "config": _json_ready_config(config),
    }
    summary.update(_optional_metric_summary(result))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _run_full(config, layer_relations):
    evaluate_k = tuple(sorted(set(tuple(config.evaluate_k) + (config.k_neighbors,))))
    result = run_pipeline(
        sc_path=config.sc_path,
        st_path=config.st_path,
        layer_relations=layer_relations,
        domain_col=config.domain_col,
        sc_spatial_cols=config.sc_spatial_cols,
        st_spatial_cols=config.st_spatial_cols,
        k_neighbors=config.k_neighbors,
        total_pos=config.total_pos,
        total_neg=config.total_neg,
        epochs=config.epochs,
        batch_size=config.batch_size,
        device=config.device,
        select_hvg=config.select_hvg,
        n_features=config.n_features,
        use_harmony=config.use_harmony,
        filter_mode=config.filter_mode,
        soft_alpha=config.soft_alpha,
        evaluate_k=evaluate_k,
        evaluate=config.evaluate,
        seed=config.seed,
    )
    result["mode"] = "full"
    result["layer_relations"] = layer_relations
    return result


def _run_lite(config, layer_relations):
    evaluate_k = tuple(sorted(set(tuple(config.evaluate_k) + (config.k_neighbors,))))
    device = config.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    adata_sc = _prepare_adata(
        ad.read_h5ad(config.sc_path),
        config.domain_col,
        config.sc_spatial_cols,
        "Query",
        require_domain=config.evaluate,
        require_spatial=config.evaluate,
    )
    adata_st = _prepare_adata(
        ad.read_h5ad(config.st_path),
        config.domain_col,
        config.st_spatial_cols,
        "ST reference",
        require_domain=True,
        require_spatial=False,
    )

    adata_sc, adata_st = preprocess(
        adata_sc,
        adata_st,
        select_hvg=config.select_hvg,
        n_features=config.n_features,
    )

    saf = SpatialAffinityFunction(device=device, lr=1e-4)
    if config.use_harmony:
        adata_sc, adata_st = saf.run_harmony(adata_sc, adata_st, batch_key="dataset")
    else:
        adata_sc.obsm["harmony"] = (
            adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else np.asarray(adata_sc.X)
        )
        adata_st.obsm["harmony"] = (
            adata_st.X.toarray() if hasattr(adata_st.X, "toarray") else np.asarray(adata_st.X)
        )

    train_x, train_y = saf.construct_training_samples_memlite(
        st_adata=adata_st,
        domain_col="domain",
        n_pos=config.total_pos,
        n_neg=config.total_neg,
        k_neighbors=config.k_neighbors,
    )

    saf.train_model(train_x, train_y, epochs=config.epochs, batch_size=config.batch_size)

    predicted_domains, norm_scores, sc_st_matrix = saf.map_cells_to_domains_memlite(
        sc_adata=adata_sc,
        st_adata=adata_st,
        domain_col="domain",
        batch_size=config.pred_batch_size,
        return_prob_matrix=config.save_dense_outputs,
    )
    domain_order = np.unique(adata_st.obs["domain"].astype(str).values)

    topk_rel_indices, topk_rel_values = saf.compute_cell_relations_sparse(
        sc_adata=adata_sc,
        predicted_domains=predicted_domains,
        layer_prior=layer_relations,
        candidate_neighbors=config.candidate_neighbors,
        rel_topk=config.rel_topk,
        batch_size=config.rel_batch_size,
        filter_mode=config.filter_mode,
        soft_alpha=config.soft_alpha,
    )

    result = {
        "mode": "lite",
        "adata_sc": adata_sc,
        "adata_st": adata_st,
        "model": saf.model,
        "sc_st_matrix": sc_st_matrix,
        "sc_domain_predictions": predicted_domains,
        "sc_domain_scores": norm_scores,
        "sc_sc_matrix": None,
        "sc_sc_filtered_matrix": None,
        "sc_sc_topk_indices": topk_rel_indices,
        "sc_sc_topk_values": topk_rel_values,
        "predicted_domains": predicted_domains,
        "norm_scores": norm_scores,
        "domain_order": domain_order,
        "topk_rel_indices": topk_rel_indices,
        "topk_rel_values": topk_rel_values,
        "layer_relations": layer_relations,
        "config": _json_ready_config(config),
    }
    if config.evaluate:
        evaluate_cell2space_result(result, evaluate_k=evaluate_k, default_k=config.k_neighbors)
    return result


def _save_lite_outputs(result, out_dir, config, layer_relations):
    out_dir.mkdir(exist_ok=True, parents=True)
    adata_sc = result["adata_sc"].copy()
    adata_sc.obs["predicted_domain"] = np.asarray(result["predicted_domains"]).astype(str)

    adata_sc.write_h5ad(out_dir / "query_with_predictions.h5ad")
    result["adata_st"].write_h5ad(out_dir / "reference_processed.h5ad")

    np.save(out_dir / "sc_domain_predictions.npy", result["sc_domain_predictions"])
    np.save(out_dir / "sc_domain_scores.npy", result["sc_domain_scores"])
    np.save(out_dir / "sc_sc_topk_indices.npy", result["sc_sc_topk_indices"])
    np.save(out_dir / "sc_sc_topk_values.npy", result["sc_sc_topk_values"])

    # Backward-compatible filenames used by earlier notebooks.
    np.save(out_dir / "predicted_domains.npy", result["predicted_domains"])
    np.save(out_dir / "norm_scores.npy", result["norm_scores"])
    np.save(out_dir / "domain_order.npy", result["domain_order"])
    np.save(out_dir / "topk_rel_indices.npy", result["topk_rel_indices"])
    np.save(out_dir / "topk_rel_values.npy", result["topk_rel_values"])

    if config.save_dense_outputs and result.get("sc_st_matrix") is not None:
        np.save(out_dir / "sc_st_matrix.npy", result["sc_st_matrix"])

    if "domain" in result["adata_sc"].obs.columns:
        np.save(out_dir / "sc_true_domains.npy", result["adata_sc"].obs["domain"].astype(str).values)
    if "spatial" in result["adata_sc"].obsm:
        np.save(out_dir / "sc_spatial.npy", result["adata_sc"].obsm["spatial"])

    summary = {
        "mode": "lite",
        "outputs": {
            "sc_st_matrix": "Optional dense ST spot-by-query SC affinity matrix. Saved when save_dense_outputs=True.",
            "sc_domain_predictions": "Predicted query domain labels.",
            "sc_domain_scores": "Query-by-domain score matrix.",
            "sc_sc_topk_indices": "Sparse query-by-query top-k neighbor indices.",
            "sc_sc_topk_values": "Sparse query-by-query top-k affinity scores.",
        },
        "layer_relations": layer_relations,
        "config": _json_ready_config(config),
    }
    summary.update(_optional_metric_summary(result))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_cell2space(config=None, **kwargs):
    """
    Unified Cell2space runner.

    Examples
    --------
    result = run_cell2space(
        mode="full",
        sc_path="/path/to/query.h5ad",
        st_path="/path/to/reference.h5ad",
        out_dir="/path/to/output",
    )

    result = run_cell2space(config)
    """
    if config is None:
        config = Cell2spaceConfig(**kwargs)
    elif isinstance(config, dict):
        merged = {**config, **kwargs}
        config = Cell2spaceConfig(**merged)
    elif kwargs:
        merged = {**asdict(config), **kwargs}
        config = Cell2spaceConfig(**merged)

    config.mode = _normalize_mode(config.mode)
    _set_seed(config.seed)

    out_dir = Path(config.out_dir)
    domain_labels = _reference_domain_labels(config.st_path, config.domain_col)
    layer_relations = resolve_layer_relations(config.layer_relations, domain_labels)

    if config.mode == "full":
        result = _run_full(config, layer_relations)
        if config.save_outputs:
            _save_full_outputs(result, out_dir, config, layer_relations)
    else:
        result = _run_lite(config, layer_relations)
        if config.save_outputs:
            _save_lite_outputs(result, out_dir, config, layer_relations)

    result["out_dir"] = str(out_dir)
    result["config"] = _json_ready_config(config)
    return result


__all__ = [
    "Cell2spaceConfig",
    "DLPFC_LAYER_RELATIONS",
    "evaluate_cell2space_result",
    "resolve_layer_relations",
    "run_cell2space",
]
