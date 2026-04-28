import json
from pathlib import Path

import anndata as ad
import numpy as np

from pipeline import SpatialAffinityFunction
from utils import preprocess


DEVICE = "cuda"  # or "cpu"
SEED = 42

SC_PATH = "/path/to/query.h5ad"
ST_PATH = "/path/to/reference.h5ad"
OUT_DIR = Path("/path/to/output_memlite")

DOMAIN_COL = "Region"
SC_SPATIAL_COLS = ("array_row", "array_col")
ST_SPATIAL_COLS = ("array_row", "array_col")

N_FEATURES = 350
SELECT_HVG = "union"
USE_HARMONY = True

N_POS = 15000
N_NEG = 45000
K_NEIGHBORS = 3
EPOCHS = 300
TRAIN_BATCH_SIZE = 256

PRED_BATCH_SIZE = 512
REL_BATCH_SIZE = 256
CANDIDATE_NEIGHBORS = 256
REL_TOPK = 30

FILTER_MODE = "hard"   # "hard" or "soft"
SOFT_ALPHA = 0.1
EVAL_K = [1, 5, 10, 15]

LAYER_RELATIONS = {
    "Layer1": ["Layer1", "Layer2"],
    "Layer2": ["Layer1", "Layer2", "Layer3"],
    "Layer3": ["Layer2", "Layer3", "Layer4"],
    "Layer4": ["Layer3", "Layer4", "Layer5"],
    "Layer5": ["Layer4", "Layer5", "Layer6"],
    "Layer6": ["Layer5", "Layer6", "WM"],
    "WM": ["Layer6", "WM"],
}


def prepare_adata(adata_obj, domain_col, spatial_cols, name):
    if domain_col not in adata_obj.obs.columns:
        raise KeyError(
            f"{name}: domain column '{domain_col}' not found in adata.obs. "
            f"Available columns include: {list(adata_obj.obs.columns[:20])}"
        )

    adata_obj = adata_obj[~adata_obj.obs[domain_col].isna()].copy()
    adata_obj.obs["domain"] = adata_obj.obs[domain_col].astype(str)

    missing_cols = [col for col in spatial_cols if col not in adata_obj.obs.columns]
    if missing_cols:
        raise KeyError(
            f"{name}: spatial columns {missing_cols} not found in adata.obs. "
            f"Requested spatial columns: {spatial_cols}"
        )

    adata_obj.obsm["spatial"] = np.asarray(adata_obj.obs[list(spatial_cols)], dtype=float)
    adata_obj.obs_names_make_unique()
    adata_obj.var_names_make_unique()
    return adata_obj


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading data...")
    adata_sc = ad.read_h5ad(SC_PATH)
    adata_st = ad.read_h5ad(ST_PATH)

    adata_sc = prepare_adata(adata_sc, DOMAIN_COL, SC_SPATIAL_COLS, "scRNA-seq")
    adata_st = prepare_adata(adata_st, DOMAIN_COL, ST_SPATIAL_COLS, "ST")

    print("Preprocessing...")
    adata_sc, adata_st = preprocess(
        adata_sc,
        adata_st,
        select_hvg=SELECT_HVG,
        n_features=N_FEATURES,
    )

    saf = SpatialAffinityFunction(device=DEVICE, lr=1e-4)

    if USE_HARMONY:
        adata_sc, adata_st = saf.run_harmony(adata_sc, adata_st, batch_key="dataset")
    else:
        adata_sc.obsm["harmony"] = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else np.asarray(adata_sc.X)
        adata_st.obsm["harmony"] = adata_st.X.toarray() if hasattr(adata_st.X, "toarray") else np.asarray(adata_st.X)

    print("Constructing training samples (memlite)...")
    train_x, train_y = saf.construct_training_samples_memlite(
        st_adata=adata_st,
        domain_col="domain",
        n_pos=N_POS,
        n_neg=N_NEG,
        k_neighbors=K_NEIGHBORS,
    )

    print("Training model...")
    saf.train_model(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
    )

    print("Predicting domains (memlite)...")
    predicted_domains, norm_scores, _ = saf.map_cells_to_domains_memlite(
        sc_adata=adata_sc,
        st_adata=adata_st,
        domain_col="domain",
        batch_size=PRED_BATCH_SIZE,
        return_prob_matrix=False,
    )

    strict_acc = float(np.mean(predicted_domains == adata_sc.obs["domain"].values))
    print(f"Strict accuracy: {strict_acc:.4f}")

    print("Computing sparse cell relations...")
    topk_rel_indices, topk_rel_values = saf.compute_cell_relations_sparse(
        sc_adata=adata_sc,
        predicted_domains=predicted_domains,
        layer_prior=LAYER_RELATIONS,
        candidate_neighbors=CANDIDATE_NEIGHBORS,
        rel_topk=REL_TOPK,
        batch_size=REL_BATCH_SIZE,
        filter_mode=FILTER_MODE,
        soft_alpha=SOFT_ALPHA,
    )

    print("Evaluating neighborhood consistency...")
    cosine_scores = {}
    for k in EVAL_K:
        cosine_scores[f"cosine_k{k}"] = float(
            saf.evaluate_consistency_sparse(
                sc_adata=adata_sc,
                topk_indices=topk_rel_indices,
                k=k,
            )
        )
        print(f"cosine_k{k}: {cosine_scores[f'cosine_k{k}']:.4f}")

    np.save(OUT_DIR / "predicted_domains.npy", predicted_domains)
    np.save(OUT_DIR / "norm_scores.npy", norm_scores)
    np.save(OUT_DIR / "topk_rel_indices.npy", topk_rel_indices)
    np.save(OUT_DIR / "topk_rel_values.npy", topk_rel_values)
    np.save(OUT_DIR / "sc_true_domains.npy", adata_sc.obs["domain"].astype(str).values)
    np.save(OUT_DIR / "sc_spatial.npy", adata_sc.obsm["spatial"])

    adata_sc.obs["predicted_domain"] = predicted_domains
    adata_sc.write_h5ad(OUT_DIR / "query_with_predictions.h5ad")

    result = {
        "sc_path": SC_PATH,
        "st_path": ST_PATH,
        "seed": SEED,
        "device": DEVICE,
        "n_features": N_FEATURES,
        "select_hvg": SELECT_HVG,
        "use_harmony": USE_HARMONY,
        "n_pos": N_POS,
        "n_neg": N_NEG,
        "k_neighbors": K_NEIGHBORS,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "pred_batch_size": PRED_BATCH_SIZE,
        "rel_batch_size": REL_BATCH_SIZE,
        "candidate_neighbors": CANDIDATE_NEIGHBORS,
        "rel_topk": REL_TOPK,
        "filter_mode": FILTER_MODE,
        "soft_alpha": SOFT_ALPHA,
        "strict_accuracy": strict_acc,
    }
    result.update(cosine_scores)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
