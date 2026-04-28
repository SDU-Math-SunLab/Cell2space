import os
import random
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from harmony import harmonize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import warnings

from models import MLP
from utils import preprocess, create_distance_matrix

warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
SC_PATH = 'xxx/SC.h5ad'
ST_PATH = 'xxx/ST.h5ad'
K_NEIGHBORS = 3
TOTAL_POS = 15000
TOTAL_NEG = 45000
EPOCHS = 300
BATCH_SIZE = 256
DOMAIN_COL = 'Region'
SC_SPATIAL_COLS = ('array_row', 'array_col')
ST_SPATIAL_COLS = ('array_row', 'array_col')
SEED = 0

LAYER_RELATIONS = {}
# LAYER_RELATIONS = {
#     'Layer1': ['Layer2', 'Layer1'],
#     'Layer2': ['Layer1', 'Layer3', 'Layer2'],
#     'Layer3': ['Layer2', 'Layer4', 'Layer3'],
#     'Layer4': ['Layer3', 'Layer5', 'Layer4'],
#     'Layer5': ['Layer4', 'Layer6', 'Layer5'],
#     'Layer6': ['Layer5', 'WM', 'Layer6'],
#     'WM': ['Layer6', 'WM'],
# }

def _resolve_device(device=None):
    if device is None:
        return DEVICE
    if isinstance(device, torch.device):
        return device
    return torch.device(device if torch.cuda.is_available() else 'cpu')


def _set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Avoid strict deterministic mode because CuBLAS linear layers can fail on CUDA >= 10.2.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _prepare_adata(adata_obj, domain_col, spatial_cols, name, require_domain=True, require_spatial=True):
    if domain_col not in adata_obj.obs.columns:
        if not require_domain:
            adata_obj = adata_obj.copy()
            adata_obj.obs_names_make_unique()
            adata_obj.var_names_make_unique()
            if spatial_cols is not None and all(col in adata_obj.obs.columns for col in spatial_cols):
                adata_obj.obsm['spatial'] = np.array(adata_obj.obs[list(spatial_cols)])
            return adata_obj
        raise KeyError(
            f"{name}: domain column '{domain_col}' not found in adata.obs. "
            f"Available columns include: {list(adata_obj.obs.columns[:20])}"
        )

    if require_domain:
        adata_obj = adata_obj[~adata_obj.obs[domain_col].isna()].copy()
        adata_obj.obs['domain'] = adata_obj.obs[domain_col].astype(str)
    else:
        adata_obj = adata_obj.copy()
        if not adata_obj.obs[domain_col].isna().any():
            adata_obj.obs['domain'] = adata_obj.obs[domain_col].astype(str)

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
        adata_obj.obsm['spatial'] = np.array(adata_obj.obs[list(spatial_cols)])
    elif require_spatial and 'spatial' not in adata_obj.obsm:
        raise KeyError(
            f"{name}: no spatial information found. Provide spatial_cols or set adata.obsm['spatial'] first."
        )

    adata_obj.obs_names_make_unique()
    adata_obj.var_names_make_unique()
    return adata_obj


def evaluate_full_result(result, evaluate_k=(1, 3, 5, 10, 15), default_k=3):
    """Evaluate a full-mode Cell2space result when query labels/spatial coordinates are available."""
    adata_sc = result["adata_sc"]
    predicted_domains = result["predicted_domains"]
    filtered_rel = result["filtered_rel"]

    metrics = {}
    if "domain" in adata_sc.obs.columns:
        metrics["domain_accuracy"] = float(np.mean(predicted_domains == adata_sc.obs["domain"].values))

    if "spatial" in adata_sc.obsm:
        dist_mat = create_distance_matrix(adata_sc.obsm["spatial"])
        sc_expr = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc.X
        evaluate_k = tuple(sorted(set(evaluate_k)))
        max_eval_k = max(evaluate_k)
        rel_order = np.argsort(filtered_rel, axis=1)[:, ::-1][:, :max_eval_k]
        dist_order = np.argsort(dist_mat, axis=1)[:, ::-1][:, :max_eval_k]

        for eval_k in evaluate_k:
            all_sims = []
            for i in range(sc_expr.shape[0]):
                rel_nb = rel_order[i, :eval_k]
                dist_nb = dist_order[i, :eval_k]
                mean_rel = np.mean(sc_expr[rel_nb], axis=0).reshape(1, -1)
                mean_dist = np.mean(sc_expr[dist_nb], axis=0).reshape(1, -1)
                all_sims.append(cosine_similarity(mean_rel, mean_dist)[0][0])
            metrics[f"cosine_k{eval_k}"] = float(np.mean(all_sims))

        default_key = f"cosine_k{default_k}"
        metrics["cosine_similarity"] = metrics.get(default_key, metrics[f"cosine_k{max_eval_k}"])

    return metrics


def run_pipeline(
    sc_path=None,
    st_path=None,
    layer_relations=None,
    domain_col=None,
    sc_spatial_cols=None,
    st_spatial_cols=None,
    k_neighbors=None,
    total_pos=None,
    total_neg=None,
    epochs=None,
    batch_size=None,
    device=None,
    select_hvg='union',
    n_features=350,
    use_harmony=True,
    filter_mode='hard',
    soft_alpha=0.1,
    evaluate_k=(1, 3, 5, 10, 15),
    evaluate=False,
    seed=None,
):
    sc_path = sc_path or SC_PATH
    st_path = st_path or ST_PATH
    domain_col = domain_col or DOMAIN_COL
    sc_spatial_cols = SC_SPATIAL_COLS if sc_spatial_cols is None else sc_spatial_cols
    st_spatial_cols = ST_SPATIAL_COLS if st_spatial_cols is None else st_spatial_cols
    k_neighbors = k_neighbors or K_NEIGHBORS
    total_pos = total_pos or TOTAL_POS
    total_neg = total_neg or TOTAL_NEG
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    layer_relations = LAYER_RELATIONS if layer_relations is None else layer_relations
    device = _resolve_device(device)
    seed = SEED if seed is None else seed
    _set_seed(seed)

    # 1. Load and Clean Data
    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)

    adata_sc = _prepare_adata(
        adata_sc,
        domain_col,
        sc_spatial_cols,
        'scRNA-seq',
        require_domain=evaluate,
        require_spatial=evaluate,
    )
    adata_st = _prepare_adata(
        adata_st,
        domain_col,
        st_spatial_cols,
        'ST',
        require_domain=True,
        require_spatial=False,
    )

    # 2. Preprocessing
    adata_sc, adata_st = preprocess(adata_sc, adata_st, select_hvg=select_hvg, n_features=n_features)

    # 3. Harmony Integration
    if use_harmony:
        print('Starting Harmony integration...')
        start_time = datetime.now()
        adata_combined = sc.concat([adata_sc, adata_st], join='outer', label='dataset', keys=['sc', 'st'])
        X = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
        adata_combined.obsm['X_harmony'] = harmonize(X, adata_combined.obs, batch_key='dataset')
        
        adata_sc.obsm['harmony'] = adata_combined[adata_combined.obs['dataset'] == 'sc'].obsm['X_harmony']
        adata_st.obsm['harmony'] = adata_combined[adata_combined.obs['dataset'] == 'st'].obsm['X_harmony']
        print(f"Harmony integration done. Time: {datetime.now() - start_time}")
    else:
        print('Skipping Harmony integration. Using preprocessed expression directly.')
        adata_sc.obsm['harmony'] = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else np.asarray(adata_sc.X)
        adata_st.obsm['harmony'] = adata_st.X.toarray() if hasattr(adata_st.X, "toarray") else np.asarray(adata_st.X)

    # 4. Generate Training Data
    spot_vectors = adata_st.obsm['harmony']
    spot_vectors_tensor = torch.tensor(spot_vectors, dtype=torch.float32)
    spot_domains = adata_st.obs['domain'].values
    unique_domains, domain_counts = np.unique(spot_domains, return_counts=True)
    domain_proportions = domain_counts / len(spot_vectors)

    train_data, train_labels = [], []
    pos_per_domain = (domain_proportions * total_pos).astype(int)

    # Positive Sampling
    for domain, num_positive in zip(unique_domains, pos_per_domain):
        idx = np.where(spot_domains == domain)[0]
        if len(idx) > 1:
            nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(spot_vectors[idx])
            _, indices = nbrs.kneighbors(spot_vectors[idx])
            
            pairs = []
            for i, neighbors in enumerate(indices[:, 1:]):
                for nb in neighbors:
                    pairs.append((idx[i], idx[nb]))
                    if len(pairs) >= num_positive: break
                if len(pairs) >= num_positive: break
            
            for id1, id2 in pairs:
                train_data.append(torch.cat((spot_vectors_tensor[id1], spot_vectors_tensor[id2])))
                train_labels.append(1)

    # Negative Sampling
    neg_idx_list = np.random.randint(0, len(spot_vectors), size=total_neg)
    for idx1 in neg_idx_list:
        diff_spots = np.where(spot_domains != spot_domains[idx1])[0]
        idx2 = np.random.choice(diff_spots)
        train_data.append(torch.cat((spot_vectors_tensor[idx1], spot_vectors_tensor[idx2])))
        train_labels.append(0)

    # 5. Train MLP
    train_x = torch.stack(train_data).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.float32).to(device)
    mlp = MLP(train_x.shape[1]).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=data_loader_generator,
    )
    for epoch in range(epochs):
        mlp.train()
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = mlp(bx.to(device))
            loss = criterion(pred, by.view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

    # 6. Prediction & Normalization
    mlp.eval()
    sc_vectors_tensor = torch.tensor(adata_sc.obsm['harmony'], dtype=torch.float32).to(device)
    num_cells = len(sc_vectors_tensor)
    num_spots = len(spot_vectors_tensor)
    prob_matrix = np.zeros((num_spots, num_cells))

    with torch.no_grad():
        spot_batch = spot_vectors_tensor.unsqueeze(1).to(device)
        for i in tqdm(range(num_cells), desc="Predicting Cell-Spot relationships"):
            cell_vec = sc_vectors_tensor[i].unsqueeze(0).expand(num_spots, -1)
            combined = torch.cat((spot_batch.squeeze(1), cell_vec), dim=1)
            prob_matrix[:, i] = mlp(combined).view(-1).cpu().numpy()

    # Squaring Normalization Logic
    indicator = np.zeros((num_spots, len(unique_domains)))
    for i, d in enumerate(unique_domains):
        indicator[:, i] = (spot_domains == d).astype(float)
    
    scores = prob_matrix.T @ indicator
    counts = indicator.sum(axis=0)
    norm_scores = (scores * scores) / (counts + 1e-10)
    
    predicted_domains = unique_domains[np.argmax(norm_scores, axis=1)]
    print("Predicted domain counts:")
    print(dict(zip(*np.unique(predicted_domains, return_counts=True))))

    # 7. Cell-Cell Relationship Matrix
    cell_rel_matrix = np.zeros((num_cells, num_cells))
    with torch.no_grad():
        for i in tqdm(range(num_cells), desc="Computing Cell-Cell relations"):
            cell_i = sc_vectors_tensor[i].unsqueeze(0)
            for start in range(0, num_cells, batch_size):
                end = min(start + batch_size, num_cells)
                batch_j = sc_vectors_tensor[start:end]
                combined = torch.cat((cell_i.expand(len(batch_j), -1), batch_j), dim=1)
                cell_rel_matrix[i, start:end] = mlp(combined).view(-1).cpu().numpy()

    # 8. Filter by Prior Knowledge
    filtered_rel = np.zeros_like(cell_rel_matrix)
    for i in range(num_cells):
        allowed = layer_relations.get(predicted_domains[i], [])
        for j in range(num_cells):
            if filter_mode == 'none':
                filtered_rel[i, j] = cell_rel_matrix[i, j]
            elif predicted_domains[j] in allowed:
                filtered_rel[i, j] = cell_rel_matrix[i, j]
            elif filter_mode == 'soft':
                filtered_rel[i, j] = soft_alpha * cell_rel_matrix[i, j]
            elif filter_mode == 'hard':
                filtered_rel[i, j] = 0
            else:
                raise ValueError("filter_mode must be one of: 'hard', 'soft', 'none'")

    result = {
        'adata_sc': adata_sc,
        'adata_st': adata_st,
        'model': mlp,
        'sc_st_matrix': prob_matrix,
        'sc_domain_scores': norm_scores,
        'sc_domain_predictions': predicted_domains,
        'sc_sc_matrix': cell_rel_matrix,
        'sc_sc_filtered_matrix': filtered_rel,
        'prob_matrix': prob_matrix,
        'predicted_domains': predicted_domains,
        'cell_rel_matrix': cell_rel_matrix,
        'filtered_rel': filtered_rel,
        'norm_scores': norm_scores,
        'unique_domains': unique_domains,
        'spot_domains': spot_domains,
        'config': {
            'sc_path': sc_path,
            'st_path': st_path,
            'domain_col': domain_col,
            'sc_spatial_cols': sc_spatial_cols,
            'st_spatial_cols': st_spatial_cols,
            'k_neighbors': k_neighbors,
            'total_pos': total_pos,
            'total_neg': total_neg,
            'epochs': epochs,
            'batch_size': batch_size,
            'select_hvg': select_hvg,
            'n_features': n_features,
            'use_harmony': use_harmony,
            'filter_mode': filter_mode,
            'soft_alpha': soft_alpha,
            'evaluate_k': evaluate_k,
            'evaluate': evaluate,
            'seed': seed,
            'device': str(device),
        },
    }
    if evaluate:
        metrics = evaluate_full_result(result, evaluate_k=evaluate_k, default_k=k_neighbors)
        result.update(metrics)
    return result

if __name__ == "__main__":
    run_pipeline()
