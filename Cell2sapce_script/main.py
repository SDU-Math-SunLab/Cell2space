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

def run_pipeline():
    # 1. Load and Clean Data
    adata_sc = ad.read_h5ad(SC_PATH)
    adata_st = ad.read_h5ad(ST_PATH)

    adata_st = adata_st[~adata_st.obs['Region'].isna()].copy()
    adata_sc = adata_sc[~adata_sc.obs['Region'].isna()].copy()
    
    # Ensure string types to prevent indexing errors
    adata_st.obs['domain'] = adata_st.obs['Region'].astype(str)
    adata_sc.obs['domain'] = adata_sc.obs['Region'].astype(str)

    adata_st.obsm['spatial'] = np.array(adata_st.obs[['array_row', 'array_col']])
    adata_sc.obsm['spatial'] = np.array(adata_sc.obs[['array_row', 'array_col']])

    # 2. Preprocessing
    adata_sc, adata_st = preprocess(adata_sc, adata_st, select_hvg='union')

    # 3. Harmony Integration
    print('Starting Harmony integration...')
    start_time = datetime.now()
    adata_combined = sc.concat([adata_sc, adata_st], join='outer', label='dataset', keys=['sc', 'st'])
    X = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
    adata_combined.obsm['X_harmony'] = harmonize(X, adata_combined.obs, batch_key='dataset')
    
    adata_sc.obsm['harmony'] = adata_combined[adata_combined.obs['dataset'] == 'sc'].obsm['X_harmony']
    adata_st.obsm['harmony'] = adata_combined[adata_combined.obs['dataset'] == 'st'].obsm['X_harmony']
    print(f"Harmony integration done. Time: {datetime.now() - start_time}")

    # 4. Generate Training Data
    spot_vectors = adata_st.obsm['harmony']
    spot_vectors_tensor = torch.tensor(spot_vectors, dtype=torch.float32)
    spot_domains = adata_st.obs['domain'].values
    unique_domains, domain_counts = np.unique(spot_domains, return_counts=True)
    domain_proportions = domain_counts / len(spot_vectors)

    train_data, train_labels = [], []
    pos_per_domain = (domain_proportions * TOTAL_POS).astype(int)

    # Positive Sampling
    for domain, num_positive in zip(unique_domains, pos_per_domain):
        idx = np.where(spot_domains == domain)[0]
        if len(idx) > 1:
            nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, algorithm='ball_tree').fit(spot_vectors[idx])
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
    neg_idx_list = np.random.randint(0, len(spot_vectors), size=TOTAL_NEG)
    for idx1 in neg_idx_list:
        diff_spots = np.where(spot_domains != spot_domains[idx1])[0]
        idx2 = np.random.choice(diff_spots)
        train_data.append(torch.cat((spot_vectors_tensor[idx1], spot_vectors_tensor[idx2])))
        train_labels.append(0)

    # 5. Train MLP
    train_x = torch.stack(train_data).to(DEVICE)
    train_y = torch.tensor(train_labels, dtype=torch.float32).to(DEVICE)
    mlp = MLP(train_x.shape[1]).to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        mlp.train()
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = mlp(bx.to(DEVICE))
            loss = criterion(pred, by.view(-1, 1).to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

    # 6. Prediction & Normalization
    mlp.eval()
    sc_vectors_tensor = torch.tensor(adata_sc.obsm['harmony'], dtype=torch.float32).to(DEVICE)
    num_cells = len(sc_vectors_tensor)
    num_spots = len(spot_vectors_tensor)
    prob_matrix = np.zeros((num_spots, num_cells))

    with torch.no_grad():
        spot_batch = spot_vectors_tensor.unsqueeze(1).to(DEVICE)
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
    accuracy = np.sum(predicted_domains == adata_sc.obs['domain'].values) / num_cells
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # 7. Cell-Cell Relationship Matrix
    cell_rel_matrix = np.zeros((num_cells, num_cells))
    with torch.no_grad():
        for i in tqdm(range(num_cells), desc="Computing Cell-Cell relations"):
            cell_i = sc_vectors_tensor[i].unsqueeze(0)
            for start in range(0, num_cells, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_cells)
                batch_j = sc_vectors_tensor[start:end]
                combined = torch.cat((cell_i.expand(len(batch_j), -1), batch_j), dim=1)
                cell_rel_matrix[i, start:end] = mlp(combined).view(-1).cpu().numpy()

    # 8. Filter by Prior Knowledge
    filtered_rel = np.zeros_like(cell_rel_matrix)
    for i in range(num_cells):
        allowed = LAYER_RELATIONS.get(predicted_domains[i], [])
        for j in range(num_cells):
            if predicted_domains[j] in allowed:
                filtered_rel[i, j] = cell_rel_matrix[i, j]

    # 9. Consistency Evaluation
    print("Calculating final consistency metrics...")
    dist_mat = create_distance_matrix(adata_sc.obsm['spatial'])
    
    sc_expr = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc.X
    all_sims = []
    for i in range(num_cells):
        rel_nb = np.argsort(filtered_rel[i])[::-1][:K_NEIGHBORS]
        dist_nb = np.argsort(dist_mat[i])[::-1][:K_NEIGHBORS]
        
        mean_rel = np.mean(sc_expr[rel_nb], axis=0).reshape(1, -1)
        mean_dist = np.mean(sc_expr[dist_nb], axis=0).reshape(1, -1)
        
        sim = cosine_similarity(mean_rel, mean_dist)[0][0]
        all_sims.append(sim)

    print(f"Average Cosine Similarity: {np.mean(all_sims):.4f}")

if __name__ == "__main__":
    run_pipeline()