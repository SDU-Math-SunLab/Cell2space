import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from harmony import harmonize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from models import RelationshipMLP

class SpatialAffinityFunction:
    def __init__(self, device='cpu', lr=1e-4):
        self.device = torch.device(device)
        self.lr = lr
        self.model = None

    def run_harmony(self, sc_adata, st_adata, batch_key='dataset'):
        """Integrated scRNA and ST data into a shared latent space."""
        import scanpy as sc
        combined = sc.concat([sc_adata, st_adata], label=batch_key, keys=['sc', 'st'])
        x_dense = combined.X.toarray() if hasattr(combined.X, "toarray") else combined.X
        
        print("Starting Harmony integration...")
        harmony_out = harmonize(x_dense, combined.obs, batch_key=batch_key)
        
        sc_adata.obsm['harmony'] = harmony_out[combined.obs[batch_key] == 'sc']
        st_adata.obsm['harmony'] = harmony_out[combined.obs[batch_key] == 'st']
        return sc_adata, st_adata

    def construct_training_samples(self, st_adata, domain_col, n_pos=15000, n_neg=45000, k_neighbors=3):
        """Generates positive pairs (same domain) and negative pairs (diff domains)."""
        features = st_adata.obsm['harmony']
        domains = st_adata.obs[domain_col].values
        unique_doms, counts = np.unique(domains, return_counts=True)
        
        train_data, train_labels = [], []

        # Positive sampling
        for dom, count in zip(unique_doms, counts):
            idx = np.where(domains == dom)[0]
            if len(idx) < 2: continue
            
            n_samples = int((count / len(domains)) * n_pos)
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(idx))).fit(features[idx])
            _, indices = nbrs.kneighbors(features[idx])
            
            curr_pos = 0
            for i, neighbors in enumerate(indices[:, 1:]):
                for nb in neighbors:
                    if curr_pos >= n_samples: break
                    pair = np.concatenate([features[idx[i]], features[idx[nb]]])
                    train_data.append(pair)
                    train_labels.append(1)
                    curr_pos += 1

        # Negative sampling
        for _ in range(n_neg):
            i = np.random.randint(len(features))
            diff_idx = np.where(domains != domains[i])[0]
            if len(diff_idx) == 0: continue
            j = np.random.choice(diff_idx)
            train_data.append(np.concatenate([features[i], features[j]]))
            train_labels.append(0)

        return torch.tensor(np.array(train_data), dtype=torch.float32), \
               torch.tensor(np.array(train_labels), dtype=torch.float32)

    def train_model(self, x, y, epochs=300, batch_size=256):
        input_dim = x.shape[1]
        self.model = RelationshipMLP(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Average Loss = {total_loss/len(loader):.8f}")

    def map_cells_to_domains(self, sc_adata, st_adata, domain_col):
        """Predicts ST domain for each single cell."""
        self.model.eval()
        sc_feat = torch.tensor(sc_adata.obsm['harmony'], dtype=torch.float32).to(self.device)
        st_feat = torch.tensor(st_adata.obsm['harmony'], dtype=torch.float32).to(self.device)
        unique_doms = np.unique(st_adata.obs[domain_col])
        
        prob_matrix = np.zeros((len(st_feat), len(sc_feat)))
        with torch.no_grad():
            for i in tqdm(range(len(sc_feat)), desc="Predicting cell-spot relationships"):
                curr_sc = sc_feat[i].repeat(len(st_feat), 1)
                combined = torch.cat([st_feat, curr_sc], dim=1)
                prob_matrix[:, i] = self.model(combined).squeeze().cpu().numpy()

        domain_mask = np.array([st_adata.obs[domain_col] == d for d in unique_doms]).astype(float).T
        scores = prob_matrix.T @ domain_mask
        counts = domain_mask.sum(axis=0)
        norm_scores = (scores * scores) / (counts + 1e-10)
        
        predictions = unique_doms[np.argmax(norm_scores, axis=1)]
        return predictions, prob_matrix

    def evaluate_consistency(self, sc_adata, rel_matrix, dist_matrix, layer_prior, sc_preds, k=3):
        """Evaluates prediction consistency using prior knowledge and spatial distances."""
        # Filter relationship matrix by domain adjacency prior
        filtered_rel = np.zeros_like(rel_matrix)
        for i in range(len(sc_preds)):
            for j in range(len(sc_preds)):
                if sc_preds[j] in layer_prior.get(sc_preds[i], []):
                    filtered_rel[i, j] = rel_matrix[i, j]

        sc_expr = sc_adata.X.toarray() if hasattr(sc_adata.X, "toarray") else sc_adata.X
        similarities = []
        
        for i in range(len(sc_expr)):
            # Neighbors by learned relationship
            rel_idx = np.argsort(filtered_rel[i])[::-1][:k]
            # Neighbors by spatial coordinate distance
            dist_idx = np.argsort(dist_matrix[i])[::-1][:k]
            
            expr_rel = np.mean(sc_expr[rel_idx], axis=0).reshape(1, -1)
            expr_dist = np.mean(sc_expr[dist_idx], axis=0).reshape(1, -1)
            
            sim = cosine_similarity(expr_rel, expr_dist)[0][0]
            similarities.append(sim)
            
        return np.mean(similarities)
    
    def compute_cell_relations(self, sc_adata, batch_size=256):
        """Computes a Cell-to-Cell relationship matrix."""
        self.model.eval()
        sc_feat = torch.tensor(sc_adata.obsm['harmony'], dtype=torch.float32).to(self.device)
        num_cells = len(sc_feat)
        rel_matrix = np.zeros((num_cells, num_cells))

        with torch.no_grad():
            for i in tqdm(range(num_cells), desc="Computing Cell-Cell relations"):
                curr_cell = sc_feat[i].unsqueeze(0) # (1, feat_dim)
                
                # Predict in batches to avoid OOM
                for start in range(0, num_cells, batch_size):
                    end = min(start + batch_size, num_cells)
                    batch_cells = sc_feat[start:end]
                    
                    # Expand curr_cell to match batch size
                    curr_cell_expanded = curr_cell.expand(len(batch_cells), -1)
                    combined = torch.cat([curr_cell_expanded, batch_cells], dim=1)
                    
                    rel_matrix[i, start:end] = self.model(combined).squeeze().cpu().numpy()
        
        return rel_matrix

    def evaluate_consistency(self, sc_adata, cell_rel_matrix, dist_matrix, layer_prior, sc_preds, k=3):
        """Evaluates prediction consistency using Cell-Cell relationships."""
        num_cells = len(sc_preds)
        filtered_rel = np.zeros_like(cell_rel_matrix)

        # Apply spatial prior to the cell-cell relationship matrix
        for i in range(num_cells):
            domain_i = sc_preds[i]
            allowed_domains = layer_prior.get(domain_i, [])
            for j in range(num_cells):
                if sc_preds[j] in allowed_domains:
                    filtered_rel[i, j] = cell_rel_matrix[i, j]

        # Convert X to dense for similarity calculation
        sc_expr = sc_adata.X.toarray() if hasattr(sc_adata.X, "toarray") else sc_adata.X
        similarities = []
        
        for i in range(num_cells):
            # Indices of top K neighbors in predicted relationship space
            rel_idx = np.argsort(filtered_rel[i])[::-1][:k]
            # Indices of top K neighbors in actual spatial coordinate space
            dist_idx = np.argsort(dist_matrix[i])[::-1][:k]
            
            # Mean expression of neighbors
            expr_rel = np.mean(sc_expr[rel_idx], axis=0).reshape(1, -1)
            expr_dist = np.mean(sc_expr[dist_idx], axis=0).reshape(1, -1)
            
            sim = cosine_similarity(expr_rel, expr_dist)[0][0]
            similarities.append(sim)
            
        return np.mean(similarities)