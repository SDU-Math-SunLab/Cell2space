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

    def construct_training_samples_memlite(self, st_adata, domain_col, n_pos=15000, n_neg=45000, k_neighbors=3):
        """Memory-efficient version of training pair construction."""
        features = np.asarray(st_adata.obsm['harmony'], dtype=np.float32)
        domains = st_adata.obs[domain_col].values
        unique_doms, counts = np.unique(domains, return_counts=True)
        proportions = counts / len(domains)
        pos_per_domain = np.maximum((proportions * n_pos).astype(int), 1)

        pos_pairs = []
        for dom, n_samples in zip(unique_doms, pos_per_domain):
            idx = np.where(domains == dom)[0]
            if len(idx) < 2:
                continue
            nbrs = NearestNeighbors(
                n_neighbors=min(k_neighbors + 1, len(idx)),
                algorithm="auto"
            ).fit(features[idx])
            _, indices = nbrs.kneighbors(features[idx])

            curr_pairs = []
            for i, neighbors in enumerate(indices[:, 1:]):
                for nb in neighbors:
                    curr_pairs.append((idx[i], idx[nb]))
                    if len(curr_pairs) >= n_samples:
                        break
                if len(curr_pairs) >= n_samples:
                    break
            pos_pairs.extend(curr_pairs)

        neg_pairs = []
        while len(neg_pairs) < n_neg:
            i = np.random.randint(len(features))
            diff_idx = np.where(domains != domains[i])[0]
            if len(diff_idx) == 0:
                continue
            j = np.random.choice(diff_idx)
            neg_pairs.append((i, j))

        total_pairs = len(pos_pairs) + len(neg_pairs)
        feat_dim = features.shape[1]
        train_x = np.empty((total_pairs, feat_dim * 2), dtype=np.float32)
        train_y = np.empty(total_pairs, dtype=np.float32)

        cursor = 0
        for i, j in pos_pairs:
            train_x[cursor, :feat_dim] = features[i]
            train_x[cursor, feat_dim:] = features[j]
            train_y[cursor] = 1.0
            cursor += 1

        for i, j in neg_pairs:
            train_x[cursor, :feat_dim] = features[i]
            train_x[cursor, feat_dim:] = features[j]
            train_y[cursor] = 0.0
            cursor += 1

        return torch.from_numpy(train_x), torch.from_numpy(train_y)

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

    def map_cells_to_domains_memlite(self, sc_adata, st_adata, domain_col, batch_size=512, return_prob_matrix=False):
        """
        Memory-efficient domain prediction.

        Computes domain scores in chunks and avoids materializing the full spot x cell
        probability matrix unless explicitly requested.
        """
        self.model.eval()
        sc_feat = np.asarray(sc_adata.obsm['harmony'], dtype=np.float32)
        st_feat = np.asarray(st_adata.obsm['harmony'], dtype=np.float32)
        st_domains = st_adata.obs[domain_col].values
        unique_doms = np.unique(st_domains)
        dom_to_idx = {d: np.where(st_domains == d)[0] for d in unique_doms}

        sc_tensor = torch.tensor(sc_feat, dtype=torch.float32).to(self.device)
        st_tensor = torch.tensor(st_feat, dtype=torch.float32).to(self.device)
        norm_scores = np.zeros((len(sc_feat), len(unique_doms)), dtype=np.float32)

        prob_matrix = None
        if return_prob_matrix:
            prob_matrix = np.zeros((len(st_feat), len(sc_feat)), dtype=np.float32)

        with torch.no_grad():
            for d_i, domain in enumerate(tqdm(unique_doms, desc="Scoring domains")):
                idx = dom_to_idx[domain]
                domain_spots = st_tensor[idx]
                domain_size = len(idx)

                for start in range(0, len(sc_feat), batch_size):
                    end = min(start + batch_size, len(sc_feat))
                    sc_batch = sc_tensor[start:end]
                    partial_sum = torch.zeros(end - start, device=self.device)

                    for st_start in range(0, domain_size, batch_size):
                        st_end = min(st_start + batch_size, domain_size)
                        st_batch = domain_spots[st_start:st_end]
                        bsz = sc_batch.shape[0]
                        n_spots = st_batch.shape[0]

                        left = sc_batch.unsqueeze(1).expand(bsz, n_spots, -1)
                        right = st_batch.unsqueeze(0).expand(bsz, n_spots, -1)
                        combined = torch.cat([right, left], dim=2).reshape(bsz * n_spots, -1)
                        pred = self.model(combined).reshape(bsz, n_spots)
                        partial_sum += pred.sum(dim=1)

                        if return_prob_matrix:
                            prob_matrix[idx[st_start:st_end], start:end] = pred.T.cpu().numpy()

                    norm_scores[start:end, d_i] = (partial_sum / domain_size).cpu().numpy()

        predictions = unique_doms[np.argmax(norm_scores, axis=1)]
        return predictions, norm_scores, prob_matrix

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

    def _score_candidate_pairs(self, sc_feat, candidate_idx, batch_size=256):
        """Scores only candidate cell-cell pairs instead of a dense all-vs-all matrix."""
        num_cells, n_cands = candidate_idx.shape
        scores = np.zeros((num_cells, n_cands), dtype=np.float32)
        sc_tensor = torch.tensor(sc_feat, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, num_cells, batch_size), desc="Scoring sparse cell relations"):
                end = min(start + batch_size, num_cells)
                anchor_batch = sc_tensor[start:end]
                cand_batch = candidate_idx[start:end]
                flat_idx = cand_batch.reshape(-1)
                neigh_batch = sc_tensor[flat_idx].reshape(end - start, n_cands, -1)

                left = anchor_batch.unsqueeze(1).expand(-1, n_cands, -1)
                combined = torch.cat([left, neigh_batch], dim=2).reshape((end - start) * n_cands, -1)
                pred = self.model(combined).reshape(end - start, n_cands)
                scores[start:end] = pred.cpu().numpy()

        return scores

    def compute_cell_relations_sparse(
        self,
        sc_adata,
        predicted_domains=None,
        layer_prior=None,
        candidate_neighbors=256,
        rel_topk=30,
        batch_size=256,
        filter_mode="hard",
        soft_alpha=0.1,
    ):
        """
        Memory-efficient alternative to compute_cell_relations.

        Returns sparse top-k neighbor indices and scores instead of an n x n matrix.
        """
        sc_feat = np.asarray(sc_adata.obsm['harmony'], dtype=np.float32)
        n_neighbors = min(candidate_neighbors + 1, len(sc_feat))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(sc_feat)
        _, indices = nbrs.kneighbors(sc_feat)
        candidate_idx = indices[:, 1:]

        scores = self._score_candidate_pairs(sc_feat, candidate_idx, batch_size=batch_size)

        if predicted_domains is not None and layer_prior:
            for i in range(candidate_idx.shape[0]):
                allowed = set(layer_prior.get(predicted_domains[i], [predicted_domains[i]]))
                keep = np.array([predicted_domains[j] in allowed for j in candidate_idx[i]], dtype=bool)
                if filter_mode == "soft":
                    scores[i][~keep] *= soft_alpha
                elif filter_mode == "hard" and keep.any():
                    keep_idx = candidate_idx[i][keep]
                    keep_scores = scores[i][keep]
                    fill_n = candidate_idx.shape[1] - len(keep_idx)
                    if fill_n > 0:
                        candidate_idx[i] = np.concatenate([keep_idx, np.repeat(keep_idx[-1], fill_n)])
                        scores[i] = np.concatenate([keep_scores, np.repeat(keep_scores[-1], fill_n)])
                    else:
                        candidate_idx[i] = keep_idx
                        scores[i] = keep_scores

        if rel_topk < scores.shape[1]:
            top_idx = np.argpartition(-scores, rel_topk - 1, axis=1)[:, :rel_topk]
            row_idx = np.arange(scores.shape[0])[:, None]
            top_scores = scores[row_idx, top_idx]
            top_neighbors = candidate_idx[row_idx, top_idx]
            order = np.argsort(-top_scores, axis=1)
            top_scores = top_scores[row_idx, order]
            top_neighbors = top_neighbors[row_idx, order]
        else:
            top_scores = scores
            top_neighbors = candidate_idx

        return top_neighbors.astype(np.int32), top_scores.astype(np.float32)

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

    def evaluate_consistency_sparse(self, sc_adata, topk_indices, k=3):
        """
        Memory-efficient consistency evaluation using sparse top-k predicted neighbors.

        The true spatial neighborhood is computed directly by KNN on coordinates, so no
        dense distance matrix is needed.
        """
        sc_expr = sc_adata.X.toarray() if hasattr(sc_adata.X, "toarray") else sc_adata.X
        spatial = sc_adata.obsm["spatial"]

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(spatial)), algorithm="auto").fit(spatial)
        _, true_idx = nbrs.kneighbors(spatial)
        true_idx = true_idx[:, 1:]

        similarities = []
        for i in range(len(sc_expr)):
            pred_idx = topk_indices[i][:k]
            expr_rel = np.mean(sc_expr[pred_idx], axis=0).reshape(1, -1)
            expr_true = np.mean(sc_expr[true_idx[i, :k]], axis=0).reshape(1, -1)
            sim = cosine_similarity(expr_rel, expr_true)[0][0]
            similarities.append(sim)

        return np.mean(similarities)
