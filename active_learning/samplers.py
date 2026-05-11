import torch
import numpy as np
import torch.nn.functional as F
import faiss


from app.schemas.pam_active_learning import ALSingleSampleScore

def _to_np(x: torch.Tensor) -> np.ndarray:
    x = F.normalize(x, p=2, dim=1)
    return x.detach().cpu().numpy().astype("float32")

def _make_hnsw_index(
    vectors: np.ndarray,
    M: int = 32, # How many neighbor connections each node keeps
    ef_search: int = 64, # How many candidate nodes are explored during search
) -> faiss.IndexHNSWFlat:
    dim = vectors.shape[1]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efSearch = ef_search
    index.add(vectors)

    return index

def uncertainty(P: torch.Tensor) -> torch.Tensor:
    """
    Multi-label uncertainty: mean binary entropy across classes.
    P: [N, C] sigmoid probabilities
    """
    # H(p) = -[p log p + (1-p) log(1-p)]
    return -(P * torch.log(P + 1e-12) + (1 - P) * torch.log(1 - P + 1e-12)).mean(dim=1)

def diversity(Z_u: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor: # INFO: 0.0331 seconds for 228 snippets

    if Z_u.numel() == 0:
        return torch.empty(0, device=Z_u.device)

    if Z_l.numel() == 0:
        return torch.zeros(Z_u.shape[0], device=Z_u.device)
    device = Z_u.device

    z_u_np = _to_np(Z_u)
    z_l_np = _to_np(Z_l)

    # OLD METHOD: pairwise distance matrix (can be large)
    # dist = torch.cdist(Z_u, Z_l)              # [N_u, N_l]
    # return dist.min(dim=1).values             # [N_u]

    # FAISS Implementation
    dim = z_l_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(z_l_np)
    distances, _ = index.search(z_u_np, k=1)
    distances = np.sqrt(distances[:, 0])
    return torch.tensor(distances, dtype=torch.float32, device=device)

    # FAISS HNSW Implementation
    # index = _make_hnsw_index(z_l_np, M=32, ef_search=64)
    # distances, _ = index.search(z_u_np, k=1)
    # distances = np.sqrt(distances[:, 0])
    # return torch.tensor(distances, dtype=torch.float32, device=device)



def density(Z_u: torch.Tensor, k: int = 15) -> torch.Tensor: # INFO: 0.0053 seconds for 228 snippets
    """
        rho(i) = 1 / avg distance to k nearest unlabeled neighbors.
    """
    if Z_u.numel() == 0:
        return torch.empty(0, device=Z_u.device)

    N_u = Z_u.shape[0]
    if N_u <= 1:
        return torch.zeros(N_u, device=Z_u.device)

    device = Z_u.device

    # OLD METHOD: pairwise distance matrix (can be large)
    # dist = torch.cdist(Z_u, Z_u)              # [N_u, N_u]
    # knn = dist.topk(k=min(k+1, N_u), largest=False).values[:, 1:]
    # avg = knn.mean(dim=1)
    # return 1.0 / (avg + 1e-8)

    # FAISS Implementation
    z_u_np = _to_np(Z_u)
    dim = z_u_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(z_u_np)
    k_eff = min(k + 1, N_u)
    distances, _ = index.search(z_u_np, k=k_eff)
    distances = distances[:, 1:]     # first neighbor is self, distance 0
    distances = np.sqrt(distances)
    avg = distances.mean(axis=1)
    scores = 1.0 / (avg + 1e-8)
    return torch.tensor(scores, dtype=torch.float32, device=device)

    # FAISS HNSW Implementation
    # z_u_np = _to_np(Z_u)
    # index = _make_hnsw_index(z_u_np, M=32, ef_search=64)
    # k_eff = min(k + 1, N_u)
    # distances, _ = index.search(z_u_np, k=k_eff)
    # distances = distances[:, 1:]     # first neighbor is usually self
    # if distances.shape[1] == 0:
    #     return torch.zeros(N_u, device=device)
    # distances = np.sqrt(distances)
    # avg = distances.mean(axis=1)
    # scores = 1.0 / (avg + 1e-8)
    # return torch.tensor(scores, dtype=torch.float32, device=device)





def random(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.rand(n, device=device)


def composite(
    uncertainty_scores: torch.Tensor,
    diversity_scores: torch.Tensor,
    density_scores: torch.Tensor,
    wu: float = 0.5,
    wd: float = 0.25,
    wr: float = 0.25,
) -> torch.Tensor:
    def normalize(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        x_min = x.min()
        x_max = x.max()
        if torch.isclose(x_min, x_max):
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    u = normalize(uncertainty_scores)
    d = normalize(diversity_scores)
    r = normalize(density_scores)

    return wu * u + wd * d + wr * r

def calculate_single_sample_scores(
    probs_row: torch.Tensor,
    sample_embedding: torch.Tensor,
    unlabeled_embeddings: torch.Tensor,
    labeled_embeddings: torch.Tensor,
    density_k: int,
    wu: float,
    wd: float,
    wr: float,
) -> ALSingleSampleScore:
    """
    Compute all acquisition values for one sample.

    probs_row: [C]
    sample_embedding: [D]
    unlabeled_embeddings: [N_u, D]
    labeled_embeddings: [N_l, D]
    """
    u_score = uncertainty(probs_row.unsqueeze(0)).squeeze(0)

    if labeled_embeddings.numel() == 0:
        d_score = None
    else:
        d_score = torch.cdist(
            sample_embedding.unsqueeze(0),
            labeled_embeddings,
        ).min(dim=1).values.squeeze(0)

    if unlabeled_embeddings.shape[0] <= 1:
        rho_score = None
    else:
        dist = torch.cdist(
            sample_embedding.unsqueeze(0),
            unlabeled_embeddings,
        ).squeeze(0)

        # remove exact self distance if present
        sorted_vals = torch.sort(dist).values
        neigh = sorted_vals[1:min(density_k + 1, sorted_vals.shape[0])]
        if neigh.numel() == 0:
            rho_score = None
        else:
            rho_score = 1.0 / (neigh.mean() + 1e-8)

    if d_score is None or rho_score is None:
        c_score = None
    else:
        # single-sample composite without global normalization is not very meaningful;
        # use raw weighted sum here, mainly for convenience
        c_score = wu * u_score + wd * d_score + wr * rho_score

    return ALSingleSampleScore(
        uncertainty=float(u_score.item()),
        diversity=None if d_score is None else float(d_score.item()),
        density=None if rho_score is None else float(rho_score.item()),
        composite=None if c_score is None else float(c_score.item()),
    )


