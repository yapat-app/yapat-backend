import torch
import numpy as np
import torch.nn.functional as F
import faiss


from app.schemas.pam_active_learning import ALSingleSampleScore


def normalize_diversity(d: torch.Tensor) -> torch.Tensor:
    # diversity already in [0, 1], clamp makes sure no outliers.
    return torch.clamp(d, 0.0, 1.0)

def normalize_density(
    r: torch.Tensor,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> torch.Tensor:
    # density values will have no bounds e.g. 1/0.03 = 33.33
    # therefore we use quantile-based normalization to mitigate outliers, and clamp to [0, 1].
    if r.numel() == 0:
        return r

    lo = torch.quantile(r, q_low)
    hi = torch.quantile(r, q_high)

    if torch.isclose(lo, hi):
        return torch.zeros_like(r)

    return torch.clamp((r - lo) / (hi - lo), 0.0, 1.0)


def _to_np(x: torch.Tensor) -> np.ndarray:
    x = F.normalize(x, p=2, dim=1) # normalizing the embeddings using l2- norm
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
    Multi-label uncertainty normalized to [0, 1].

    Raw binary entropy has max log(2) at p=0.5.
    Dividing by log(2) gives:
        0 = confident
        1 = maximally uncertain
    """
    entropy = -(
        P * torch.log(P + 1e-12)
        + (1 - P) * torch.log(1 - P + 1e-12)
    ).mean(dim=1)

    return torch.clamp(entropy / np.log(2), 0.0, 1.0)

def diversity(Z_u: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
    """
    Diversity / novelty normalized to [0, 1].

    Uses distance to nearest labeled embedding.
    Since your observed L2-normalized distances are usually <= 1,
    we clip values to [0, 1].
    """
    if Z_u.numel() == 0:
        return torch.empty(0, device=Z_u.device)

    if Z_l.numel() == 0:
        return torch.zeros(Z_u.shape[0], device=Z_u.device)

    device = Z_u.device

    z_u_np = _to_np(Z_u)
    z_l_np = _to_np(Z_l)

    dim = z_l_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(z_l_np)

    distances, _ = index.search(z_u_np, k=1)
    distances = np.sqrt(distances[:, 0])

    scores = torch.tensor(distances, dtype=torch.float32, device=device)

    return torch.clamp(scores, 0.0, 1.0)

def density(
    Z_u: torch.Tensor,
    k: int = 15,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> torch.Tensor:
    """
    Density / representativeness normalized to [0, 1].

    Raw density:
        rho(i) = 1 / avg distance to k nearest unlabeled neighbors

    Then percentile-normalized:
        0 = sparse / outlier-like
        1 = dense / representative

    Uses HNSW approximate nearest-neighbor index (O(n log n)) rather than the
    brute-force flat index (O(n²)) so that large datasets (100k+ snippets) don't
    stall the retrain worker for 30–90 seconds.
    """
    if Z_u.numel() == 0:
        return torch.empty(0, device=Z_u.device)

    N_u = Z_u.shape[0]
    if N_u <= 1:
        return torch.zeros(N_u, device=Z_u.device)

    device = Z_u.device

    z_u_np = _to_np(Z_u)

    # HNSW is ~30–100x faster than IndexFlatL2 for large n at negligible accuracy cost.
    index = _make_hnsw_index(z_u_np)

    k_eff = min(k + 1, N_u)
    distances, _ = index.search(z_u_np, k=k_eff)

    # HNSW distances are squared L2; exclude the self-match (index 0, distance ≈ 0).
    distances = distances[:, 1:]
    distances = np.sqrt(np.maximum(distances, 0.0))

    avg = distances.mean(axis=1)
    raw_scores = 1.0 / (avg + 1e-8)

    scores = torch.tensor(raw_scores, dtype=torch.float32, device=device)

    lo = torch.quantile(scores, q_low)
    hi = torch.quantile(scores, q_high)

    if torch.isclose(lo, hi):
        return torch.zeros_like(scores)

    scores = (scores - lo) / (hi - lo)
    return torch.clamp(scores, 0.0, 1.0)


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
    """
    Composite score from already-normalized component scores.

    Assumes all inputs are already in [0, 1].
    """

    total = wu + wd + wr
    if total <= 0:
        return torch.zeros_like(uncertainty_scores)

    wu = wu / total
    wd = wd / total
    wr = wr / total

    return (
        wu * uncertainty_scores
        + wd * diversity_scores
        + wr * density_scores
    )

