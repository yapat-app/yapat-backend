import logging
import torch
import numpy as np
import torch.nn.functional as F
import faiss


from app.schemas.pam_active_learning import ALSingleSampleScore
from active_learning.config import DIVERSITY_HNSW_MIN_NL, DIVERSITY_POOL_SIZE

logger = logging.getLogger(__name__)

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

def _nearest_labeled_distances(
    z_u_np: np.ndarray,
    z_l_np: np.ndarray,
    hnsw_min_nl: int,
) -> np.ndarray:
    """
    1-NN distance from each row of z_u_np to the nearest row of z_l_np.

    Below hnsw_min_nl labelled points, brute-force (Flat) search: building an
    HNSW graph has more per-point overhead than a flat scan is worth at small
    Nl, and O(N * Nl) is cheap when Nl is small regardless.

    At or above hnsw_min_nl, uses an approximate HNSW index instead: exact
    search here is O(N * Nl) and was measured as the dominant per-cycle AL
    scoring cost (see paper Section 2.6 / Fig. 5d), dwarfing density's
    O(N log N) HNSW-based search. HNSW brings this to ~O(N log Nl), same
    complexity class as density(), at the cost of being approximate (may
    occasionally return the second-nearest labelled point instead of the
    true nearest). Since this score only feeds a ranking/composite acquisition
    score -- not an exact measurement -- that's an acceptable trade, the same
    one already made for density().
    """
    dim = z_l_np.shape[1]
    n_l = z_l_np.shape[0]

    if n_l < hnsw_min_nl:
        index = faiss.IndexFlatL2(dim)
        index.add(z_l_np)
    else:
        # _make_hnsw_index() already adds the vectors internally.
        index = _make_hnsw_index(z_l_np)

    distances, _ = index.search(z_u_np, k=1)
    return np.sqrt(np.maximum(distances[:, 0], 0.0))


def diversity(
    Z_u: torch.Tensor,
    Z_l: torch.Tensor,
    hnsw_min_nl: int | None = None,
    pool_size: int | None = None # Limits how many candidates actually go through expensive greedy coreset selection"
) -> torch.Tensor:
    """
    Diversity normalized to [0, 1], with intra-batch redundancy
    correction via greedy farthest-point (core-set) traversal.
    Simulates greedy farthest-point picks

    hnsw_min_nl: labelled-set size threshold above which the nearest-labelled
    search switches from exact (Flat) to approximate (HNSW). Defaults to
    DIVERSITY_HNSW_MIN_NL from active_learning/config.yaml.
    """

    stats = diversity_coreset_stats(Z_u, Z_l)
    logger.info("Diversity stats: %s", stats)

    if Z_u.numel() == 0:
        return torch.empty(0, device=Z_u.device)

    device = Z_u.device
    n_u = Z_u.shape[0]
    z_u_np = _to_np(Z_u)

    if Z_l.numel() == 0:
        # nothing labeled yet
        #return torch.zeros(Z_u.shape[0], device=Z_u.device)
        nearest_ref_dist = np.full(n_u, 1.0, dtype="float32")
    else:
        z_l_np = _to_np(Z_l)
        threshold = hnsw_min_nl if hnsw_min_nl is not None else DIVERSITY_HNSW_MIN_NL
        nearest_ref_dist = _nearest_labeled_distances(z_u_np, z_l_np, threshold)

    scores = nearest_ref_dist.copy()

    pool_n = min(pool_size, n_u) if pool_size is not None else min(DIVERSITY_POOL_SIZE, n_u)
    pool_idx = np.argsort(-nearest_ref_dist)[:pool_n]
    pool_dist = nearest_ref_dist[pool_idx].copy()
    remaining = np.ones(pool_n, dtype=bool)

    for _ in range(pool_n):
        masked = np.where(remaining, pool_dist, -np.inf)
        local_pick = int(np.argmax(masked))
        if masked[local_pick] == -np.inf:
            break

        global_pick = pool_idx[local_pick]
        remaining[local_pick] = False
        scores[global_pick] = pool_dist[local_pick]  # value at time of pick

        # fold the pick into the reference set; redundant neighbors in the
        # pool immediately lose score on the next iteration
        picked_vec = z_u_np[global_pick]
        d_to_pick = np.linalg.norm(z_u_np[pool_idx] - picked_vec, axis=1)
        pool_dist = np.minimum(pool_dist, d_to_pick)

    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    return torch.clamp(scores_t, 0.0, 1.0)


def diversity_approx_error(Z_u: torch.Tensor, Z_l: torch.Tensor) -> dict:
    """
    Benchmarking/diagnostic helper: quantify how much the HNSW approximation
    in diversity() actually costs in accuracy, by comparing it against exact
    (Flat) search on the same inputs.

    Not used in the production scoring path -- computing both exact and
    approximate results defeats the point of the optimization. Intended for
    the benchmark suite (see docs/benchmark-handoff.md) when re-measuring
    Fig. 5d / Section 2.6 with the HNSW path, to report an actual error rate
    alongside the timing improvement rather than assuming the approximation
    is harmless.

    Returns
    -------
    dict with:
        recall_at_1: fraction of Z_u rows where HNSW's nearest labelled
            neighbour is the *same point* Flat found (exact match rate).
        mean_abs_distance_error / max_abs_distance_error: |approx - exact|
            over the (Euclidean, post-L2-normalization) 1-NN distances,
            for rows where HNSW picked a different (necessarily farther,
            since Flat is exact) neighbour.
        n: number of Z_u rows compared.
    """
    if Z_u.numel() == 0 or Z_l.numel() == 0:
        return {"recall_at_1": None, "mean_abs_distance_error": None, "max_abs_distance_error": None, "n": 0}

    z_u_np = _to_np(Z_u)
    z_l_np = _to_np(Z_l)
    dim = z_l_np.shape[1]

    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(z_l_np)
    exact_distances, exact_indices = exact_index.search(z_u_np, k=1)
    exact_distances = np.sqrt(np.maximum(exact_distances[:, 0], 0.0))

    approx_index = _make_hnsw_index(z_l_np)
    approx_distances, approx_indices = approx_index.search(z_u_np, k=1)
    approx_distances = np.sqrt(np.maximum(approx_distances[:, 0], 0.0))

    matches = exact_indices[:, 0] == approx_indices[:, 0]
    abs_error = np.abs(approx_distances - exact_distances)

    return {
        "recall_at_1": float(matches.mean()),
        "mean_abs_distance_error": float(abs_error.mean()),
        "max_abs_distance_error": float(abs_error.max()) if abs_error.size else 0.0,
        "n": int(z_u_np.shape[0]),
    }

def diversity_coreset_stats(Z_u: torch.Tensor, Z_l: torch.Tensor) -> dict:
    """
    Diagnostic only -- reports the actual empirical range of nearest-labeled
    distances on real data, to determine the correct normalization divisor
    (or whether quantile normalization is needed instead of a fixed constant).
    Not used in the production scoring path.
    """
    if Z_u.numel() == 0 or Z_l.numel() == 0:
        return {"min": None, "max": None, "mean": None, "p50": None, "p95": None, "p99": None, "n": 0}

    z_u_np = _to_np(Z_u)
    z_l_np = _to_np(Z_l)
    dists = _nearest_labeled_distances(z_u_np, z_l_np, DIVERSITY_HNSW_MIN_NL)

    return {
        "min": float(dists.min()),
        "max": float(dists.max()),
        "mean": float(dists.mean()),
        "p50": float(np.percentile(dists, 50)),
        "p95": float(np.percentile(dists, 95)),
        "p99": float(np.percentile(dists, 99)),
        "n": int(dists.shape[0]),
    }


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

