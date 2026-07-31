import torch
import numpy as np
import torch.nn.functional as F
import faiss


from active_learning.config import (HNSW_NEIGHBORS, HNSW_EF_SEARCH, HNSW_MIN_NL, HNSW_MIN_NU,
                                    DEFAULT_DENSITY_K, DIVERSITY_NUM_CENTERS, DIVERSITY_UPDATE_K)



def _to_np(x: torch.Tensor) -> np.ndarray:
    x = F.normalize(x, p=2, dim=1) # normalizing the embeddings using l2- norm
    return x.detach().cpu().numpy().astype("float32")

def _make_hnsw_index(
    vectors: np.ndarray,
    M: int = HNSW_NEIGHBORS, # How many neighbor connections each node keeps
    ef_search: int = HNSW_EF_SEARCH, # How many candidate nodes are explored during search
) -> faiss.IndexHNSWFlat:
    dim = vectors.shape[1]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efSearch = ef_search
    index.add(vectors)

    return index

def zscore(x: torch.Tensor) -> torch.Tensor:
    """
    Standardize x to zero mean, unit variance.

    Edge cases:
    - Empty tensor (numel == 0): returned unchanged.
    - Near-zero std (all values identical or near-identical): returns
      zeros rather than NaN/inf.
    """
    if x.numel() == 0:
        return x
    std = x.std()
    if std < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / std



def _nearest_labeled_distances(
    z_u_np: np.ndarray,
    z_l_np: np.ndarray,
    hnsw_min_nl: int,
    hnsw_index: faiss.IndexHNSWFlat | None = None,
) -> np.ndarray:
    """
    1-NN distance from each row of z_u_np to the nearest row of z_l_np.

    hnsw_index: pre-built index to reuse (e.g. from ALQueryScorer's cache).
    If None, builds one internally using the hnsw_min_nl threshold -- kept
    for standalone callers.
    """
    n_l = z_l_np.shape[0]

    if hnsw_index is not None:
        index = hnsw_index
    elif n_l < hnsw_min_nl:
        index = faiss.IndexFlatL2(z_l_np.shape[1])
        index.add(z_l_np)
    else:
        index = _make_hnsw_index(z_l_np)

    distances, _ = index.search(z_u_np, k=1)
    return np.sqrt(np.maximum(distances[:, 0], 0.0))



def diversity_approx_error(
    Z_u: torch.Tensor,
    Z_l: torch.Tensor,
    z_u_np: np.ndarray | None = None,
    z_l_np: np.ndarray | None = None,
) -> dict:
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

    z_u_np/z_l_np: pre-converted arrays to skip re-running _to_np, e.g.
    scorer.z_u_np / scorer.z_l_np from an existing ALQueryScorer instance
    for the same Z_u/Z_l. Recomputed from Z_u/Z_l if not supplied. Note
    this function always builds its own exact and approximate indices
    regardless -- it deliberately bypasses any cached hnsw_l_index, since
    the whole point is comparing both paths on the same data, not reusing
    whichever one the class happened to pick via its threshold.

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

    z_u_np = z_u_np if z_u_np is not None else _to_np(Z_u)
    z_l_np = z_l_np if z_l_np is not None else _to_np(Z_l)
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
    Composite score from component acquisition scores.

    Inputs are expected to be z-scored (mean 0, std 1) before calling this
    function. The output is therefore unbounded and should be interpreted
    only as a ranking signal, not as a probability.
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

def _greedy_farthest_point_select(
    z_u_np: np.ndarray,
    nearest_ref_dist: np.ndarray,
    k: int,
    hnsw_index: faiss.IndexHNSWFlat | None,
    update_k: int,
) -> np.ndarray:

    n_u = z_u_np.shape[0]
    k = min(k, n_u)

    min_dist = nearest_ref_dist.copy()
    picked = np.zeros(n_u, dtype=bool)

    for _ in range(k):
        masked = np.where(picked, -np.inf, min_dist)
        center = int(np.argmax(masked)) # will give highest value index
        if masked[center] == -np.inf: #if every point has been picked
            break
        picked[center] = True # mark True for the picked point

        if hnsw_index is not None:
            k_eff = min(update_k, n_u)
            _, neighbor_ids = hnsw_index.search(z_u_np[center : center + 1], k=k_eff)
            candidates = neighbor_ids[0]
            candidates = candidates[candidates >= 0]  # faiss pads with -1 if k_eff > n_u
        else:
            candidates = np.arange(n_u)

        d = np.linalg.norm(z_u_np[candidates] - z_u_np[center], axis=1)
        min_dist[candidates] = np.minimum(min_dist[candidates], d)

    return min_dist

class ALQueryScorer:
    """
            Computes uncertainty/diversity/density acquisition scores for one AL
            scoring cycle, caching the embedding conversions (L2 normalizations) and the HNSW index
            shared across diversity() and density() so a single cycle doesn't
            rebuild them twice.
            """

    def __init__(
            self,
            Z_u: torch.Tensor,
            Z_l: torch.Tensor,
            hnsw_min_nl: int | None = None,
            hnsw_min_nu: int | None = None,

    ):
        self.Z_u = Z_u
        self.Z_l = Z_l
        self.device = Z_u.device
        self.n_u = Z_u.shape[0]
        self.n_l = Z_l.shape[0] if Z_l.numel() else 0

        self.hnsw_min_nl = hnsw_min_nl if hnsw_min_nl is not None else HNSW_MIN_NL
        self.hnsw_min_nu = hnsw_min_nu if hnsw_min_nu is not None else HNSW_MIN_NU

        self._z_u_np: np.ndarray | None = None
        self._z_l_np: np.ndarray | None = None
        self._hnsw_l_index: faiss.IndexHNSWFlat | None = None
        self._hnsw_l_index_built = False
        self._hnsw_u_index: faiss.IndexHNSWFlat | None = None
        self._hnsw_u_index_built = False
        self._nearest_ref_dist: np.ndarray | None = None

        self._diversity_raw: torch.Tensor | None = None
        self._density_raw: torch.Tensor | None = None

    @property
    def z_u_np(self) -> np.ndarray:
        if self._z_u_np is None:
            self._z_u_np = _to_np(self.Z_u)
        return self._z_u_np

    @property
    def z_l_np(self) -> np.ndarray | None:
        if self.n_l == 0:
            return None
        if self._z_l_np is None:
            self._z_l_np = _to_np(self.Z_l)
        return self._z_l_np

    @property
    def hnsw_l_index(self) -> faiss.IndexHNSWFlat | None:
        """
        Cached HNSW index over Z_l (labeled points), used by
        nearest_ref_dist's underlying search. None if n_l < hnsw_min_nl --
        caller falls back to exact Flat search in that case, same threshold
        logic as hnsw_u_index.
        """
        if self.n_l < self.hnsw_min_nl:
            return None
        if not self._hnsw_l_index_built:
            self._hnsw_l_index = _make_hnsw_index(self.z_l_np)
            self._hnsw_l_index_built = True
        return self._hnsw_l_index

    @property
    def hnsw_u_index(self) -> faiss.IndexHNSWFlat | None:
        """Shared by density() and diversity()'s k-center update step."""
        if self.n_u < self.hnsw_min_nu:
            return None
        if not self._hnsw_u_index_built:
            self._hnsw_u_index = _make_hnsw_index(self.z_u_np)
            self._hnsw_u_index_built = True
        return self._hnsw_u_index

    @property
    def nearest_ref_dist(self) -> np.ndarray:
        """almost identical to nearest_labeled_distances() but caches the result for repeated calls in a single AL cycle."""
        if self._nearest_ref_dist is None:
            if self.n_l == 0:
                self._nearest_ref_dist = np.full(self.n_u, 1.0, dtype="float32")
            else:
                index = self.hnsw_l_index
                if index is None:
                    index = faiss.IndexFlatL2(self.z_l_np.shape[1])
                    index.add(self.z_l_np)
                distances, _ = index.search(self.z_u_np, k=1)
                self._nearest_ref_dist = np.sqrt(np.maximum(distances[:, 0], 0.0))
        return self._nearest_ref_dist

    def diversity(self, zscored: bool = True, k: int | None = None, update_k: int | None = None) -> torch.Tensor:
        if self.n_u == 0:
            return torch.empty(0, device=self.device)

        if self._diversity_raw is None:
            if self.n_l == 0:
                # No labeled data yet -- nothing to be "diverse from," so every
                # unlabeled point is equally uninformative on this axis. Skip
                self._diversity_raw = torch.full((self.n_u,), 1.0, device=self.device)
            else:
                k_val = k if k is not None else DIVERSITY_NUM_CENTERS
                uk = update_k if update_k is not None else DIVERSITY_UPDATE_K

                # nearest_ref_dist is a starting score before any redundancy correction.
                corrected = _greedy_farthest_point_select(
                    self.z_u_np, self.nearest_ref_dist, k_val, self.hnsw_u_index, uk
                )
                self._diversity_raw = torch.tensor(corrected, dtype=torch.float32, device=self.device)

        return zscore(self._diversity_raw) if zscored else self._diversity_raw

    def density(self, k: int | None = None, zscored: bool = True, q_low: float = 0.05, q_high: float = 0.95) -> torch.Tensor:

        if self.n_u == 0:
            return torch.empty(0, device=self.device)
        if self.n_u <= 1:
            return torch.zeros(self.n_u, device=self.device)

        if self._density_raw is None:
            k_val = k if k is not None else DEFAULT_DENSITY_K

            if self.hnsw_u_index is not None:
                index = self.hnsw_u_index
            else:
                index = faiss.IndexFlatL2(self.z_u_np.shape[1])
                index.add(self.z_u_np)

            k_eff = min(k_val + 1, self.n_u)
            distances, _ = index.search(self.z_u_np, k=k_eff)

            # exclude self-match (index 0, distance ≈ 0)
            distances = distances[:, 1:]
            distances = np.sqrt(np.maximum(distances, 0.0))

            avg = distances.mean(axis=1)
            raw_scores = 1.0 / (avg + 1e-8)

            self._density_raw = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)

        if not zscored:
            return self._density_raw

        # raw_scores = 1/avg_dist is right-skewed by construction -- a few
        # near-duplicate points can produce extreme outlier values that
        # dominate the mean/std z-scoring is based on. Clip to the
        # [q_low, q_high] percentile range before z-scoring so those outliers
        # don't compress everyone else's score toward the mean.
        #lo = torch.quantile(self._density_raw, q_low)
        #hi = torch.quantile(self._density_raw, q_high)
        #clipped = torch.clamp(self._density_raw, lo, hi)
        #return zscore(clipped)
        return zscore(self._density_raw)

    def uncertainty(self, P: torch.Tensor, zscored: bool = True) -> torch.Tensor:

        if P.numel() == 0:
            return torch.empty(0, device=P.device)

        entropy = -(
                P * torch.log(P + 1e-12)
                + (1 - P) * torch.log(1 - P + 1e-12)
        ).mean(dim=1) # [0, log(2)] ≈ [0, 0.693]

        return zscore(entropy) if zscored else entropy







