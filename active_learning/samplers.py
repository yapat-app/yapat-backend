import logging
import torch
import numpy as np
import torch.nn.functional as F
import faiss


from app.schemas.pam_active_learning import ALSingleSampleScore
from active_learning.config import (DIVERSITY_HNSW_MIN_NL, DIVERSITY_HNSW_MIN_NU, DIVERSITY_UPDATE_K,
                                    DIVERSITY_NUM_CENTERS,
                                    DENSITY_K)

logger = logging.getLogger(__name__)



def _to_np(x: torch.Tensor) -> np.ndarray:
    x = F.normalize(x, p=2, dim=1) # normalizing the embeddings using l2- norm
    return x.detach().cpu().numpy().astype("float32")


def _make_hnsw_index(
    vectors: np.ndarray,
    M: int = 64, # How many neighbor connections each node keeps
    ef_search: int = 128, # How many candidate nodes are explored during search
) -> faiss.IndexHNSWFlat:
    dim = vectors.shape[1]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efSearch = ef_search
    index.add(vectors)

    return index

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
    true nearest).
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

def _greedy_farthest_point_select(
    z_u_np: np.ndarray,
    nearest_ref_dist: np.ndarray,
    k: int,
    hnsw_index: faiss.IndexHNSWFlat | None,
    update_k: int,
) -> np.ndarray:
    """
    Select k batch centers via greedy farthest-point (core-set) traversal,
    returning a min-distance-to-{labeled ∪ centers} score for EVERY point in
    z_u_np -- not just the k selected. Replaces the old pool_size-capped
    greedy loop: every unlabeled point gets a genuine correction
    opportunity rather than only the top pool_size.

    hnsw_index: pre-built HNSW index over z_u_np if n_u >= hnsw_min_nu,
    else None.
      - None -> exact fallback: each pick sweeps all N candidates and
        updates any whose distance improves. O(N) per pick, correct.
      - given -> each pick only refreshes its update_k nearest neighbors
        via the index. O(log N) per pick. Candidates just outside that
        neighbor list keep a stale min_dist for that pick -- a boundary
        effect localized to areas near picks, not a coverage gap (every
        point always starts with a valid nearest_ref_dist and is never
        left unscored). Size update_k using diversity_k_center_stats
        before relying on this in production.
    """
    n_u = z_u_np.shape[0]
    k = min(k, n_u)

    min_dist = nearest_ref_dist.copy()
    picked = np.zeros(n_u, dtype=bool)

    for _ in range(k):
        masked = np.where(picked, -np.inf, min_dist)
        center = int(np.argmax(masked))
        if masked[center] == -np.inf:
            break
        picked[center] = True

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




class ALScorer:
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

        self.hnsw_min_nl = hnsw_min_nl if hnsw_min_nl is not None else DIVERSITY_HNSW_MIN_NL
        self.hnsw_min_nu = hnsw_min_nu if hnsw_min_nu is not None else DIVERSITY_HNSW_MIN_NU

        self._z_u_np: np.ndarray | None = None
        self._z_l_np: np.ndarray | None = None
        self._hnsw_u_index: faiss.IndexHNSWFlat | None = None
        self._hnsw_u_index_built = False
        self._nearest_ref_dist: np.ndarray | None = None

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
        """Distance from each Z_u point to its nearest Z_l point."""
        if self._nearest_ref_dist is None:
            if self.n_l == 0:
                self._nearest_ref_dist = np.full(self.n_u, 1.0, dtype="float32")
            else:
                self._nearest_ref_dist = _nearest_labeled_distances(
                    self.z_u_np, self.z_l_np, self.hnsw_min_nl
                )
        return self._nearest_ref_dist


    def diversity(
            self,
            k: int | None = None,
            update_k: int | None = None,
    ) -> torch.Tensor:
        """
        Diversity normalized to [0, 1] via greedy farthest point selection.

        k: number of greedy farthest-point picks (batch centers) -- the outer
        loop count. The function runs its pick-then-update cycle k times
        total, so k points end up marked as "picked." Defaults to
        DIVERSITY_NUM_CENTERS from config.

        update_k: how many neighbors get their distance score corrected after
        EACH pick -- an inner, per-pick count, not related to k except that
        it's queried once per iteration of the k-sized outer loop. If
        hnsw_u_index is available, only a picked point's update_k nearest
        neighbors get updated (points just outside that neighbor list keep a
        stale score for that pick); without an index, every remaining point
        is checked exactly, so update_k has no effect. Defaults to
        DIVERSITY_UPDATE_K from config.

        k and update_k are independent: k controls how many centers get
        selected, update_k controls how far each individual pick's
        correction reaches. E.g. k=5, update_k=1000 does few picks with a
        wide correction radius each; k=500, update_k=10 does many picks with
        a narrow correction radius each.
        """
        if self.n_u == 0:
            return torch.empty(0, device=self.device)

        k = k if k is not None else DIVERSITY_NUM_CENTERS
        uk = update_k if update_k is not None else DIVERSITY_UPDATE_K

        scores = _greedy_farthest_point_select(self.z_u_np, self.nearest_ref_dist, k, self.hnsw_u_index, uk)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return torch.clamp(scores_t, 0.0, 1.0)

    def density(
            self,
            k: int | None = None, # k must be comfortably smaller than ef_search of HNSW, not just smaller.
            q_low: float = 0.05,
            q_high: float = 0.95,
    ) -> torch.Tensor:
        """
        Density / representativeness normalized to [0, 1].

        Raw density: rho(i) = 1 / avg distance to k nearest unlabeled
        neighbors, then quantile-normalized. Reuses hnsw_u_index instead of
        building its own -- see class docstring for the resulting exact vs.
        HNSW threshold now matching diversity()'s.
        k: number of nearest unlabeled neighbors to average over. Defaults
        to DENSITY_K from config
        """
        if self.n_u == 0:
            return torch.empty(0, device=self.device)
        if self.n_u <= 1:
            return torch.zeros(self.n_u, device=self.device)

        k = k if k is not None else DENSITY_K

        if self.hnsw_u_index is not None:
            index = self.hnsw_u_index
        else:
            index = faiss.IndexFlatL2(self.z_u_np.shape[1])
            index.add(self.z_u_np)

        k_eff = min(k + 1, self.n_u)
        distances, _ = index.search(self.z_u_np, k=k_eff)

        # exclude self-match (index 0, distance ≈ 0)
        distances = distances[:, 1:]
        distances = np.sqrt(np.maximum(distances, 0.0))

        avg = distances.mean(axis=1)
        raw_scores = 1.0 / (avg + 1e-8)

        scores = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)

        lo = torch.quantile(scores, q_low)
        hi = torch.quantile(scores, q_high)
        if torch.isclose(lo, hi):
            return torch.zeros_like(scores)

        scores = (scores - lo) / (hi - lo)
        return torch.clamp(scores, 0.0, 1.0)

    @staticmethod
    def uncertainty(P: torch.Tensor) -> torch.Tensor:
        """
        Multi-label uncertainty normalized to [0, 1].

        Raw binary entropy has max log(2) at p=0.5. Dividing by log(2)
        gives: 0 = confident, 1 = maximally uncertain.

        Doesn't touch Z_u/Z_l, so it's stateless -- kept on the class only
        for a single call surface alongside diversity()/density().
        """
        if P.numel() == 0:
            return torch.empty(0, device=P.device)

        entropy = -(
                P * torch.log(P + 1e-12)
                + (1 - P) * torch.log(1 - P + 1e-12)
        ).mean(dim=1)

        return torch.clamp(entropy / np.log(2), 0.0, 1.0)


def diversity_k_center_stats(
        Z_u: torch.Tensor,
        Z_l: torch.Tensor,
        update_k_candidates: list[int],
        k: int | None = None,
) -> dict:
    """
    Diagnostic only -- NOT on the production scoring path. For a range of
    candidate update_k values, measures how many Z_u points the HNSW-limited
    k-center update leaves stale relative to the exact O(N) sweep, so
    DIVERSITY_UPDATE_K can be chosen empirically instead of guessed.

    k: number of greedy picks to simulate; defaults to DIVERSITY_NUM_CENTERS,
    matching diversity()'s own default so this diagnostic reflects production.

    Returns a dict keyed by update_k with stale_fraction / mean_abs_error /
    max_abs_error against the exact-sweep scores.
    """
    z_u_np = _to_np(Z_u)
    n_u = z_u_np.shape[0]

    if Z_l.numel() == 0:
        nearest_ref_dist = np.full(n_u, 1.0, dtype="float32")
    else:
        z_l_np = _to_np(Z_l)
        nearest_ref_dist = _nearest_labeled_distances(z_u_np, z_l_np, DIVERSITY_HNSW_MIN_NL)

    k = k if k is not None else DIVERSITY_NUM_CENTERS

    exact_scores = _greedy_farthest_point_select(z_u_np, nearest_ref_dist, k, hnsw_index=None, update_k=n_u)

    results = {}
    hnsw_index = _make_hnsw_index(z_u_np)
    for uk in update_k_candidates:
        approx_scores = _greedy_farthest_point_select(z_u_np, nearest_ref_dist, k, hnsw_index, uk)
        diff = np.abs(approx_scores - exact_scores)
        stale = diff > 1e-6
        results[uk] = {
            "stale_fraction": float(stale.mean()),
            "mean_abs_error": float(diff[stale].mean()) if stale.any() else 0.0,
            "max_abs_error": float(diff.max()) if diff.size else 0.0,
        }

    return results


def diversity_approx_error(Z_u: torch.Tensor, Z_l: torch.Tensor) -> dict:
    """
    Benchmarking/diagnostic helper: quantify how much the HNSW approximation
    in _nearest_labeled_distances actually costs in accuracy, by comparing
    it against exact (Flat) search on the same inputs.

    Not used in the production scoring path. See docs/benchmark-handoff.md.
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
    Diagnostic only -- reports the empirical range of nearest-labeled
    distances on real data. Not used in the production scoring path.
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


