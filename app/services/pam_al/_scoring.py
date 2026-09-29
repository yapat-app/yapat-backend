"""
Acquisition scoring against a cached, all-snippets HNSW index.

The index covers every snippet in the set (see ``_ann_index_cache``), so
"nearest unlabelled neighbour" queries ask for a few extra and drop the
labelled hits. Row indices here are positions in the snippet set's embedding
matrix, which is also what faiss returns -- no separate id mapping.

Score definitions are unchanged from ``active_learning.samplers``:

* density   = 1 / mean(distance to the k nearest *unlabelled* neighbours)
* diversity = distance to the nearest *labelled* point, then reduced for the
              neighbours of each greedily-chosen farthest point

What changed is only how the neighbours are found.
"""

from __future__ import annotations

import logging
import time

import faiss
import numpy as np

from active_learning.config import HNSW_EF_SEARCH, HNSW_MIN_NL, HNSW_NEIGHBORS
from app.services.pam_al._ann_index_cache import normalize_rows

logger = logging.getLogger(__name__)

QUERY_CHUNK = 20_000
# Extra neighbours requested so that dropping labelled hits still leaves k.
# Labelled points are ~0.1% of a set but are deliberately clustered in the
# regions active learning finds interesting, so the local rate can be far
# higher than the global one; deficient rows are re-queried below.
NEIGHBOUR_MARGIN = 64
MAX_NEIGHBOUR_QUERY = 4096


def _rows_normalized(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Fetch and normalise specific rows, keeping a memmap out of RAM."""
    return normalize_rows(X[rows])


def compute_density(
    index: faiss.Index,
    X: np.ndarray,
    unlabeled_rows: np.ndarray,
    is_labeled: np.ndarray,
    k: int,
) -> np.ndarray:
    """1 / mean distance to the k nearest unlabelled neighbours, per unlabelled row."""
    n_u = unlabeled_rows.shape[0]
    if n_u == 0:
        return np.empty(0, dtype=np.float32)
    if n_u == 1:
        return np.zeros(1, dtype=np.float32)

    k_eff = min(k, n_u - 1)
    avg = np.full(n_u, np.nan, dtype=np.float32)
    started = time.perf_counter()

    for start in range(0, n_u, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_u)
        rows = unlabeled_rows[start:stop]
        queries = _rows_normalized(X, rows)
        avg[start:stop] = _mean_unlabeled_distance(index, queries, rows, is_labeled, k_eff)
        if (start // QUERY_CHUNK) % 25 == 0:
            logger.info("density: %s/%s rows (%.0fs)", stop, n_u, time.perf_counter() - started)

    raw = 1.0 / (avg + 1e-8)
    logger.info("density: %s rows in %.1fs", n_u, time.perf_counter() - started)
    return raw.astype(np.float32)


def _mean_valid_distance(
    index: faiss.Index,
    queries: np.ndarray,
    query_rows: np.ndarray,
    is_labeled: np.ndarray,
    k: int,
    k_query: int,
) -> np.ndarray:
    """Mean distance to the k nearest neighbours that are neither self nor labelled.

    Fully vectorised: invalid neighbours are pushed to +inf and each row sorted,
    so the first k columns are the k valid nearest. A row with fewer than k
    valid neighbours keeps an inf in that window and comes back non-finite,
    which is how the caller detects rows needing a deeper search.
    """
    distances, neighbours = index.search(queries, k=k_query)
    distances = np.sqrt(np.maximum(distances, 0.0))

    safe_ids = np.maximum(neighbours, 0)
    keep = (
        (neighbours >= 0)
        & (neighbours != query_rows[:, None])
        & ~is_labeled[safe_ids]
    )
    masked = np.where(keep, distances, np.inf)
    masked.sort(axis=1)
    return masked[:, :k].mean(axis=1)


def _mean_unlabeled_distance(
    index: faiss.Index,
    queries: np.ndarray,
    query_rows: np.ndarray,
    is_labeled: np.ndarray,
    k: int,
) -> np.ndarray:
    """As above, widening the search for any row that came up short."""
    k_query = min(k + 1 + NEIGHBOUR_MARGIN, index.ntotal)
    out = _mean_valid_distance(index, queries, query_rows, is_labeled, k, k_query)

    while True:
        short = ~np.isfinite(out)
        if not short.any() or k_query >= min(MAX_NEIGHBOUR_QUERY, index.ntotal):
            break
        k_query = min(k_query * 4, MAX_NEIGHBOUR_QUERY, index.ntotal)
        logger.info("density: re-querying %s rows with k=%s", int(short.sum()), k_query)
        out[short] = _mean_valid_distance(
            index, queries[short], query_rows[short], is_labeled, k, k_query
        )

    # Budget exhausted for a few rows: treat them as maximally sparse rather
    # than letting inf propagate into the score.
    stubborn = ~np.isfinite(out)
    if stubborn.any():
        logger.warning("density: %s rows had fewer than %s unlabelled neighbours", int(stubborn.sum()), k)
        out[stubborn] = 1.0
    return out


def compute_nearest_labeled(
    X: np.ndarray,
    unlabeled_rows: np.ndarray,
    labeled_rows: np.ndarray,
) -> np.ndarray:
    """Distance from each unlabelled row to the nearest labelled row.

    The labelled set is small (thousands), so this index is rebuilt every cycle
    rather than cached -- it is the part that legitimately changes when someone
    annotates.
    """
    n_u = unlabeled_rows.shape[0]
    if labeled_rows.shape[0] == 0:
        return np.full(n_u, 1.0, dtype=np.float32)

    labeled_vectors = _rows_normalized(X, labeled_rows)
    dim = labeled_vectors.shape[1]

    if labeled_rows.shape[0] >= HNSW_MIN_NL:
        index = faiss.IndexHNSWFlat(dim, int(HNSW_NEIGHBORS), faiss.METRIC_L2)
        index.hnsw.efSearch = int(HNSW_EF_SEARCH)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(labeled_vectors)

    out = np.empty(n_u, dtype=np.float32)
    for start in range(0, n_u, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_u)
        queries = _rows_normalized(X, unlabeled_rows[start:stop])
        distances, _ = index.search(queries, k=1)
        out[start:stop] = np.sqrt(np.maximum(distances[:, 0], 0.0))
    return out


def compute_diversity(
    index: faiss.Index,
    X: np.ndarray,
    unlabeled_rows: np.ndarray,
    is_labeled: np.ndarray,
    nearest_labeled: np.ndarray,
    num_centers: int,
    update_k: int,
) -> np.ndarray:
    """Greedy farthest-point selection, reducing each centre's neighbourhood.

    Same algorithm as ``_greedy_farthest_point_select``; the candidate lookup
    goes through the shared index and drops labelled hits, and positions are
    translated back into unlabelled-array offsets.
    """
    n_u = unlabeled_rows.shape[0]
    if n_u == 0:
        return np.empty(0, dtype=np.float32)

    min_dist = nearest_labeled.astype(np.float32, copy=True)
    picked = np.zeros(n_u, dtype=bool)
    # unlabeled_rows is sorted, so searchsorted maps a matrix row back to its
    # offset in the unlabelled arrays.
    centers = min(num_centers, n_u)
    k_query = min(update_k + NEIGHBOUR_MARGIN, index.ntotal)
    started = time.perf_counter()

    for _ in range(centers):
        masked = np.where(picked, -np.inf, min_dist)
        center = int(np.argmax(masked))
        if masked[center] == -np.inf:
            break
        picked[center] = True

        center_row = unlabeled_rows[center]
        query = _rows_normalized(X, np.asarray([center_row]))
        _, neighbours = index.search(query, k=k_query)
        ids = neighbours[0]
        ids = ids[(ids >= 0) & ~is_labeled[np.maximum(ids, 0)]]
        if ids.size == 0:
            continue

        # Map matrix rows back to offsets in the unlabelled arrays. Keep the
        # ids and their positions aligned through both filters, or the mask
        # below compares mismatched pairs.
        positions = np.searchsorted(unlabeled_rows, ids)
        in_range = positions < n_u
        positions, ids = positions[in_range], ids[in_range]
        offsets = positions[unlabeled_rows[positions] == ids]
        if offsets.size == 0:
            continue

        candidates = _rows_normalized(X, unlabeled_rows[offsets])
        d = np.linalg.norm(candidates - query[0], axis=1).astype(np.float32)
        min_dist[offsets] = np.minimum(min_dist[offsets], d)

    logger.info("diversity: %s centres over %s rows in %.1fs", centers, n_u, time.perf_counter() - started)
    return min_dist


def compute_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Mean binary entropy across labels -- same formula as ALQueryScorer.uncertainty."""
    p = np.asarray(probs, dtype=np.float64)
    entropy = -(p * np.log(p + 1e-12) + (1.0 - p) * np.log(1.0 - p + 1e-12)).mean(axis=1)
    return entropy.astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise to zero mean, unit variance.

    Matches active_learning.samplers.zscore, including its intent for the
    degenerate cases -- but guards them properly. torch's ``.std()`` is
    unbiased (ddof=1) and returns NaN for a single element, which slips past
    that implementation's ``std < 1e-8`` check and poisons the result; here a
    non-finite or near-zero std returns zeros as the docstring there promises.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    std = x.std(ddof=1) if x.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(x)
    return ((x - x.mean()) / std).astype(np.float32)


def compute_composite(
    uncertainty_z: np.ndarray,
    diversity_z: np.ndarray,
    density_z: np.ndarray,
    wu: float,
    wd: float,
    wr: float,
) -> np.ndarray:
    """Weighted sum of z-scored components, weights renormalised to sum to 1."""
    total = wu + wd + wr
    if total <= 0:
        return np.zeros_like(uncertainty_z)
    return (
        (wu / total) * uncertainty_z
        + (wd / total) * diversity_z
        + (wr / total) * density_z
    ).astype(np.float32)
