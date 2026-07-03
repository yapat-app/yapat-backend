"""
Shared approximate-nearest-neighbour (ANN) backend for DR and AL.

Both pipelines need k-NN search over the same frozen BirdNET embedding space,
but historically did it three different ways: DR used pynndescent
(build_knn_graph), AL's diversity() used ad hoc faiss.IndexFlatL2 /
IndexHNSWFlat, and AL's density() used a third ad hoc HNSW setup. This module
gives all of them one implementation and one place to add GPU support later,
instead of three.

Two search shapes are supported, both expressed as "query points against an
index built over index_points":
  - self-search (index_points is query itself): DR's build_knn_graph,
    AL's density()
  - asymmetric search (index_points is a different, usually smaller, set):
    AL's diversity() -- unlabeled points queried against labeled points

Distance convention: FAISS's L2 metric returns *squared* Euclidean distance.
Every function here returns true Euclidean distance (post-sqrt), matching the
convention already established in active_learning/samplers.py.

GPU: faiss-cpu is the only build currently installed (see requirements.txt),
so the GPU branch below is scaffolding, not yet active -- prefer_gpu=True is
safe to pass everywhere; it silently falls back to CPU when no GPU-capable
faiss build is present. This keeps the "GPU is additive, never required"
principle: calling code doesn't need to know which build is installed.
"""
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Same idea as active_learning.config.DIVERSITY_HNSW_MIN_NL (index size above
# which approximate HNSW search is used instead of exact Flat search), but
# not yet unified with it -- DR and AL are tuned independently until we have
# benchmark numbers to justify sharing one constant. This is DR's default;
# callers can override via the hnsw_min_n parameter.
DEFAULT_HNSW_MIN_N = 4096

_GPU_RESOURCES = None
_GPU_CHECKED = False


def _gpu_available() -> bool:
    """True if a GPU-capable faiss build is installed AND a GPU is visible.

    Returns False (never raises) on faiss-cpu, where StandardGpuResources
    doesn't exist. This is the single place that will need to change when
    faiss-gpu / cuVS adoption actually happens.
    """
    global _GPU_RESOURCES, _GPU_CHECKED
    if _GPU_CHECKED:
        return _GPU_RESOURCES is not None
    _GPU_CHECKED = True
    try:
        if not hasattr(faiss, "StandardGpuResources"):
            return False
        if faiss.get_num_gpus() < 1:
            return False
        _GPU_RESOURCES = faiss.StandardGpuResources()
        return True
    except Exception:
        logger.debug("ann_backend: GPU probe failed, falling back to CPU", exc_info=True)
        _GPU_RESOURCES = None
        return False


def build_index(
    index_points: np.ndarray,
    exact: bool | None = None,
    hnsw_min_n: int = DEFAULT_HNSW_MIN_N,
    hnsw_m: int = 32,
    ef_search: int = 64,
    prefer_gpu: bool = True,
) -> faiss.Index:
    """Build a FAISS index over index_points.

    exact=None (default): auto-select -- Flat below hnsw_min_n points, HNSW
    at/above it. exact=True/False forces the choice regardless of size.
    """
    index_points = np.ascontiguousarray(index_points, dtype="float32")
    dim = index_points.shape[1]
    n = index_points.shape[0]

    use_exact = exact if exact is not None else (n < hnsw_min_n)

    if use_exact:
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
        index.hnsw.efSearch = ef_search

    if prefer_gpu and _gpu_available():
        try:
            index = faiss.index_cpu_to_gpu(_GPU_RESOURCES, 0, index)
        except Exception:
            logger.debug("ann_backend: index_cpu_to_gpu failed, staying on CPU", exc_info=True)

    index.add(index_points)
    return index


def search(index: faiss.Index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search query against a prebuilt index.

    Returns (distances, indices) -- FAISS's native argument order -- with
    distances as true Euclidean (sqrt already applied), matching the
    convention used throughout active_learning/samplers.py. k is clamped to
    the index size so callers don't need to know ntotal in advance.
    """
    query = np.ascontiguousarray(query, dtype="float32")
    k_eff = min(k, index.ntotal)
    distances, indices = index.search(query, k=k_eff)
    distances = np.sqrt(np.maximum(distances, 0.0))
    return distances, indices


def nearest_neighbors(
    query: np.ndarray,
    index_points: np.ndarray,
    k: int,
    exact: bool | None = None,
    hnsw_min_n: int = DEFAULT_HNSW_MIN_N,
    hnsw_m: int = 32,
    ef_search: int = 64,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, faiss.Index]:
    """Convenience wrapper: build an index over index_points, search query
    against it, and return (indices, distances, index) -- note this order
    matches DR's build_knn_graph contract (indices, distances, ...), not
    search()'s FAISS-native (distances, indices) order. Passing query=
    index_points gives self-search including the point itself as its own
    nearest neighbour at distance 0, matching pynndescent's convention
    (required by UMAP's precomputed_knn / openTSNE's PerplexityBasedNN).

    The returned index is included so callers needing a persistent, queryable
    index (as UMAP's precomputed_knn 3-tuple expects) can keep it; it is not
    currently used for out-of-sample transform anywhere in this codebase.
    """
    index = build_index(
        index_points,
        exact=exact,
        hnsw_min_n=hnsw_min_n,
        hnsw_m=hnsw_m,
        ef_search=ef_search,
        prefer_gpu=prefer_gpu,
    )
    distances, indices = search(index, query, k)
    return indices, distances, index
