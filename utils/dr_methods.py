import logging

import numpy as np
import scipy.sparse as sp
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import sort_graph_by_row_values

logger = logging.getLogger(__name__)

_PRE_REDUCE_DIMS = 50


def pre_reduce_pca(embeddings: np.ndarray, max_vis_dims: int = 2) -> np.ndarray:
    """Single PCA covering both visualisation slices and pre-reduction for kNN methods.

    Computes PCA to max(max_vis_dims, _PRE_REDUCE_DIMS) so the caller can:
      - slice [:, :2] / [:, :3] for PCA visualisation coordinates
      - pass the full result as already-reduced input to UMAP / t-SNE / Isomap

    Logs explained variance and warns if it falls below 95%.
    """
    n_components = min(
        max(max_vis_dims, _PRE_REDUCE_DIMS),
        embeddings.shape[0] - 1,
        embeddings.shape[1],
    )
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())
    msg = "pre_reduce_pca: %d → %d dims, %.1f%% variance explained"
    args = (embeddings.shape[1], n_components, explained * 100)
    if explained < 0.95:
        logger.warning(msg, *args)
    else:
        logger.info(msg, *args)
    return result


def build_knn_graph(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Compute kNN graph via the shared FAISS-backed ANN abstraction
    (utils/ann_backend.py), replacing the previous pynndescent backend.

    Self-search (embeddings queried against themselves), so each row
    includes itself as neighbour 0 at distance 0 -- same convention
    pynndescent used, required by UMAP's precomputed_knn and openTSNE's
    PerplexityBasedNN. Auto-selects exact (Flat) vs approximate (HNSW)
    search based on embeddings.shape[0] via ann_backend.DEFAULT_HNSW_MIN_N;
    below that size (the common case for this codebase's dataset sizes
    today) this is exact, i.e. equivalent to the old pynndescent output for
    correctness purposes.

    Returns (indices, distances, index) where indices and distances are each
    of shape (n, n_neighbors). The FAISS index is included as the third
    element so UMAP can receive the full 3-tuple via precomputed_knn (it is
    not currently used for out-of-sample transform).
    """
    from utils.ann_backend import nearest_neighbors
    embeddings = np.asarray(embeddings, dtype="float32")
    k = min(n_neighbors, embeddings.shape[0])
    return nearest_neighbors(embeddings, embeddings, k=k)


class _PrecomputedKNNIndex:
    """Thin wrapper so openTSNE can consume a precomputed kNN graph."""
    def __init__(self, indices: np.ndarray, distances: np.ndarray) -> None:
        self._indices = indices
        self._distances = distances
        self.k = indices.shape[1]

    def build(self):
        return self._indices, self._distances

    def query(self, data: np.ndarray, k: int):
        return self._indices[:, :k], self._distances[:, :k]


def run_dr_isomap(
    embeddings,
    dimensions: int,
    n_neighbors: Optional[int] = 30,
    precomputed_knn: Optional[tuple] = None,
):
    if precomputed_knn is not None:
        indices, distances = precomputed_knn[:2]
        n = len(indices)
        rows = np.repeat(np.arange(n), indices.shape[1])
        dist_matrix = sp.csr_matrix(
            (distances.ravel(), (rows, indices.ravel())), shape=(n, n)
        )
        dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)
        # sklearn internally requests n_neighbors+1 to exclude the self-match,
        # so we compensate by passing n_neighbors-1 here.
        reducer = Isomap(n_neighbors=n_neighbors - 1, n_components=dimensions, metric="precomputed")
        return reducer.fit_transform(dist_matrix)
    reducer = Isomap(n_neighbors=n_neighbors, n_components=dimensions)
    return reducer.fit_transform(embeddings)


def run_dr_tsne(
    embeddings,
    dimensions: int,
    perplexity: Optional[int] = 30,
    precomputed_knn: Optional[tuple] = None,
):
    from openTSNE import TSNE as OpenTSNE
    from openTSNE.affinity import PerplexityBasedNN
    if precomputed_knn is not None:
        indices, distances = precomputed_knn[:2]
        affinities = PerplexityBasedNN(
            knn_index=_PrecomputedKNNIndex(indices, distances),
            perplexity=perplexity,
        )
        return np.array(OpenTSNE(n_components=dimensions).fit(embeddings, affinities=affinities))
    return np.array(OpenTSNE(n_components=dimensions, perplexity=perplexity).fit(embeddings))


def run_dr_umap(
    embeddings,
    dimensions: int,
    n_neighbors: Optional[int] = 30,
    min_dist: Optional[float] = 0.6,
    low_memory: bool = False,
    precomputed_knn: Optional[tuple] = None,
):
    """
    min_dist default raised from umap-learn's own library default (0.25) to
    0.6. Both UMAP and t-SNE already use non-random, structure-aware
    initialization here (UMAP's own "spectral" default, openTSNE's own "pca"
    default) -- per Kobak & Linderman (Nat. Biotechnol. 2021), initialization
    matters more than algorithm choice for preserving global structure, so
    that part was already right. What was missing is contrast: at
    library-default settings, UMAP and t-SNE sit close together on the
    attraction/repulsion spectrum (Bohm, Berens & Kobak, JMLR 2022) and tend
    to produce visibly similar layouts. A low min_dist packs points into
    tight, well-separated local clumps (favoring local/cluster structure,
    t-SNE's niche); raising it spreads points out, favoring preservation of
    relative global distances instead. t-SNE's perplexity is left as-is (its
    role here is local neighbourhood fidelity), so the pair now sits at
    genuinely different points on the spectrum rather than two similar-
    looking views. This is a parameter choice, not a proof -- worth an eyeball
    check against real data (e.g. dataset 9) once run, not just the theory.
    """
    from umap import UMAP
    kwargs = dict(n_components=dimensions, n_neighbors=n_neighbors, min_dist=min_dist)
    if precomputed_knn is not None:
        kwargs["precomputed_knn"] = precomputed_knn
    try:
        reducer = UMAP(**kwargs, low_memory=low_memory)
    except TypeError:
        reducer = UMAP(**kwargs)
    return reducer.fit_transform(np.asarray(embeddings))
