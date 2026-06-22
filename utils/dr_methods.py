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
    """Compute kNN graph with pynndescent (UMAP's own backend).

    Returns (indices, distances, index) where indices and distances are each
    of shape (n, n_neighbors). The NNDescent index is included as the third
    element so UMAP can receive the full 3-tuple via precomputed_knn.
    """
    from pynndescent import NNDescent
    index = NNDescent(embeddings, n_neighbors=n_neighbors)
    indices, distances = index.neighbor_graph
    return indices, distances, index


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
    min_dist: Optional[float] = 0.25,
    low_memory: bool = False,
    precomputed_knn: Optional[tuple] = None,
):
    from umap import UMAP
    kwargs = dict(n_components=dimensions, n_neighbors=n_neighbors, min_dist=min_dist)
    if precomputed_knn is not None:
        kwargs["precomputed_knn"] = precomputed_knn
    try:
        reducer = UMAP(**kwargs, low_memory=low_memory)
    except TypeError:
        reducer = UMAP(**kwargs)
    return reducer.fit_transform(np.asarray(embeddings))
