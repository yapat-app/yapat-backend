import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

# When input dimensionality exceeds this threshold, pre-reduce via PCA before
# running UMAP or t-SNE/Isomap. Drops memory ~20x for 1024-dim embeddings.
_UMAP_PRE_REDUCE_THRESHOLD = 50
_UMAP_PRE_REDUCE_DIMS = 50


def run_dr_pca(embeddings, dimensions: int):
    reducer = PCA(n_components=dimensions)
    return reducer.fit_transform(embeddings)


def run_dr_isomap(embeddings, dimensions: int, n_neighbors: Optional[int] = 30):
    reducer = Isomap(n_neighbors=n_neighbors, n_components=dimensions)
    return reducer.fit_transform(embeddings)


def run_dr_tsne(embeddings, dimensions: int, perplexity: Optional[int] = 30):
    reducer = TSNE(n_components=dimensions, perplexity=perplexity)
    return reducer.fit_transform(embeddings)


def _maybe_pre_reduce(embeddings: np.ndarray) -> np.ndarray:
    """PCA pre-reduction: if dims > threshold, reduce to _UMAP_PRE_REDUCE_DIMS first.

    UMAP and t-SNE kNN search memory scales with input dimensionality.
    Pre-reducing 1024-dim BirdNET embeddings to 50 dims cuts peak RAM ~20x
    with negligible loss of neighbourhood structure.
    """
    if embeddings.shape[1] > _UMAP_PRE_REDUCE_THRESHOLD:
        n_components = min(_UMAP_PRE_REDUCE_DIMS, embeddings.shape[0] - 1, embeddings.shape[1])
        return PCA(n_components=n_components).fit_transform(embeddings)
    return embeddings


def run_dr_umap(
    embeddings,
    dimensions: int,
    n_neighbors: Optional[int] = 30,
    min_dist: Optional[float] = 0.25,
    low_memory: bool = False,
):
    from umap import UMAP
    reduced = _maybe_pre_reduce(np.asarray(embeddings))
    kwargs = dict(
        n_components=dimensions,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    try:
        reducer = UMAP(**kwargs, low_memory=low_memory)
    except TypeError:
        reducer = UMAP(**kwargs)
    return reducer.fit_transform(reduced)