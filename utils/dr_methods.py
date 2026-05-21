from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE


def run_dr_pca(embeddings, dimensions: int):
    reducer = PCA(n_components=dimensions)
    return reducer.fit_transform(embeddings)


def run_dr_isomap(embeddings, dimensions: int, n_neighbors: Optional[int] = 30):
    reducer = Isomap(n_neighbors=n_neighbors, n_components=dimensions)
    return reducer.fit_transform(embeddings)


def run_dr_tsne(embeddings, dimensions: int, perplexity: Optional[int] = 30):
    reducer = TSNE(n_components=dimensions, perplexity=perplexity)
    return reducer.fit_transform(embeddings)


def run_dr_umap(
    embeddings,
    dimensions: int,
    n_neighbors: Optional[int] = 30,
    min_dist: Optional[float] = 0.25,
):
    from umap import UMAP
    reducer = UMAP(n_components=dimensions, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(embeddings)