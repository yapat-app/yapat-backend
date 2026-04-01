from typing import Optional

def run_dr_umap(embeddings, n_components, n_neighbours: Optional[int] = 30, min_dist: Optional[float] = 0.25):
    from umap import UMAP
    dim_reducer = UMAP(n_components=n_components, n_neighbors=n_neighbours, min_dist=min_dist)
    reduced_embeddings = dim_reducer.fit_transform(embeddings)
    return reduced_embeddings