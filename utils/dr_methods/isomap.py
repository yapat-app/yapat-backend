from typing import Optional
from sklearn.manifold import Isomap


def run_dr_isomap(embeddings, dimensions, n_neighbours: Optional[int] = 30):
    dim_reducer = Isomap(n_neighbours=n_neighbours, n_components=dimensions)
    reduced_embeddings = dim_reducer.fit_transform(embeddings)
    return reduced_embeddings