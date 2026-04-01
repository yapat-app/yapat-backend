from typing import Optional
from sklearn.manifold import TSNE as tsne

def run_dr_isomap(embeddings, dimensions, perplexity: Optional[int] = 30):
    dim_reducer = tsne(n_components=dimensions, perplexity=perplexity)
    reduced_embeddings = dim_reducer.fit_transform(embeddings)
    return reduced_embeddings