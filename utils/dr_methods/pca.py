from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as pca

def run_dr_pca(embeddings, dimensions):
    dim_reducer = pca(n_components=dimensions)
    reduced_embeddings = dim_reducer.fit(embeddings)
    return reduced_embeddings


