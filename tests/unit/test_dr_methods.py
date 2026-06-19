"""Unit tests for utils/dr_methods.py.

Uses small synthetic numpy arrays. All external dependencies (pynndescent,
umap-learn, openTSNE) must be installed — tests fail with ImportError if not.
"""
import logging

import numpy as np
import pytest

RNG = np.random.default_rng(42)


@pytest.fixture()
def X():
    """100 samples × 128 dims — above _PRE_REDUCE_DIMS (50)."""
    return RNG.standard_normal((100, 128)).astype("float32")


@pytest.fixture()
def X_low_dim():
    """100 samples × 20 dims — below _PRE_REDUCE_DIMS."""
    return RNG.standard_normal((100, 20)).astype("float32")


@pytest.fixture()
def X_low_var():
    """100 samples × 200 independent dims — 50 PCs explain well under 95%."""
    return RNG.standard_normal((100, 200)).astype("float32")


@pytest.fixture()
def X_r(X):
    from utils.dr_methods import pre_reduce_pca
    return pre_reduce_pca(X)


@pytest.fixture()
def knn(X_r):
    """3-tuple (indices, distances, NNDescent index) as returned by build_knn_graph."""
    from utils.dr_methods import build_knn_graph
    return build_knn_graph(X_r, n_neighbors=15)


# ---------------------------------------------------------------------------
# pre_reduce_pca
# ---------------------------------------------------------------------------

class TestPreReducePca:
    def test_output_shape_high_dim(self, X):
        from utils.dr_methods import pre_reduce_pca, _PRE_REDUCE_DIMS
        result = pre_reduce_pca(X)
        assert result.shape == (X.shape[0], _PRE_REDUCE_DIMS)

    def test_output_shape_low_dim(self, X_low_dim):
        from utils.dr_methods import pre_reduce_pca
        result = pre_reduce_pca(X_low_dim)
        assert result.shape == X_low_dim.shape

    def test_max_vis_dims_respected(self, X):
        from utils.dr_methods import pre_reduce_pca
        result = pre_reduce_pca(X, max_vis_dims=3)
        assert result.shape[1] >= 3

    def test_2d_slice_shape(self, X):
        from utils.dr_methods import pre_reduce_pca
        X_r = pre_reduce_pca(X, max_vis_dims=2)
        assert X_r[:, :2].shape == (X.shape[0], 2)

    def test_warns_on_low_explained_variance(self, X_low_var, caplog):
        from utils.dr_methods import pre_reduce_pca
        with caplog.at_level(logging.WARNING, logger="utils.dr_methods"):
            pre_reduce_pca(X_low_var)
        assert any(
            "variance explained" in r.message
            for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_no_warning_on_high_explained_variance(self, caplog):
        from utils.dr_methods import pre_reduce_pca
        # Rank-5 signal embedded in 128 dims — 50 PCs capture it fully.
        latent = RNG.standard_normal((200, 5))
        proj = RNG.standard_normal((5, 128))
        X = (latent @ proj).astype("float32")
        with caplog.at_level(logging.WARNING, logger="utils.dr_methods"):
            pre_reduce_pca(X)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# build_knn_graph
# ---------------------------------------------------------------------------

class TestBuildKnnGraph:
    def test_output_shapes(self, X_r):
        from utils.dr_methods import build_knn_graph
        indices, distances, _ = build_knn_graph(X_r, n_neighbors=10)
        assert indices.shape == (X_r.shape[0], 10)
        assert distances.shape == (X_r.shape[0], 10)

    def test_distances_non_negative(self, X_r):
        from utils.dr_methods import build_knn_graph
        _, distances, _ = build_knn_graph(X_r, n_neighbors=10)
        assert np.all(distances >= 0)

    def test_indices_in_range(self, X_r):
        from utils.dr_methods import build_knn_graph
        n = X_r.shape[0]
        indices, _, _ = build_knn_graph(X_r, n_neighbors=10)
        assert np.all(indices >= 0)
        assert np.all(indices < n)


# ---------------------------------------------------------------------------
# run_dr_umap
# ---------------------------------------------------------------------------

class TestRunDrUmap:
    def test_2d_output_shape(self, X_r, knn):
        from utils.dr_methods import run_dr_umap
        result = run_dr_umap(X_r, dimensions=2, n_neighbors=15, precomputed_knn=knn)
        assert result.shape == (X_r.shape[0], 2)

    def test_3d_output_shape(self, X):
        from utils.dr_methods import pre_reduce_pca, build_knn_graph, run_dr_umap
        X_r = pre_reduce_pca(X, max_vis_dims=3)
        knn3 = build_knn_graph(X_r, n_neighbors=15)
        result = run_dr_umap(X_r, dimensions=3, n_neighbors=15, precomputed_knn=knn3)
        assert result.shape == (X_r.shape[0], 3)

    def test_without_precomputed_knn(self, X_r):
        from utils.dr_methods import run_dr_umap
        result = run_dr_umap(X_r, dimensions=2)
        assert result.shape == (X_r.shape[0], 2)


# ---------------------------------------------------------------------------
# run_dr_tsne
# ---------------------------------------------------------------------------

class TestRunDrTsne:
    def test_2d_output_shape(self, X_r, knn):
        from utils.dr_methods import run_dr_tsne
        result = run_dr_tsne(X_r, dimensions=2, precomputed_knn=knn)
        assert result.shape == (X_r.shape[0], 2)

    def test_without_precomputed_knn(self, X_r):
        from utils.dr_methods import run_dr_tsne
        result = run_dr_tsne(X_r, dimensions=2)
        assert result.shape == (X_r.shape[0], 2)


# ---------------------------------------------------------------------------
# run_dr_isomap
# ---------------------------------------------------------------------------

class TestRunDrIsomap:
    def test_2d_output_shape_with_precomputed(self, X_r, knn):
        from utils.dr_methods import run_dr_isomap
        result = run_dr_isomap(X_r, dimensions=2, n_neighbors=15, precomputed_knn=knn)
        assert result.shape == (X_r.shape[0], 2)

    def test_without_precomputed_knn(self, X_r):
        from utils.dr_methods import run_dr_isomap
        result = run_dr_isomap(X_r, dimensions=2)
        assert result.shape == (X_r.shape[0], 2)

    def test_sparse_matrix_shape_and_values(self, X_r):
        """kNN indices and distances are assembled into a valid sparse matrix."""
        import scipy.sparse as sp
        from utils.dr_methods import build_knn_graph
        k = 10
        indices, distances, _ = build_knn_graph(X_r, n_neighbors=k)
        n = len(indices)
        rows = np.repeat(np.arange(n), indices.shape[1])
        dist_matrix = sp.csr_matrix(
            (distances.ravel(), (rows, indices.ravel())), shape=(n, n)
        )
        assert dist_matrix.shape == (n, n)
        assert dist_matrix.nnz == n * k
        assert np.all(dist_matrix.data >= 0)
