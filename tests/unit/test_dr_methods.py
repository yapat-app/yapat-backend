"""Unit tests for utils/dr_methods.py.

Uses small synthetic numpy arrays so no DB or heavy models are needed.
External dependencies (pynndescent, umap-learn, openTSNE) are mocked via
sys.modules so the tests run regardless of what is installed.
"""
import logging
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _fake_knn(n: int, k: int):
    indices = np.tile(np.arange(k, dtype=np.int32), (n, 1))
    distances = RNG.random((n, k)).astype("float32")
    return indices, distances


# ---------------------------------------------------------------------------
# pre_reduce_pca  (sklearn only — no mocking needed)
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
# build_knn_graph  (mocks pynndescent)
# ---------------------------------------------------------------------------

class TestBuildKnnGraph:
    @pytest.fixture(autouse=True)
    def mock_pynndescent(self, X):
        n, k = X.shape[0], 10
        fake_idx, fake_dist = _fake_knn(n, k)

        mock_index = MagicMock()
        mock_index.neighbor_graph = (fake_idx, fake_dist)

        mock_module = MagicMock()
        mock_module.NNDescent.return_value = mock_index

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pynndescent", mock_module)
            self._mock = mock_module
            self._fake_idx = fake_idx
            self._fake_dist = fake_dist
            yield

    def test_output_shapes(self, X):
        from utils.dr_methods import build_knn_graph
        indices, distances = build_knn_graph(X, n_neighbors=10)
        assert indices.shape == (X.shape[0], 10)
        assert distances.shape == (X.shape[0], 10)

    def test_distances_non_negative(self, X):
        from utils.dr_methods import build_knn_graph
        _, distances = build_knn_graph(X, n_neighbors=10)
        assert np.all(distances >= 0)

    def test_indices_in_range(self, X):
        from utils.dr_methods import build_knn_graph
        indices, _ = build_knn_graph(X, n_neighbors=10)
        assert np.all(indices >= 0)

    def test_nndescent_called_with_correct_n_neighbors(self, X):
        from utils.dr_methods import build_knn_graph
        build_knn_graph(X, n_neighbors=15)
        self._mock.NNDescent.assert_called_once_with(X, n_neighbors=15)


# ---------------------------------------------------------------------------
# run_dr_umap  (mocks umap + pynndescent)
# ---------------------------------------------------------------------------

class TestRunDrUmap:
    @pytest.fixture(autouse=True)
    def mock_umap(self, X):
        n = X.shape[0]

        def make_mock_reducer(dims):
            r = MagicMock()
            r.fit_transform.return_value = np.zeros((n, dims), dtype="float32")
            return r

        mock_umap_module = MagicMock()
        mock_umap_module.UMAP.side_effect = lambda **kw: make_mock_reducer(kw["n_components"])

        mock_pynn = MagicMock()
        mock_pynn.NNDescent.return_value.neighbor_graph = _fake_knn(n, 15)

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "umap", mock_umap_module)
            mp.setitem(sys.modules, "pynndescent", mock_pynn)
            self._mock_umap = mock_umap_module
            yield

    def test_2d_output_shape(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_umap
        X_r = pre_reduce_pca(X)
        knn = build_knn_graph(X_r, n_neighbors=15)
        result = run_dr_umap(X_r, dimensions=2, n_neighbors=15, precomputed_knn=knn)
        assert result.shape == (X.shape[0], 2)

    def test_3d_output_shape(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_umap
        X_r = pre_reduce_pca(X, max_vis_dims=3)
        knn = build_knn_graph(X_r, n_neighbors=15)
        result = run_dr_umap(X_r, dimensions=3, n_neighbors=15, precomputed_knn=knn)
        assert result.shape == (X.shape[0], 3)

    def test_precomputed_knn_forwarded_to_umap(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_umap
        X_r = pre_reduce_pca(X)
        knn = build_knn_graph(X_r, n_neighbors=15)
        run_dr_umap(X_r, dimensions=2, n_neighbors=15, precomputed_knn=knn)
        call_kwargs = self._mock_umap.UMAP.call_args.kwargs
        assert "precomputed_knn" in call_kwargs
        assert call_kwargs["precomputed_knn"] is knn

    def test_without_precomputed_knn(self, X):
        from utils.dr_methods import run_dr_umap
        result = run_dr_umap(X, dimensions=2)
        call_kwargs = self._mock_umap.UMAP.call_args.kwargs
        assert "precomputed_knn" not in call_kwargs
        assert result.shape == (X.shape[0], 2)


# ---------------------------------------------------------------------------
# run_dr_tsne  (mocks openTSNE + pynndescent)
# ---------------------------------------------------------------------------

class TestRunDrTsne:
    @pytest.fixture(autouse=True)
    def mock_opentsne(self, X):
        n = X.shape[0]

        def make_embedding(dims):
            return np.zeros((n, dims), dtype="float32")

        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit.side_effect = lambda data, affinities=None: make_embedding(
            mock_tsne_instance._dims
        )

        mock_ot = MagicMock()
        def tsne_constructor(**kw):
            mock_tsne_instance._dims = kw.get("n_components", 2)
            return mock_tsne_instance
        mock_ot.TSNE.side_effect = tsne_constructor

        mock_affinity = MagicMock()
        mock_pynn = MagicMock()
        mock_pynn.NNDescent.return_value.neighbor_graph = _fake_knn(n, 30)

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "openTSNE", mock_ot)
            mp.setitem(sys.modules, "openTSNE.affinity", mock_affinity)
            mp.setitem(sys.modules, "pynndescent", mock_pynn)
            self._mock_affinity = mock_affinity
            yield

    def test_2d_output_shape(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_tsne
        X_r = pre_reduce_pca(X)
        knn = build_knn_graph(X_r, n_neighbors=30)
        result = run_dr_tsne(X_r, dimensions=2, precomputed_knn=knn)
        assert result.shape == (X.shape[0], 2)

    def test_precomputed_knn_uses_perplexity_based_nn(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_tsne
        X_r = pre_reduce_pca(X)
        knn = build_knn_graph(X_r, n_neighbors=30)
        run_dr_tsne(X_r, dimensions=2, precomputed_knn=knn)
        self._mock_affinity.PerplexityBasedNN.assert_called_once()

    def test_without_precomputed_knn(self, X):
        from utils.dr_methods import pre_reduce_pca, run_dr_tsne
        X_r = pre_reduce_pca(X)
        result = run_dr_tsne(X_r, dimensions=2)
        assert result.shape == (X.shape[0], 2)
        self._mock_affinity.PerplexityBasedNN.assert_not_called()


# ---------------------------------------------------------------------------
# run_dr_isomap  (mocks pynndescent; sklearn Isomap is available)
# ---------------------------------------------------------------------------

class TestRunDrIsomap:
    @pytest.fixture(autouse=True)
    def mock_pynndescent(self, X):
        n, k = X.shape[0], 10
        fake_idx, fake_dist = _fake_knn(n, k)

        mock_module = MagicMock()
        mock_module.NNDescent.return_value.neighbor_graph = (fake_idx, fake_dist)

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pynndescent", mock_module)
            yield

    def test_2d_output_shape_with_precomputed(self, X):
        from utils.dr_methods import build_knn_graph, pre_reduce_pca, run_dr_isomap
        X_r = pre_reduce_pca(X)
        knn = build_knn_graph(X_r, n_neighbors=10)
        result = run_dr_isomap(X_r, dimensions=2, n_neighbors=10, precomputed_knn=knn)
        assert result.shape == (X.shape[0], 2)

    def test_without_precomputed_knn(self, X):
        from utils.dr_methods import run_dr_isomap
        result = run_dr_isomap(X, dimensions=2)
        assert result.shape == (X.shape[0], 2)

    def test_sparse_matrix_shape_and_values(self, X):
        """kNN indices and distances are assembled into a valid sparse matrix."""
        import scipy.sparse as sp
        from utils.dr_methods import build_knn_graph, pre_reduce_pca
        X_r = pre_reduce_pca(X)
        indices, distances = build_knn_graph(X_r, n_neighbors=10)
        n = len(indices)
        rows = np.repeat(np.arange(n), indices.shape[1])
        dist_matrix = sp.csr_matrix(
            (distances.ravel(), (rows, indices.ravel())), shape=(n, n)
        )
        assert dist_matrix.shape == (n, n)
        assert dist_matrix.nnz == n * 10
        assert np.all(dist_matrix.data >= 0)
