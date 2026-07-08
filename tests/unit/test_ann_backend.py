"""Unit tests for utils/ann_backend.py -- the shared FAISS-backed ANN
abstraction now used by DR's build_knn_graph (utils/dr_methods.py).

AL's samplers.py still has its own ad hoc faiss calls (diversity/density);
unifying those onto this module is a separate, deferred change.
"""
import numpy as np
import pytest

RNG = np.random.default_rng(7)


def _unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _brute_force_self_knn(X: np.ndarray, k: int) -> np.ndarray:
    """Independent (no FAISS) reference for self-search k-NN indices, sorted
    by distance, self included as neighbour 0."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    return np.argsort(d2, axis=1)[:, :k]


class TestBuildIndexAndSearch:
    def test_self_search_includes_self_as_first_neighbor(self):
        from utils.ann_backend import nearest_neighbors

        X = _unit_vectors(40, 8, RNG)
        indices, distances, _ = nearest_neighbors(X, X, k=5, exact=True)
        assert np.all(indices[:, 0] == np.arange(40))
        assert np.allclose(distances[:, 0], 0.0, atol=1e-5)

    def test_self_search_matches_brute_force_on_exact_path(self):
        from utils.ann_backend import nearest_neighbors

        X = _unit_vectors(60, 12, RNG)
        indices, distances, _ = nearest_neighbors(X, X, k=6, exact=True)
        ref = _brute_force_self_knn(X, 6)
        assert np.array_equal(indices, ref)

    def test_distances_are_true_euclidean_not_squared(self):
        from utils.ann_backend import nearest_neighbors

        X = _unit_vectors(30, 4, RNG)
        _, distances, _ = nearest_neighbors(X, X, k=5, exact=True)
        # unit vectors: max pairwise Euclidean distance is 2 (antipodal).
        # Squared distances would range up to 4 -- if this were squared L2
        # leaking through unconverted, we'd see values > 2 routinely.
        assert distances.max() <= 2.0 + 1e-4

    def test_asymmetric_search_shape_and_nonnegativity(self):
        from utils.ann_backend import nearest_neighbors

        z_u = _unit_vectors(25, 10, RNG)
        z_l = _unit_vectors(5, 10, RNG)
        indices, distances, _ = nearest_neighbors(z_u, z_l, k=1, exact=True)
        assert indices.shape == (25, 1)
        assert distances.shape == (25, 1)
        assert np.all(distances >= 0)
        assert np.all(indices < 5)

    def test_k_clamped_to_index_size(self):
        from utils.ann_backend import nearest_neighbors

        z_u = _unit_vectors(10, 6, RNG)
        z_l = _unit_vectors(3, 6, RNG)
        indices, distances, _ = nearest_neighbors(z_u, z_l, k=100, exact=True)
        # only 3 index points exist -- can't return more than 3 neighbours
        assert indices.shape == (10, 3)


class TestExactVsApproxSelection:
    def test_auto_selects_flat_below_threshold(self):
        from utils.ann_backend import build_index

        X = _unit_vectors(50, 8, RNG)
        index = build_index(X, hnsw_min_n=1000)
        assert not hasattr(index, "hnsw") or index.hnsw is None or type(index).__name__ == "IndexFlatL2"

    def test_auto_selects_hnsw_at_or_above_threshold(self):
        from utils.ann_backend import build_index

        X = _unit_vectors(50, 8, RNG)
        index = build_index(X, hnsw_min_n=10)
        assert "HNSW" in type(index).__name__

    def test_exact_override_forces_flat_regardless_of_size(self):
        from utils.ann_backend import build_index

        X = _unit_vectors(50, 8, RNG)
        index = build_index(X, exact=True, hnsw_min_n=10)
        assert "HNSW" not in type(index).__name__

    def test_exact_false_forces_hnsw_regardless_of_size(self):
        from utils.ann_backend import build_index

        X = _unit_vectors(50, 8, RNG)
        index = build_index(X, exact=False, hnsw_min_n=1000)
        assert "HNSW" in type(index).__name__


class TestGpuProbeIsInert:
    def test_gpu_available_does_not_raise_on_cpu_only_faiss(self):
        from utils.ann_backend import _gpu_available

        # Just must not raise -- whether it returns True or False depends on
        # which faiss build/hardware the test happens to run on.
        result = _gpu_available()
        assert isinstance(result, bool)
