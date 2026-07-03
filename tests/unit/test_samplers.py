"""Unit tests for active_learning/samplers.py -- diversity()'s Flat/HNSW switch.

Uses small synthetic numpy/torch arrays. Requires torch and faiss (both
already project dependencies). No prior test coverage existed for this
module before the DIVERSITY_HNSW_MIN_NL change.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

RNG = np.random.default_rng(42)


def _brute_force_nearest(z_u: np.ndarray, z_l: np.ndarray) -> np.ndarray:
    """Independent (no FAISS) reference: 1-NN Euclidean distance, row-wise."""
    diff = z_u[:, None, :] - z_l[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def _unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture()
def small_labeled_set():
    """Nl well below the default DIVERSITY_HNSW_MIN_NL (500) -- Flat path."""
    z_u = _unit_vectors(200, 16, RNG)
    z_l = _unit_vectors(30, 16, RNG)
    return torch.tensor(z_u), torch.tensor(z_l)


@pytest.fixture()
def large_labeled_set():
    """Nl above the default DIVERSITY_HNSW_MIN_NL (500) -- HNSW path."""
    z_u = _unit_vectors(1000, 32, RNG)
    z_l = _unit_vectors(800, 32, RNG)
    return torch.tensor(z_u), torch.tensor(z_l)


class TestDiversityFlatPath:
    def test_matches_brute_force_reference(self, small_labeled_set):
        from active_learning.samplers import diversity, _to_np

        z_u, z_l = small_labeled_set
        scores = diversity(z_u, z_l)

        # diversity() L2-normalizes internally (_to_np) before computing
        # distance -- replicate that here for a fair independent comparison.
        ref = _brute_force_nearest(_to_np(z_u), _to_np(z_l))
        ref_clamped = np.clip(ref, 0.0, 1.0)

        assert torch.allclose(scores, torch.tensor(ref_clamped, dtype=torch.float32), atol=1e-4)

    def test_stays_on_flat_below_default_threshold(self, small_labeled_set):
        """Nl=30 is below DIVERSITY_HNSW_MIN_NL=500, so results must be exact,
        not merely close -- this is what distinguishes "used Flat" from
        "used HNSW and got lucky"."""
        from active_learning.samplers import diversity, _to_np

        z_u, z_l = small_labeled_set
        scores = diversity(z_u, z_l)
        ref = np.clip(_brute_force_nearest(_to_np(z_u), _to_np(z_l)), 0.0, 1.0)

        max_err = (scores.numpy() - ref).__abs__().max()
        assert max_err < 1e-5, f"expected near-exact match on Flat path, max_err={max_err}"


class TestDiversityHnswPath:
    def test_high_agreement_with_brute_force(self, large_labeled_set):
        from active_learning.samplers import diversity, _to_np

        z_u, z_l = large_labeled_set
        scores = diversity(z_u, z_l)
        ref = np.clip(_brute_force_nearest(_to_np(z_u), _to_np(z_l)), 0.0, 1.0)

        # Approximate: allow small deviation, but results should be close for
        # well-separated random data.
        mean_abs_err = float((scores.numpy() - ref).__abs__().mean())
        assert mean_abs_err < 0.05, f"HNSW approximation error too high: {mean_abs_err}"

    def test_threshold_override_forces_hnsw_at_small_nl(self, small_labeled_set):
        """hnsw_min_nl=0 forces the HNSW branch even for a tiny labeled set --
        should run without error and stay in [0, 1]."""
        from active_learning.samplers import diversity

        z_u, z_l = small_labeled_set
        scores = diversity(z_u, z_l, hnsw_min_nl=0)
        assert scores.shape == (z_u.shape[0],)
        assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)

    def test_threshold_override_forces_flat_at_large_nl(self, large_labeled_set):
        """hnsw_min_nl larger than Nl forces Flat even for a large labeled
        set -- should match the brute-force reference exactly."""
        from active_learning.samplers import diversity, _to_np

        z_u, z_l = large_labeled_set
        scores = diversity(z_u, z_l, hnsw_min_nl=10_000_000)
        ref = np.clip(_brute_force_nearest(_to_np(z_u), _to_np(z_l)), 0.0, 1.0)

        max_err = (scores.numpy() - ref).__abs__().max()
        assert max_err < 1e-4, f"expected near-exact match when forced onto Flat, max_err={max_err}"


class TestDiversityEdgeCases:
    def test_empty_unlabeled_returns_empty(self):
        from active_learning.samplers import diversity

        z_u = torch.empty((0, 16))
        z_l = torch.tensor(_unit_vectors(10, 16, RNG))
        scores = diversity(z_u, z_l)
        assert scores.shape == (0,)

    def test_empty_labeled_returns_zeros(self):
        from active_learning.samplers import diversity

        z_u = torch.tensor(_unit_vectors(10, 16, RNG))
        z_l = torch.empty((0, 16))
        scores = diversity(z_u, z_l)
        assert scores.shape == (10,)
        assert torch.all(scores == 0.0)


class TestDiversityApproxError:
    def test_returns_expected_keys(self, large_labeled_set):
        from active_learning.samplers import diversity_approx_error

        z_u, z_l = large_labeled_set
        result = diversity_approx_error(z_u, z_l)
        assert set(result.keys()) == {
            "recall_at_1", "mean_abs_distance_error", "max_abs_distance_error", "n",
        }
        assert result["n"] == z_u.shape[0]
        assert 0.0 <= result["recall_at_1"] <= 1.0

    def test_high_recall_on_well_separated_random_data(self, large_labeled_set):
        from active_learning.samplers import diversity_approx_error

        z_u, z_l = large_labeled_set
        result = diversity_approx_error(z_u, z_l)
        assert result["recall_at_1"] > 0.9

    def test_empty_inputs_return_none_stats(self):
        from active_learning.samplers import diversity_approx_error

        result = diversity_approx_error(torch.empty((0, 16)), torch.empty((0, 16)))
        assert result["recall_at_1"] is None
        assert result["n"] == 0
