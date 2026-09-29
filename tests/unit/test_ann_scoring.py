"""
Acquisition scoring against the cached all-snippets index.

The index now covers labelled points too, so "nearest unlabelled neighbour"
means "search wider and drop the labelled hits". These tests pin that the
resulting scores still match an exact brute-force computation over the
unlabelled set only -- the definitions are meant to be unchanged, only the
neighbour lookup.

See docs/superpowers/plans/2026-09-22-retrain-scaling-fixes.md (R2 / option C).
"""

import numpy as np
import pytest

from app.config import settings
from app.services.pam_al import _ann_index_cache as idx_cache
from app.services.pam_al import _scoring


@pytest.fixture(autouse=True)
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PAM_CHECKPOINTS_DIR", str(tmp_path / "pam" / "checkpoints"))
    idx_cache._MEMORY_CACHE.clear()
    yield
    idx_cache._MEMORY_CACHE.clear()


def _fingerprint(x: np.ndarray) -> dict:
    return {"count": int(x.shape[0]), "max_vector_id": int(x.shape[0]), "dim": int(x.shape[1])}


def _world(n=600, dim=16, n_labeled=90, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    labeled_rows = np.sort(rng.choice(n, size=n_labeled, replace=False)).astype(np.int64)
    is_labeled = np.zeros(n, dtype=bool)
    is_labeled[labeled_rows] = True
    unlabeled_rows = np.flatnonzero(~is_labeled).astype(np.int64)
    index = idx_cache.load_or_build_index(x, 1, 1, _fingerprint(x))
    return x, index, unlabeled_rows, labeled_rows, is_labeled


def _brute_density(x, unlabeled_rows, k):
    """Exact 1 / mean distance to the k nearest *unlabelled* neighbours."""
    vectors = idx_cache.normalize_rows(x[unlabeled_rows])
    out = np.empty(vectors.shape[0], dtype=np.float64)
    for i in range(vectors.shape[0]):
        d = np.linalg.norm(vectors - vectors[i], axis=1)
        d = np.delete(d, i)
        out[i] = 1.0 / (np.sort(d)[:k].mean() + 1e-8)
    return out


def test_density_matches_brute_force():
    x, index, unlabeled_rows, _, is_labeled = _world()
    got = _scoring.compute_density(index, x, unlabeled_rows, is_labeled, k=10)
    want = _brute_density(x, unlabeled_rows, k=10)

    assert np.all(np.isfinite(got))
    rel = np.abs(got - want) / want
    assert rel.max() < 0.01, f"worst relative error {rel.max():.4f}"


def test_labeled_neighbours_are_excluded():
    """The behaviour that changed: a labelled twin must not count as a neighbour."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    # Row 1 is a near-duplicate of row 0. Labelling it must leave row 0's
    # density reflecting its *other* neighbours only.
    x[1] = x[0] + 1e-6

    is_labeled = np.zeros(400, dtype=bool)
    is_labeled[1] = True
    unlabeled_rows = np.flatnonzero(~is_labeled).astype(np.int64)
    index = idx_cache.load_or_build_index(x, 2, 1, _fingerprint(x))

    got = _scoring.compute_density(index, x, unlabeled_rows, is_labeled, k=5)
    want = _brute_density(x, unlabeled_rows, k=5)

    row0 = int(np.flatnonzero(unlabeled_rows == 0)[0])
    assert np.isclose(got[row0], want[row0], rtol=0.01)

    # Sanity: had the labelled twin counted, the distance would collapse and
    # the score would be orders of magnitude larger.
    with_twin = 1.0 / (np.mean([0.0] * 1 + [1.0] * 4) + 1e-8)
    assert got[row0] < with_twin * 100


def test_nearest_labeled_matches_brute_force():
    x, _, unlabeled_rows, labeled_rows, _ = _world()
    got = _scoring.compute_nearest_labeled(x, unlabeled_rows, labeled_rows)

    u = idx_cache.normalize_rows(x[unlabeled_rows])
    l = idx_cache.normalize_rows(x[labeled_rows])
    want = np.array([np.linalg.norm(l - row, axis=1).min() for row in u])

    assert np.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_nearest_labeled_with_no_labels_is_uniform():
    x, _, _, _, _ = _world()
    unlabeled_rows = np.arange(x.shape[0], dtype=np.int64)
    got = _scoring.compute_nearest_labeled(x, unlabeled_rows, np.empty(0, dtype=np.int64))
    assert np.all(got == 1.0)


def test_diversity_matches_exact_greedy():
    """With update_k >= n every candidate is visited, so the greedy is deterministic."""
    x, index, unlabeled_rows, labeled_rows, is_labeled = _world(n=300, dim=8, n_labeled=40, seed=3)
    nearest = _scoring.compute_nearest_labeled(x, unlabeled_rows, labeled_rows)

    got = _scoring.compute_diversity(
        index, x, unlabeled_rows, is_labeled, nearest, num_centers=5, update_k=len(unlabeled_rows)
    )

    vectors = idx_cache.normalize_rows(x[unlabeled_rows])
    min_dist = nearest.astype(np.float32).copy()
    picked = np.zeros(vectors.shape[0], dtype=bool)
    for _ in range(5):
        center = int(np.argmax(np.where(picked, -np.inf, min_dist)))
        picked[center] = True
        d = np.linalg.norm(vectors - vectors[center], axis=1).astype(np.float32)
        min_dist = np.minimum(min_dist, d)

    assert np.allclose(got, min_dist, rtol=1e-3, atol=1e-4)


def test_diversity_never_exceeds_nearest_labeled():
    """The greedy only ever reduces scores; it must not invent novelty."""
    x, index, unlabeled_rows, labeled_rows, is_labeled = _world(seed=11)
    nearest = _scoring.compute_nearest_labeled(x, unlabeled_rows, labeled_rows)
    got = _scoring.compute_diversity(
        index, x, unlabeled_rows, is_labeled, nearest, num_centers=20, update_k=50
    )
    assert np.all(got <= nearest + 1e-5)


def test_index_is_cached_and_reused(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    fp = _fingerprint(x)

    first = idx_cache.load_or_build_index(x, 9, 9, fp)
    cache_dir = idx_cache.get_index_dir(9, 9)
    assert idx_cache._cache_files_exist(cache_dir)

    # Drop the in-process cache: the next call must come off disk, not rebuild.
    idx_cache._MEMORY_CACHE.clear()
    second = idx_cache.load_or_build_index(x, 9, 9, fp)
    assert second.ntotal == first.ntotal == 200


def test_changed_fingerprint_rebuilds():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(150, 8)).astype(np.float32)
    idx_cache.load_or_build_index(x, 4, 4, _fingerprint(x))

    bigger = rng.normal(size=(250, 8)).astype(np.float32)
    idx_cache._MEMORY_CACHE.clear()
    rebuilt = idx_cache.load_or_build_index(bigger, 4, 4, _fingerprint(bigger))
    assert rebuilt.ntotal == 250


def test_invalidate_removes_the_cache():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    idx_cache.load_or_build_index(x, 3, 3, _fingerprint(x))
    assert idx_cache._cache_files_exist(idx_cache.get_index_dir(3, 3))

    idx_cache.invalidate_ann_index_cache(3, 3)
    assert not idx_cache._cache_files_exist(idx_cache.get_index_dir(3, 3))
