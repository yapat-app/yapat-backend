import json

import numpy as np
import pytest

from app.services.explore import storage
from app.services.explore.sampling import density_grid, priority_permutation, stratified_sample


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "EXPLORE_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_sample_returns_every_point_under_cap():
    x = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    idx = stratified_sample(x, y, cap=10, priority=priority_permutation(4, 1))
    assert idx.tolist() == [0, 1, 3]


def test_sample_respects_cap_and_keeps_outliers():
    rng = np.random.default_rng(0)
    dense = rng.normal(0, 0.01, size=(10_000, 2))
    outlier = np.array([[50.0, 50.0]])
    pts = np.vstack([dense, outlier]).astype(np.float32)
    idx = stratified_sample(pts[:, 0], pts[:, 1], cap=500, priority=priority_permutation(len(pts), 7))
    assert len(idx) == 500
    assert len(set(idx.tolist())) == 500
    assert 10_000 in idx  # the isolated point survives thinning


def test_sample_is_deterministic():
    rng = np.random.default_rng(1)
    pts = rng.random((5000, 2)).astype(np.float32)
    prio = priority_permutation(5000, 42)
    a = stratified_sample(pts[:, 0], pts[:, 1], cap=300, priority=prio)
    b = stratified_sample(pts[:, 0], pts[:, 1], cap=300, priority=prio)
    assert a.tolist() == b.tolist()


def test_density_grid_counts_every_point():
    x = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    grid = density_grid(x, y, (0.0, 1.0, 0.0, 1.0), 4, 4)
    assert grid.sum() == 3
    assert grid[0, 0] == 1 and grid[3, 3] == 2


def test_write_load_and_prune(cache_dir):
    v1 = storage.write_layer("base", "ss1", {"a": np.arange(3)}, {"hello": "world"})
    pointer = storage.read_pointer("base", "ss1")
    assert pointer.version == v1
    layer = storage.load_layer("base", "ss1", v1)
    assert layer["a"].tolist() == [0, 1, 2]
    assert layer.meta["hello"] == "world"

    storage.write_layer("base", "ss1", {"a": np.arange(4)})
    v3 = storage.write_layer("base", "ss1", {"a": np.arange(5)})
    versions = [p.name for p in (cache_dir / "base" / "ss1").iterdir() if p.is_dir()]
    assert len(versions) == storage.KEEP_VERSIONS and v3 in versions and v1 not in versions
    assert json.loads((cache_dir / "base" / "ss1" / "current.json").read_text())["version"] == v3


def test_invalidate_and_invalid_array_name(cache_dir):
    storage.write_layer("model", "ckpt1_ss1", {"a": np.zeros(1)})
    storage.invalidate_layer("model", "ckpt1_ss1")
    assert storage.read_pointer("model", "ckpt1_ss1") is None
    storage.invalidate_layer("model", "never-existed")  # no error
    with pytest.raises(ValueError):
        storage.write_layer("model", "bad", {"../x": np.zeros(1)})
    assert not any(p.name.startswith(".tmp-") for p in (cache_dir / "model" / "bad").iterdir())


def test_prune_model_layers_keeps_newest_checkpoints(cache_dir):
    from app.services.explore.hooks import prune_model_layers

    for ckpt in (1, 2, 3, 4, 5):
        storage.write_layer("model", f"ckpt{ckpt}_ss7", {"a": np.zeros(1)})
    storage.write_layer("model", "ckpt1_ss8", {"a": np.zeros(1)})
    prune_model_layers(7, keep_checkpoint_id=2, keep=2)
    remaining = sorted(p.name for p in (cache_dir / "model").iterdir())
    assert remaining == ["ckpt1_ss8", "ckpt2_ss7", "ckpt4_ss7", "ckpt5_ss7"]
