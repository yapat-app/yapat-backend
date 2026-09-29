"""
On-disk cache for the HNSW index used by acquisition scoring.

The index is built over **every** snippet in a snippet set, labelled or not.
That is the point of the cache: an index over only the unlabelled points would
change membership every time someone annotates, and faiss's ``IndexHNSWFlat``
cannot remove points, so it would have to be rebuilt. Over all points it
depends solely on the embeddings, which never change once computed -- so it is
built once and reused for the life of the snippet set.

Callers that need "nearest unlabelled neighbours" search for a few more than
they need and drop the labelled hits (see ``_scoring.py``). With ~0.1% of a
snippet set labelled, a small margin covers it comfortably.

Measured before this cache existed (Nova, 3,000,000 snippets): building the
index cost ~173 s on every retrain, inside the 2 min 54 s diversity stage.

Mirrors _embedding_cache.py: same fingerprint, same lock, same atomic publish.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any

import faiss
import numpy as np

from active_learning.config import HNSW_EF_SEARCH, HNSW_NEIGHBORS
from app.services.pam_al._embedding_cache import (
    _acquire_build_lock,
    _release_build_lock,
    get_embedding_cache_root,
)

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_ADD_CHUNK_SIZE = 100_000

# One index only: at 1024 dims over 3M points it is ~14 GB, so a second entry
# would double the worker's footprint for no benefit.
_MEMORY_CACHE: dict[tuple[int, int], faiss.Index] = {}


def get_ann_index_root() -> str:
    return os.path.join(os.path.dirname(get_embedding_cache_root()), "ann_index")


def get_index_dir(snippet_set_id: int, embedding_model_id: int) -> str:
    return os.path.join(get_ann_index_root(), str(snippet_set_id), str(embedding_model_id))


def _index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "index.faiss")


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.json")


def normalize_rows(chunk: np.ndarray) -> np.ndarray:
    """L2-normalise rows, matching torch.nn.functional.normalize(p=2, dim=1).

    Queries must be normalised exactly as the indexed vectors were, or the
    distances are meaningless. Both sides go through this function.
    """
    arr = np.asarray(chunk, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _expected_meta(fingerprint: dict[str, int]) -> dict[str, Any]:
    return {
        "version": _CACHE_VERSION,
        "fingerprint": {
            "count": int(fingerprint["count"]),
            "max_vector_id": int(fingerprint["max_vector_id"]),
            "dim": int(fingerprint["dim"]),
        },
        "hnsw_m": int(HNSW_NEIGHBORS),
        "ef_search": int(HNSW_EF_SEARCH),
    }


def _meta_matches(meta: dict[str, Any], fingerprint: dict[str, int]) -> bool:
    want = _expected_meta(fingerprint)
    if int(meta.get("version", -1)) != want["version"]:
        return False
    # The graph's shape depends on M; efSearch is a query-time knob but is
    # stored on the index, so a change to either invalidates the cache.
    if int(meta.get("hnsw_m", -1)) != want["hnsw_m"]:
        return False
    stored = meta.get("fingerprint") or {}
    return (
        int(stored.get("count", -1)) == want["fingerprint"]["count"]
        and int(stored.get("max_vector_id", -1)) == want["fingerprint"]["max_vector_id"]
        and int(stored.get("dim", -1)) == want["fingerprint"]["dim"]
        and want["fingerprint"]["count"] > 0
        and want["fingerprint"]["dim"] > 0
    )


def _cache_files_exist(cache_dir: str) -> bool:
    return os.path.isfile(_meta_path(cache_dir)) and os.path.isfile(_index_path(cache_dir))


def _read_meta(cache_dir: str) -> dict[str, Any]:
    with open(_meta_path(cache_dir), "r", encoding="utf-8") as fh:
        return json.load(fh)


def invalidate_ann_index_cache(snippet_set_id: int, embedding_model_id: int) -> None:
    cache_dir = get_index_dir(snippet_set_id, embedding_model_id)
    _MEMORY_CACHE.pop((snippet_set_id, embedding_model_id), None)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(
            "Invalidated ANN index cache snippet_set_id=%s embedding_model_id=%s",
            snippet_set_id,
            embedding_model_id,
        )


def _load_index(cache_dir: str) -> faiss.Index:
    index = faiss.read_index(_index_path(cache_dir))
    index.hnsw.efSearch = int(HNSW_EF_SEARCH)
    return index


def _build_index(X: np.ndarray, cache_dir: str, fingerprint: dict[str, int]) -> faiss.Index:
    """Build over all rows of X, normalising in chunks so a memmap stays a memmap."""
    n, dim = X.shape
    index = faiss.IndexHNSWFlat(dim, int(HNSW_NEIGHBORS), faiss.METRIC_L2)
    index.hnsw.efSearch = int(HNSW_EF_SEARCH)

    for start in range(0, n, _ADD_CHUNK_SIZE):
        stop = min(start + _ADD_CHUNK_SIZE, n)
        index.add(normalize_rows(X[start:stop]))
        logger.info("ann index: added %s/%s vectors", stop, n)

    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="annidx_", dir=os.path.dirname(cache_dir))
    try:
        faiss.write_index(index, _index_path(tmp_dir))
        with open(_meta_path(tmp_dir), "w", encoding="utf-8") as fh:
            json.dump(_expected_meta(fingerprint), fh)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(tmp_dir, cache_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logger.info("ann index: built and cached %s vectors dim=%s at %s", n, dim, cache_dir)
    return index


def build_transient_index(X: np.ndarray) -> faiss.Index:
    """An uncached index over all rows of X.

    Used when the feature space is not the stored embeddings -- the MLP
    classifier's ``extract_features`` returns hidden-layer activations, which
    change with every retrain, so there is nothing stable to cache.
    """
    n, dim = X.shape
    index = faiss.IndexHNSWFlat(dim, int(HNSW_NEIGHBORS), faiss.METRIC_L2)
    index.hnsw.efSearch = int(HNSW_EF_SEARCH)
    for start in range(0, n, _ADD_CHUNK_SIZE):
        index.add(normalize_rows(X[start : min(start + _ADD_CHUNK_SIZE, n)]))
    logger.info("ann index: built transient index over %s vectors (not cacheable)", n)
    return index


def load_or_build_index(
    X: np.ndarray,
    snippet_set_id: int,
    embedding_model_id: int,
    fingerprint: dict[str, int],
) -> faiss.Index:
    """Return the HNSW index over every row of X, building and caching on miss.

    ``X`` must be the snippet set's full embedding matrix in snippet_id order --
    the same ordering the caller uses for its row indices, since faiss returns
    positions in this matrix.
    """
    cache_key = (snippet_set_id, embedding_model_id)
    cached = _MEMORY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_dir = get_index_dir(snippet_set_id, embedding_model_id)

    if _cache_files_exist(cache_dir) and _meta_matches(_read_meta(cache_dir), fingerprint):
        index = _load_index(cache_dir)
        logger.info(
            "ann index: loaded from cache snippet_set_id=%s embedding_model_id=%s vectors=%s",
            snippet_set_id, embedding_model_id, index.ntotal,
        )
        _MEMORY_CACHE.clear()
        _MEMORY_CACHE[cache_key] = index
        return index

    lock_file = _acquire_build_lock(cache_dir)
    try:
        # Another process may have built it while we waited for the lock.
        if _cache_files_exist(cache_dir) and _meta_matches(_read_meta(cache_dir), fingerprint):
            index = _load_index(cache_dir)
        else:
            index = _build_index(X, cache_dir, fingerprint)
        _MEMORY_CACHE.clear()
        _MEMORY_CACHE[cache_key] = index
        return index
    finally:
        _release_build_lock(lock_file)
