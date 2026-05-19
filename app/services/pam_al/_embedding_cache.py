"""
On-disk cache for PAM AL embedding matrices.

Caches (X, snippet_rows) per (snippet_set_id, embedding_model_id) to avoid
re-loading large pgvector payloads from Postgres on every train/inference pass.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.models.embedding import EmbeddingVector
from app.models.recording import Recording
from app.models.snippet import Snippet

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_CACHE_CHUNK_SIZE = 4096

# Optional in-process cache for repeated access within one worker process.
_MEMORY_CACHE: dict[tuple[int, int], tuple[np.ndarray, List[Dict[str, Any]]]] = {}
_MEMORY_CACHE_MAX_ENTRIES = 2


def get_embedding_cache_root() -> str:
    checkpoints_dir = settings.PAM_CHECKPOINTS_DIR or "models_AL/pam/checkpoints"
    return os.path.join(os.path.dirname(checkpoints_dir), "embedding_cache")


def get_cache_dir(snippet_set_id: int, embedding_model_id: int) -> str:
    return os.path.join(
        get_embedding_cache_root(),
        str(snippet_set_id),
        str(embedding_model_id),
    )


def compute_embedding_fingerprint(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
) -> dict[str, int]:
    count, max_vector_id, dim = (
        db.query(
            func.count(EmbeddingVector.id),
            func.max(EmbeddingVector.id),
            func.max(EmbeddingVector.dim),
        )
        .join(Snippet, Snippet.id == EmbeddingVector.snippet_id)
        .filter(Snippet.snippet_set_id == snippet_set_id)
        .filter(EmbeddingVector.embedding_model_id == embedding_model_id)
        .one()
    )
    return {
        "count": int(count or 0),
        "max_vector_id": int(max_vector_id or 0),
        "dim": int(dim or 0),
    }


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.json")


def _snippets_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "snippets.json")


def _embeddings_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "embeddings.npy")


def _fingerprint_matches(meta: dict[str, Any], fingerprint: dict[str, int]) -> bool:
    stored = meta.get("fingerprint") or {}
    return (
        int(stored.get("count", -1)) == fingerprint["count"]
        and int(stored.get("max_vector_id", -1)) == fingerprint["max_vector_id"]
        and int(stored.get("dim", -1)) == fingerprint["dim"]
        and fingerprint["count"] > 0
        and fingerprint["dim"] > 0
    )


def _cache_files_exist(cache_dir: str) -> bool:
    return (
        os.path.isfile(_meta_path(cache_dir))
        and os.path.isfile(_snippets_path(cache_dir))
        and os.path.isfile(_embeddings_path(cache_dir))
    )


def invalidate_embedding_cache(snippet_set_id: int, embedding_model_id: int) -> None:
    cache_dir = get_cache_dir(snippet_set_id, embedding_model_id)
    _MEMORY_CACHE.pop((snippet_set_id, embedding_model_id), None)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(
            "Invalidated embedding cache snippet_set_id=%s embedding_model_id=%s",
            snippet_set_id,
            embedding_model_id,
        )


def _load_from_cache_dir(cache_dir: str) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    with open(_meta_path(cache_dir), "r", encoding="utf-8") as f:
        meta = json.load(f)

    X = np.load(_embeddings_path(cache_dir), mmap_mode="r")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    with open(_snippets_path(cache_dir), "r", encoding="utf-8") as f:
        snippet_rows = json.load(f)

    expected_count = int(meta.get("fingerprint", {}).get("count", 0))
    if expected_count != X.shape[0] or expected_count != len(snippet_rows):
        raise ValueError(
            f"Embedding cache metadata mismatch in {cache_dir}: "
            f"meta.count={expected_count}, X.rows={X.shape[0]}, snippets={len(snippet_rows)}"
        )

    return X, snippet_rows


def _vector_to_float32(vector: Any) -> np.ndarray:
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32, copy=False)
    return np.asarray(vector, dtype=np.float32)


def _build_cache_from_db(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
    cache_dir: str,
    fingerprint: dict[str, int],
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    count = fingerprint["count"]
    dim = fingerprint["dim"]

    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    tmp_dir = tempfile.mkdtemp(
        prefix=f"build_{snippet_set_id}_{embedding_model_id}_",
        dir=os.path.dirname(cache_dir),
    )

    try:
        embeddings_path = _embeddings_path(tmp_dir)
        X = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, dim),
        )

        snippet_rows: List[Dict[str, Any]] = []
        offset = 0

        query = (
            db.query(
                Snippet.id,
                Snippet.recording_id,
                Snippet.start_time,
                Snippet.end_time,
                Recording.file_name,
                Recording.file_path,
                EmbeddingVector.vector,
            )
            .join(Recording, Snippet.recording_id == Recording.id)
            .join(EmbeddingVector, Snippet.id == EmbeddingVector.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .filter(EmbeddingVector.embedding_model_id == embedding_model_id)
            .order_by(Snippet.id)
            .yield_per(_CACHE_CHUNK_SIZE)
        )

        for row in query:
            X[offset] = _vector_to_float32(row[6])
            snippet_rows.append(
                {
                    "snippet_id": row[0],
                    "recording_id": row[1],
                    "start_time": float(row[2]),
                    "end_time": float(row[3]),
                    "file_name": row[4],
                    "file_path": row[5],
                }
            )
            offset += 1

        if offset != count:
            raise ValueError(
                f"Embedding cache build row count mismatch: expected={count}, wrote={offset}"
            )

        X.flush()
        del X

        meta = {
            "version": _CACHE_VERSION,
            "snippet_set_id": snippet_set_id,
            "embedding_model_id": embedding_model_id,
            "fingerprint": fingerprint,
            "dtype": "float32",
        }
        with open(_meta_path(tmp_dir), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(_snippets_path(tmp_dir), "w", encoding="utf-8") as f:
            json.dump(snippet_rows, f)

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(tmp_dir, cache_dir)

        logger.info(
            "Built embedding cache snippet_set_id=%s embedding_model_id=%s rows=%s dim=%s path=%s",
            snippet_set_id,
            embedding_model_id,
            count,
            dim,
            cache_dir,
        )
        return _load_from_cache_dir(cache_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _acquire_build_lock(cache_dir: str):
    try:
        import fcntl
    except ImportError:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    lock_path = os.path.join(cache_dir, ".building.lock")
    lock_file = open(lock_path, "w", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _release_build_lock(lock_file) -> None:
    if lock_file is None:
        return
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def load_embeddings_cached(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load embeddings from cache when valid; otherwise build the cache from Postgres.
    """
    cache_key = (snippet_set_id, embedding_model_id)
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    fingerprint = compute_embedding_fingerprint(db, snippet_set_id, embedding_model_id)
    if fingerprint["count"] == 0:
        raise ValueError(
            f"No embeddings found for snippet_set_id={snippet_set_id}, "
            f"embedding_model_id={embedding_model_id}"
        )

    cache_dir = get_cache_dir(snippet_set_id, embedding_model_id)

    if _cache_files_exist(cache_dir):
        with open(_meta_path(cache_dir), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if _fingerprint_matches(meta, fingerprint):
            logger.info(
                "Loaded embedding cache snippet_set_id=%s embedding_model_id=%s rows=%s",
                snippet_set_id,
                embedding_model_id,
                fingerprint["count"],
            )
            result = _load_from_cache_dir(cache_dir)
            if len(_MEMORY_CACHE) >= _MEMORY_CACHE_MAX_ENTRIES:
                _MEMORY_CACHE.clear()
            _MEMORY_CACHE[cache_key] = result
            return result

    lock_file = _acquire_build_lock(cache_dir)
    try:
        if _cache_files_exist(cache_dir):
            with open(_meta_path(cache_dir), "r", encoding="utf-8") as f:
                meta = json.load(f)
            if _fingerprint_matches(meta, fingerprint):
                result = _load_from_cache_dir(cache_dir)
                if len(_MEMORY_CACHE) >= _MEMORY_CACHE_MAX_ENTRIES:
                    _MEMORY_CACHE.clear()
                _MEMORY_CACHE[cache_key] = result
                return result

        result = _build_cache_from_db(
            db,
            snippet_set_id,
            embedding_model_id,
            cache_dir,
            fingerprint,
        )
        if len(_MEMORY_CACHE) >= _MEMORY_CACHE_MAX_ENTRIES:
            _MEMORY_CACHE.clear()
        _MEMORY_CACHE[cache_key] = result
        return result
    finally:
        _release_build_lock(lock_file)
