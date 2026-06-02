"""
Redis-backed cache for dataset-level Feature Projection (FPV) responses.

The dataset-level FPV projection for a given (dataset, embedding_model, method,
run_3d) is **static** — it only changes when projections are regenerated. Yet
without caching it is recomputed (load ~130k rows, build ~130k objects, serialize)
on every page load by every user, taking 15-25s each and starving other requests
(the work is CPU-bound and holds the GIL).

This module caches the fully-serialized JSON bytes so the first request warms the
cache and every subsequent request (any user) is served in milliseconds. Cache
entries are invalidated whenever projections are regenerated.

All operations fail soft: if Redis is unavailable the caller simply computes the
response as before.
"""

from __future__ import annotations

import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# 7 days. This is only a safety net — entries are explicitly invalidated on
# regeneration, so staleness should never occur in practice.
FPV_CACHE_TTL_SECONDS = 7 * 24 * 3600

_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("fpv cache: could not create redis client: %s", exc)
            return None
    return _client


def _key(dataset_id: int, embedding_model_id: int, method: str, run_3d: bool) -> str:
    return f"fpv_ds:{dataset_id}:{embedding_model_id}:{method}:{int(run_3d)}"


def get_cached_fpv(
    dataset_id: int, embedding_model_id: int, method: str, run_3d: bool
) -> bytes | None:
    """Return cached serialized JSON bytes, or None on miss / error."""
    client = _redis()
    if client is None:
        return None
    try:
        return client.get(_key(dataset_id, embedding_model_id, method, run_3d))
    except Exception as exc:
        logger.warning("fpv cache: get failed: %s", exc)
        return None


def set_cached_fpv(
    dataset_id: int,
    embedding_model_id: int,
    method: str,
    run_3d: bool,
    payload: bytes,
) -> None:
    """Store serialized JSON bytes for this projection."""
    client = _redis()
    if client is None:
        return
    try:
        client.set(
            _key(dataset_id, embedding_model_id, method, run_3d),
            payload,
            ex=FPV_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("fpv cache: set failed: %s", exc)


def invalidate_fpv(dataset_id: int, embedding_model_id: int) -> None:
    """Drop all cached projection variants for a (dataset, embedding_model) pair."""
    client = _redis()
    if client is None:
        return
    try:
        pattern = f"fpv_ds:{dataset_id}:{embedding_model_id}:*"
        keys = list(client.scan_iter(match=pattern, count=200))
        if keys:
            client.delete(*keys)
            logger.info(
                "fpv cache: invalidated %d keys for dataset_id=%s embedding_model_id=%s",
                len(keys), dataset_id, embedding_model_id,
            )
    except Exception as exc:
        logger.warning("fpv cache: invalidate failed: %s", exc)
