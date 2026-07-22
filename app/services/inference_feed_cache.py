"""
Redis-backed cache for confidence-ranked feed results.

The confidence strategy in get_top_prediction_suggestions ranks predictions by
noisy-OR score. PostgreSQL builds the cold ranking from the JSON probabilities,
and this module caches the resulting compact (pred_id, snippet_id, score) list
so subsequent calls avoid repeating that scan and sort.

Invalidation: call invalidate_inference_feed() whenever run_and_store_inference
writes new predictions for a checkpoint.

All operations fail soft: if Redis is unavailable PostgreSQL rebuilds the
ranking for that request (or non-PostgreSQL environments use the Python path).
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

FEED_CACHE_TTL_SECONDS = 24 * 3600  # 24h; invalidated explicitly on new inference


_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        except Exception as exc:
            logger.warning("inference_feed_cache: could not create redis client: %s", exc)
            return None
    return _client


def _scope_hash(label_scope: list[str] | None) -> str:
    if not label_scope:
        return "all"
    return hashlib.md5("|".join(sorted(label_scope)).encode()).hexdigest()[:12]


def _key(checkpoint_id: int, snippet_set_id: int, label_scope: list[str] | None) -> str:
    return f"inf_feed_conf:{checkpoint_id}:{snippet_set_id}:{_scope_hash(label_scope)}"


def get_cached_confidence_ranking(
    checkpoint_id: int,
    snippet_set_id: int,
    label_scope: list[str] | None,
) -> list[tuple[int, int, float]] | None:
    """Return cached [(pred_id, snippet_id, score)] sorted desc, or None on miss."""
    client = _redis()
    if client is None:
        return None
    try:
        raw = client.get(_key(checkpoint_id, snippet_set_id, label_scope))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("inference_feed_cache: get failed: %s", exc)
        return None


def set_cached_confidence_ranking(
    checkpoint_id: int,
    snippet_set_id: int,
    label_scope: list[str] | None,
    ranked: list[tuple[int, int, float]],
) -> None:
    """Store [(pred_id, snippet_id, score)] sorted desc."""
    client = _redis()
    if client is None:
        return
    try:
        client.set(
            _key(checkpoint_id, snippet_set_id, label_scope),
            json.dumps(ranked),
            ex=FEED_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("inference_feed_cache: set failed: %s", exc)


def invalidate_inference_feed(checkpoint_id: int) -> None:
    """Drop all cached feed variants for a checkpoint (called after new inference)."""
    client = _redis()
    if client is None:
        return
    try:
        pattern = f"inf_feed_conf:{checkpoint_id}:*"
        keys = list(client.scan_iter(match=pattern, count=200))
        if keys:
            client.delete(*keys)
            logger.info(
                "inference_feed_cache: invalidated %d keys for checkpoint_id=%s",
                len(keys),
                checkpoint_id,
            )
    except Exception as exc:
        logger.warning("inference_feed_cache: invalidate failed: %s", exc)
