"""
Redis-backed progress counter for dataset ingest ("add recordings") jobs.

Same idea as app/services/embedding_progress.py: the pipeline fans out into
Celery tasks, so each task INCRBYs a per-job counter and pollers read it with one
MGET, independent of dataset size. The counter tracks the *current stage* only;
the stage itself lives in Postgres (dataset_ingest_jobs.stage). Every stage
change resets the counter via start_stage().

Keys (per job):
  ingest:job:{id}:total  units of work in the current stage
  ingest:job:{id}:done   units finished so far in the current stage
  ingest:job:{id}:ts     unix time of the last update (heartbeat for stall detection)

All Redis operations fail soft: an unavailable Redis never breaks the pipeline, it
only hides the numbers.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import redis

from app.config import settings

logger = logging.getLogger(__name__)

PROGRESS_TTL_SECONDS = 24 * 3600
STALL_AFTER_SECONDS = 10 * 60

_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ingest progress: could not create redis client: %s", exc)
            return None
    return _client


def _keys(job_id: int) -> tuple[str, str, str]:
    base = f"ingest:job:{job_id}"
    return f"{base}:total", f"{base}:done", f"{base}:ts"


def start_stage(job_id: int, total: int = 0) -> None:
    """Reset the counter for a new stage. total=0 means the stage has no count."""
    client = _redis()
    if client is None:
        return
    k_total, k_done, k_ts = _keys(job_id)
    try:
        pipe = client.pipeline()
        pipe.set(k_total, int(total), ex=PROGRESS_TTL_SECONDS)
        pipe.set(k_done, 0, ex=PROGRESS_TTL_SECONDS)
        pipe.set(k_ts, int(time.time()), ex=PROGRESS_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("ingest progress: start_stage failed for job %s: %s", job_id, exc)


def incr(job_id: int, n: int) -> None:
    client = _redis()
    if client is None or n <= 0:
        return
    k_total, k_done, k_ts = _keys(job_id)
    try:
        pipe = client.pipeline()
        pipe.incrby(k_done, int(n))
        pipe.set(k_ts, int(time.time()), ex=PROGRESS_TTL_SECONDS)
        pipe.expire(k_done, PROGRESS_TTL_SECONDS)
        pipe.expire(k_total, PROGRESS_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("ingest progress: incr failed for job %s: %s", job_id, exc)


def heartbeat(job_id: int) -> None:
    """Mark the job alive during a stage step that has no countable progress."""
    client = _redis()
    if client is None:
        return
    try:
        client.set(_keys(job_id)[2], int(time.time()), ex=PROGRESS_TTL_SECONDS)
    except Exception as exc:
        logger.warning("ingest progress: heartbeat failed for job %s: %s", job_id, exc)


def read(job_id: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(total, done, last_update), each None when unknown."""
    client = _redis()
    if client is None:
        return None, None, None
    try:
        raw = client.mget(*_keys(job_id))
        return tuple(int(v) if v is not None else None for v in raw)  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("ingest progress: read failed for job %s: %s", job_id, exc)
        return None, None, None


def is_stalled(last_update: Optional[int]) -> bool:
    return last_update is not None and time.time() - last_update > STALL_AFTER_SECONDS


def seconds_since_epoch(unix_ts: int) -> float:
    return time.time() - unix_ts


def seconds_since(moment) -> Optional[float]:
    """Age of a DB timestamp in seconds (naive values, as SQLite returns them, are UTC)."""
    if moment is None:
        return None
    from datetime import timezone

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return time.time() - moment.timestamp()
