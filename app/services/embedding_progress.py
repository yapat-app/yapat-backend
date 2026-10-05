"""
Redis-backed progress counter for embedding jobs.

An embedding job fans out into a Celery chord of per-recording tasks, so no single
Celery task knows the overall progress. Instead every recording task atomically
INCRBYs a per-job counter, and `run_embedding` stores the total once segmentation
is done. Reading progress is then two Redis GETs, independent of dataset size, so
the frontend can poll it cheaply while a job runs.

Keys (per job):
  emb:job:{id}:total  snippets to embed (set once by run_embedding)
  emb:job:{id}:done   snippets processed so far (INCRBY per recording)
  emb:job:{id}:ts     unix time of the last increment (heartbeat for stall detection)

If the keys are missing (Redis flushed/restarted mid-job) `get_progress` rebuilds
them once from Postgres, guarded by a SETNX lock so concurrent pollers don't all
run the COUNT. All Redis operations fail soft: an unavailable Redis never breaks
the embedding pipeline, it only hides the progress numbers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import redis
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, EmbeddingVector
from app.models.snippet import Snippet

logger = logging.getLogger(__name__)

# Keys outlive the job by a day so a last poll after completion still has numbers.
# While running they are refreshed on every increment, so they never expire mid-job.
PROGRESS_TTL_SECONDS = 24 * 3600
# No increment for this long while RUNNING => report the job as stalled
# (e.g. the worker was killed and the chord will never finish).
STALL_AFTER_SECONDS = 10 * 60
# Minimum gap between Postgres recounts for one job (see _recount).
RECOUNT_INTERVAL_SECONDS = 2 * 60

_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("embedding progress: could not create redis client: %s", exc)
            return None
    return _client


def _keys(job_id: int) -> tuple[str, str, str]:
    base = f"emb:job:{job_id}"
    return f"{base}:total", f"{base}:done", f"{base}:ts"


def mark_started(job_id: int) -> None:
    """Called by run_embedding before segmentation. Sets only the heartbeat, so a
    reader seeing `ts` without `total` knows the job is segmenting (not that the
    keys were lost) and skips the Postgres recount."""
    client = _redis()
    if client is None:
        return
    try:
        client.set(_keys(job_id)[2], int(time.time()), ex=PROGRESS_TTL_SECONDS)
    except Exception as exc:
        logger.warning("embedding progress: mark_started failed for job %s: %s", job_id, exc)


def init_progress(job_id: int, total: int, done: int = 0) -> None:
    """Called by run_embedding once the snippet total is known."""
    client = _redis()
    if client is None:
        return
    k_total, k_done, k_ts = _keys(job_id)
    try:
        pipe = client.pipeline()
        pipe.set(k_total, int(total), ex=PROGRESS_TTL_SECONDS)
        pipe.set(k_done, int(done), ex=PROGRESS_TTL_SECONDS)
        pipe.set(k_ts, int(time.time()), ex=PROGRESS_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("embedding progress: init failed for job %s: %s", job_id, exc)


def incr_progress(job_id: int, n: int) -> None:
    """Called by each recording task after its snippets are processed."""
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
        logger.warning("embedding progress: incr failed for job %s: %s", job_id, exc)


def _read(client: redis.Redis, job_id: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
    raw = client.mget(*_keys(job_id))
    return tuple(int(v) if v is not None else None for v in raw)  # type: ignore[return-value]


def _recount(db: Session, client: redis.Redis, job: EmbeddingJob) -> tuple[Optional[int], Optional[int]]:
    """Count total/done in Postgres. Throttled per job via a SETNX key that is left to
    expire, so at most one COUNT runs per RECOUNT_INTERVAL_SECONDS however many tabs
    poll. Returns (None, None) when throttled."""
    throttle_key = f"emb:job:{job.id}:recount"
    if not client.set(throttle_key, 1, nx=True, ex=RECOUNT_INTERVAL_SECONDS):
        return None, None
    total = (
        db.query(func.count(Snippet.id))
        .filter(Snippet.snippet_set_id == job.snippet_set_id)
        .scalar()
    ) or 0
    # One job per snippet set (create_embedding_job enforces it), so this job's
    # vectors are exactly the processed snippets.
    done = (
        db.query(func.count(EmbeddingVector.id))
        .filter(EmbeddingVector.embedding_job_id == job.id)
        .scalar()
    ) or 0
    # Release the read transaction right away (invariant: no idle-in-transaction).
    db.commit()
    return total, done


def get_progress(db: Session, job: EmbeddingJob) -> dict[str, Any]:
    """Progress payload for GET /embeddings/{job_id}/progress."""
    status = job.status.value
    total: Optional[int] = None
    done: Optional[int] = None
    last_update: Optional[int] = None

    client = _redis()
    if client is not None:
        try:
            total, done, last_update = _read(client, job.id)
            now = time.time()
            all_missing = total is None and done is None and last_update is None
            idle = (
                total is not None
                and last_update is not None
                and now - last_update > RECOUNT_INTERVAL_SECONDS
            )
            # Self-heal from Postgres when the keys were lost (Redis flush) or the
            # counter has gone quiet (lost increments, or a worker still running code
            # from before this counter existed). A heartbeat without a total means
            # "still segmenting" and is left alone.
            if job.status == EmbeddingJobStatus.RUNNING and (all_missing or idle):
                db_total, db_done = _recount(db, client, job)
                if db_total and (done is None or (db_done or 0) > done):
                    init_progress(job.id, db_total, db_done or 0)
                    total, done, last_update = db_total, db_done, int(now)
                # No change: keep the old heartbeat so stall detection can fire.
        except Exception as exc:
            logger.warning("embedding progress: read failed for job %s: %s", job.id, exc)

    if job.status == EmbeddingJobStatus.COMPLETED and total:
        done = total

    percent: Optional[float] = None
    if job.status == EmbeddingJobStatus.COMPLETED:
        percent = 100.0
    elif total:
        percent = round(min(100.0, 100.0 * (done or 0) / total), 1)

    stalled = (
        job.status == EmbeddingJobStatus.RUNNING
        and last_update is not None
        and time.time() - last_update > STALL_AFTER_SECONDS
    )

    return {
        "embedding_job_id": job.id,
        "dataset_id": job.dataset_id,
        "status": status,
        # "segmenting" while run_embedding is still creating snippets (no total yet).
        "stage": (
            "segmenting"
            if status in ("pending", "running") and total is None
            else "embedding"
            if status == "running"
            else status
        ),
        "done": done,
        "total": total,
        "percent": percent,
        "stalled": stalled,
        "last_update": last_update,
        "error_message": job.error_message,
    }
