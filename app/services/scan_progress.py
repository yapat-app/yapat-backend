"""
Redis-backed progress for dataset scans.

A scan (``scan_dataset`` task -> ``DatasetService.scan_recordings``) has no row
of its own in Postgres, unlike an embedding job, so nothing in the database can
tell a page refresh that a scan is running. Progress therefore lives in one small
Redis hash per dataset, which the dataset card polls and ``GET /datasets`` reads
in a single round trip for the whole page.

Hash ``scan:ds:{dataset_id}`` fields:
  stage    queued | listing | reading | saving | completed | failed
  total    files to process in the current stage (new files only on a rescan)
  done     files processed so far in the current stage
  ts       unix time of the last update (heartbeat for stall detection)
  error    failure message (stage == failed)
  created  recordings created by the finished scan (stage == completed)

All Redis calls fail soft: an unavailable Redis never breaks a scan, it only
hides the progress numbers (the card then falls back to "Processing dataset").
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Optional

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# The hash outlives the scan by a day so a late poll still sees the final state,
# and a failure stays visible on the card after a refresh. Every update renews it.
PROGRESS_TTL_SECONDS = 24 * 3600
# No update for this long in a working stage => report the scan as stalled
# (e.g. the worker was killed by the hard time limit or a restart).
STALL_AFTER_SECONDS = 10 * 60
# Reporter throttle: write at most every this many seconds or files, so hashing
# thousands of small files does not turn into thousands of Redis round trips.
REPORT_MIN_INTERVAL_SECONDS = 2.0
REPORT_MIN_STEP = 25
# Keep stored error messages short; full tracebacks belong in the worker log.
MAX_ERROR_LENGTH = 500

WORKING_STAGES = ("listing", "reading", "saving")
# Stages surfaced on the dataset list. Failed is included so a refresh shows the
# error instead of an endless "Processing dataset"; completed is not, because the
# recording count already tells the card the scan is done.
ACTIVE_STAGES = ("queued", *WORKING_STAGES, "failed")

# Share of the overall progress bar given to each stage. Reading (header +
# checksum per file) is by far the slowest part; saving is batched inserts.
_READING_SHARE = 95.0

_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("scan progress: could not create redis client: %s", exc)
            return None
    return _client


def _key(dataset_id: int) -> str:
    return f"scan:ds:{dataset_id}"


def _write(dataset_id: int, fields: dict[str, Any], *, reset: bool = False) -> None:
    client = _redis()
    if client is None:
        return
    key = _key(dataset_id)
    mapping = {k: v for k, v in fields.items() if v is not None}
    mapping["ts"] = int(time.time())
    try:
        pipe = client.pipeline()
        if reset:
            # A new scan must not inherit the previous scan's totals or error.
            pipe.delete(key)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, PROGRESS_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("scan progress: write failed for dataset %s: %s", dataset_id, exc)


def mark_queued(dataset_id: int) -> None:
    """Called by the API right before dispatching a scan. Must run before the task is
    queued: a fast worker could otherwise reach "listing" first and be overwritten."""
    _write(dataset_id, {"stage": "queued"}, reset=True)


def update(dataset_id: int, stage: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
    _write(dataset_id, {"stage": stage, "done": done, "total": total})


def mark_completed(dataset_id: int, created: int) -> None:
    _write(dataset_id, {"stage": "completed", "created": int(created)})


def mark_failed(dataset_id: int, error: str) -> None:
    _write(dataset_id, {"stage": "failed", "error": (error or "Scan failed")[:MAX_ERROR_LENGTH]})


def make_reporter(
    dataset_id: int,
    *,
    min_interval: float = REPORT_MIN_INTERVAL_SECONDS,
    min_step: int = REPORT_MIN_STEP,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[str, int, int], None]:
    """Return a ``progress(stage, done, total)`` callback for ``scan_recordings``.

    Always writes on a stage change and at the end of a stage; in between only
    when ``min_interval`` seconds or ``min_step`` files have passed."""
    last = {"stage": None, "done": 0, "t": 0.0}

    def report(stage: str, done: int, total: int) -> None:
        now = clock()
        if (
            stage == last["stage"]
            and done < total
            and done - last["done"] < min_step
            and now - last["t"] < min_interval
        ):
            return
        update(dataset_id, stage, done=done, total=total)
        last.update(stage=stage, done=done, t=now)

    return report


def _decode(raw: dict) -> dict[str, str]:
    return {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def build_progress(dataset_id: int, raw: dict, now: Optional[float] = None) -> dict[str, Any]:
    """Turn a stored hash (possibly empty) into the API payload. Pure, for testing."""
    fields = _decode(raw or {})
    now = time.time() if now is None else now
    stage = fields.get("stage") or "idle"
    done = _int(fields.get("done"))
    total = _int(fields.get("total"))
    last_update = _int(fields.get("ts"))

    percent: Optional[float] = None
    if stage == "completed":
        percent = 100.0
    elif stage in ("queued", "listing"):
        percent = 0.0
    elif stage in ("reading", "saving") and total:
        fraction = min(1.0, (done or 0) / total)
        if stage == "reading":
            percent = round(_READING_SHARE * fraction, 1)
        else:
            percent = round(_READING_SHARE + (100.0 - _READING_SHARE) * fraction, 1)

    stalled = (
        stage in WORKING_STAGES
        and last_update is not None
        and now - last_update > STALL_AFTER_SECONDS
    )

    return {
        "dataset_id": dataset_id,
        "stage": stage,
        "done": done,
        "total": total,
        "percent": percent,
        "stalled": stalled,
        "last_update": last_update,
        "error_message": fields.get("error"),
        "recordings_created": _int(fields.get("created")),
    }


def get_progress(dataset_id: int) -> dict[str, Any]:
    """Payload for GET /datasets/{id}/scan-progress; stage "idle" when unknown."""
    raw: dict = {}
    client = _redis()
    if client is not None:
        try:
            raw = client.hgetall(_key(dataset_id))
        except Exception as exc:
            logger.warning("scan progress: read failed for dataset %s: %s", dataset_id, exc)
    return build_progress(dataset_id, raw)


def get_active_scans(dataset_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Scans to show on the dataset list (running, queued or failed), one Redis
    round trip for the whole page. Datasets without one are left out."""
    ids = list(dataset_ids)
    client = _redis()
    if client is None or not ids:
        return {}
    try:
        pipe = client.pipeline()
        for dataset_id in ids:
            pipe.hgetall(_key(dataset_id))
        raws = pipe.execute()
    except Exception as exc:
        logger.warning("scan progress: bulk read failed: %s", exc)
        return {}

    now = time.time()
    active: dict[int, dict[str, Any]] = {}
    for dataset_id, raw in zip(ids, raws):
        if not raw:
            continue
        progress = build_progress(dataset_id, raw, now)
        if progress["stage"] in ACTIVE_STAGES:
            active[dataset_id] = progress
    return active
