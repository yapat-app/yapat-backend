"""
Retrains and cold starts wait for a running "add recordings" run on the same dataset.

Without this, a retrain triggered by feedback (or the manual button), or a
cold start from metadata, started while an
ingest run is embedding new snippets trains on the old embedding matrix, then
replaces the active checkpoint after the ingest scored the new snippets with
the old one, so the new snippets end up with no predictions for the new
checkpoint. Labelling is never blocked: the retrain job is created as usual
(PENDING) but not sent to Celery, and the run dispatches it when it ends
(completed, failed, or failed as dead).

No column links the two. A run can only start while no retrain is PENDING or
RUNNING on its dataset (409 retrain_job_running), so a retrain that is PENDING,
never started and created after the run started was held back by that run.
"Waiting" is computed the same way for the API (waiting_ingest_job_id).

A retrain job may therefore be dispatched twice in rare races (e.g. created the
instant the run ends, or re-sent by the cleanup safety net). The retrain tasks
claim their job atomically (claim_retrain_job), so a second copy exits.

The bootstrap cold start (first labels, trained synchronously inside the
feedback request) never goes through here: it stamps started_at as it starts,
so it never looks held.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from app.models.dataset_ingest_job import ACTIVE_INGEST_STATUSES, DatasetIngestJob
from app.models.pam_active_learning import ALRetrainJob, ALRetrainStatus

logger = logging.getLogger(__name__)

# Only these wait for a run; inference jobs never do.
WAITING_TRIGGERS = ("manual", "auto_feedback", "cold_start")

# Safety net: a never-started PENDING retrain older than this, with no run on
# its dataset, is re-dispatched by the cleanup beat (duplicates exit on claim).
UNDISPATCHED_AFTER = timedelta(minutes=10)


def _utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=timezone.utc)  # SQLite hands back naive UTC


def active_ingest_job(db: Session, dataset_id: int) -> Optional[DatasetIngestJob]:
    return (
        db.query(DatasetIngestJob)
        .filter(
            DatasetIngestJob.dataset_id == dataset_id,
            DatasetIngestJob.status.in_(ACTIVE_INGEST_STATUSES),
        )
        .order_by(DatasetIngestJob.created_at.desc())
        .first()
    )


def _held_query(db: Session, dataset_id: int):
    return db.query(ALRetrainJob).filter(
        ALRetrainJob.dataset_id == dataset_id,
        ALRetrainJob.trigger.in_(WAITING_TRIGGERS),
        ALRetrainJob.status == ALRetrainStatus.PENDING,
        ALRetrainJob.started_at.is_(None),
    )


def _dispatch(job: ALRetrainJob) -> None:
    from app.tasks.pam_al_tasks import (
        pam_al_auto_retrain,
        pam_al_manual_retrain,
        pam_al_train_from_scratch,
    )

    task = {
        "manual": pam_al_manual_retrain,
        "cold_start": pam_al_train_from_scratch,
    }.get(job.trigger, pam_al_auto_retrain)
    task.delay(checkpoint_id=job.model_checkpoint_id, job_id=job.id)


def dispatch_or_wait(db: Session, job: ALRetrainJob) -> Optional[int]:
    """Dispatch a freshly created retrain job, or leave it PENDING behind a running ingest.

    Returns the ingest job id it now waits for, or None when it was dispatched.
    """
    run = active_ingest_job(db, job.dataset_id)
    if run is None:
        _dispatch(job)
        return None

    # The run may end between the check above and its own release pass having
    # looked for held retrains; check again so the job never waits for nothing.
    db.expire_all()
    if active_ingest_job(db, job.dataset_id) is None:
        _dispatch(job)
        return None
    logger.info("retrain job %s waits for ingest job %s", job.id, run.id)
    return run.id


def waiting_ingest_job_id(db: Session, job: ALRetrainJob) -> Optional[int]:
    """The run this retrain job is waiting for, or None (computed, nothing stored)."""
    if (
        job.trigger not in WAITING_TRIGGERS
        or job.status != ALRetrainStatus.PENDING
        or job.started_at is not None
    ):
        return None
    run = active_ingest_job(db, job.dataset_id)
    if run is None or _utc(job.created_at) < _utc(run.created_at):
        return None
    return run.id


def release_waiting_retrains(db: Session, ingest_job_ids: Iterable[int]) -> list[int]:
    """Dispatch the retrains these (now finished) runs held back. Returns their ids."""
    released: list[int] = []
    for ingest_id in {int(i) for i in ingest_job_ids if i is not None}:
        run = db.get(DatasetIngestJob, ingest_id)
        if run is None:
            continue
        held = (
            _held_query(db, run.dataset_id)
            .filter(ALRetrainJob.created_at >= run.created_at)
            .order_by(ALRetrainJob.id)
            .all()
        )
        for job in held:
            try:
                _dispatch(job)
                released.append(job.id)
                logger.info("released retrain job %s after ingest job %s ended", job.id, ingest_id)
            except Exception:
                logger.exception("could not dispatch released retrain job %s", job.id)
    return released


def release_orphaned_waiting_retrains(db: Session) -> list[int]:
    """Safety net (cleanup beat): re-send never-started retrains no run is holding.

    Covers a release that was lost (worker died between a run's end and its
    release pass). Re-sending a job that is merely queued in Celery is harmless:
    whichever copy starts first claims it, the other exits.
    """
    cutoff = datetime.now(timezone.utc) - UNDISPATCHED_AFTER
    candidates = (
        db.query(ALRetrainJob)
        .filter(
            ALRetrainJob.trigger.in_(WAITING_TRIGGERS),
            ALRetrainJob.status == ALRetrainStatus.PENDING,
            ALRetrainJob.started_at.is_(None),
        )
        .all()
    )
    released: list[int] = []
    for job in candidates:
        if _utc(job.created_at) > cutoff or active_ingest_job(db, job.dataset_id) is not None:
            continue
        try:
            _dispatch(job)
            released.append(job.id)
        except Exception:
            logger.exception("could not re-dispatch retrain job %s", job.id)
    return released


def claim_retrain_job(db: Session, job_id: int) -> bool:
    """Atomically move a retrain job PENDING -> RUNNING. False when another copy got it.

    Called first by the retrain tasks, so a job sent twice runs once.
    """
    claimed = (
        db.query(ALRetrainJob)
        .filter(ALRetrainJob.id == job_id, ALRetrainJob.status == ALRetrainStatus.PENDING)
        .update(
            {ALRetrainJob.status: ALRetrainStatus.RUNNING, ALRetrainJob.started_at: datetime.now(timezone.utc)},
            synchronize_session=False,
        )
    )
    db.commit()
    return bool(claimed)
