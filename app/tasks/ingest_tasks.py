"""
Celery tasks for "add recordings to an existing dataset" (yapat-backend#70).

Pipeline: scanning -> segmenting -> embedding -> finalizing -> scoring.
Job state lives in dataset_ingest_jobs; live counts in app/services/ingest_progress.

Only what is new is touched: the scan skips known file paths, segmentation only
covers recordings without snippets in the set, and embedding only covers snippets
without a vector. The snippet set stays READY throughout, so the Annotation Hub
keeps working while a run is in progress.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from celery import chord
from sqlalchemy import exists, select
from sqlalchemy.sql import func

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.models.dataset import Dataset
from app.models.dataset_ingest_job import DatasetIngestJob, IngestJobStatus, IngestMode, IngestStage
from app.models.embedding import EmbeddingVector, SnippetSet
from app.models.recording import Recording
from app.models.recording_embedding_failure import RecordingEmbeddingFailure
from app.models.snippet import Snippet
from app.services import ingest_progress
from app.services.dataset_service import DatasetService
from app.tasks.embedding_tasks import generate_embeddings_for_recording, segment_recordings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Job bookkeeping
# ----------------------------------------------------------------------

def _set_stage(db, job: DatasetIngestJob, stage: IngestStage, total: int = 0) -> None:
    job.stage = stage
    db.commit()
    ingest_progress.start_stage(job.id, total)


def _update_summary(db, job: DatasetIngestJob, **values) -> None:
    # Reassign rather than mutate: plain JSON columns don't track in-place changes.
    job.summary = {**(job.summary or {}), **values}
    db.commit()


def complete_job(job_id: int, error: Optional[str] = None, **summary) -> None:
    """Mark the run COMPLETED, merging `summary` values. Opens its own session.

    `error` is only used by the scoring stage: recordings and vectors are already
    in when scoring fails, so the run still completes, with the reason recorded.
    """
    db = SessionLocal()
    try:
        job = db.get(DatasetIngestJob, job_id)
        if job is None:
            return
        job.summary = {**(job.summary or {}), **summary}
        job.status = IngestJobStatus.COMPLETED
        job.stage = IngestStage.COMPLETED
        job.completed_at = func.now()
        if error:
            job.error_message = error
        db.commit()
        _release_waiting_retrains(db, job_id)
    finally:
        db.close()


def _mark_failed(job_id: int, message: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(DatasetIngestJob, job_id)
        if job is None:
            return
        job.status = IngestJobStatus.FAILED
        job.stage = IngestStage.FAILED
        job.error_message = message
        job.completed_at = func.now()
        db.commit()
        _release_waiting_retrains(db, job_id)
    finally:
        db.close()


def _release_waiting_retrains(db, job_id: int) -> None:
    """The run has ended: start the retrains that waited for it."""
    from app.services.retrain_deferral import release_waiting_retrains

    try:
        release_waiting_retrains(db, [job_id])
    except Exception:
        # The cleanup beat releases them later (release_orphaned_waiting_retrains).
        logger.exception("ingest job %s: releasing waiting retrains failed", job_id)


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

def _scan(db, job: DatasetIngestJob) -> None:
    """Register new audio files as recordings. Progress unit: files read."""
    _set_stage(db, job, IngestStage.SCANNING)
    dataset = db.get(Dataset, job.dataset_id)
    stats: Dict[str, int] = {}
    DatasetService(db).scan_recordings(
        dataset,
        stats=stats,
        on_files_to_read=lambda n: ingest_progress.start_stage(job.id, n),
        on_file_read=lambda: ingest_progress.incr(job.id, 1),
    )
    _update_summary(db, job, **stats)


def _segment(db, job: DatasetIngestJob) -> None:
    """Create snippets for recordings that have none in the set. Progress unit: recordings."""
    snippet_set = db.get(SnippetSet, job.snippet_set_id)
    has_snippets = exists().where(
        Snippet.recording_id == Recording.id,
        Snippet.snippet_set_id == snippet_set.id,
    )
    recordings = (
        db.query(Recording.id, Recording.duration)
        .filter(Recording.dataset_id == job.dataset_id, ~has_snippets)
        .all()
    )
    _set_stage(db, job, IngestStage.SEGMENTING, total=len(recordings))
    new_count = 0
    if recordings:
        _ids, new_count = segment_recordings(
            db, snippet_set, recordings, only_these_recordings=True
        )
        db.commit()
        ingest_progress.incr(job.id, len(recordings))
    _update_summary(db, job, snippets_new=new_count)


def _snippets_without_vectors(db, snippet_set_id: int, mode: IngestMode) -> Dict[int, List[int]]:
    """{recording_id: [snippet_id, ...]} for the snippets this run should embed.

    SCAN: every snippet in the set with no vector, i.e. this run's new snippets
    plus earlier failures, except recordings the user dismissed ("continue
    without them"), so a permanently broken file is not retried on every run.
    RETRY_FAILED: only the snippets of recorded failures, dismissed or not.
    """
    failed = select(RecordingEmbeddingFailure.recording_id).where(
        RecordingEmbeddingFailure.snippet_set_id == snippet_set_id
    )
    query = (
        db.query(Snippet.recording_id, Snippet.id)
        .outerjoin(EmbeddingVector, EmbeddingVector.snippet_id == Snippet.id)
        .filter(Snippet.snippet_set_id == snippet_set_id, EmbeddingVector.id.is_(None))
    )
    if mode == IngestMode.RETRY_FAILED:
        query = query.filter(Snippet.recording_id.in_(failed))
    else:
        dismissed = failed.where(RecordingEmbeddingFailure.dismissed_at.isnot(None))
        query = query.filter(Snippet.recording_id.notin_(dismissed))
    rows = query.order_by(Snippet.recording_id, Snippet.start_time).all()
    grouped: Dict[int, List[int]] = defaultdict(list)
    for recording_id, snippet_id in rows:
        grouped[recording_id].append(snippet_id)
    return dict(grouped)


def _dispatch_embedding(db, job: DatasetIngestJob) -> int:
    """Fan out per-recording embedding tasks; finalize_dataset_ingest runs after.

    Same chunked chord as run_embedding. Vectors are attached to the set's
    original EmbeddingJob (job.embedding_job_id). Progress unit: snippets.
    """
    snippet_set = db.get(SnippetSet, job.snippet_set_id)
    plan = _snippets_without_vectors(db, snippet_set.id, job.mode)
    total = sum(len(ids) for ids in plan.values())
    _set_stage(db, job, IngestStage.EMBEDDING, total=total)

    if not plan:
        finalize_dataset_ingest.delay([], job.id, 0)
        return 0

    chunked_group = generate_embeddings_for_recording.chunks(
        (
            (recording_id, snippet_ids, snippet_set.embedding_model_id, job.embedding_job_id, job.id)
            for recording_id, snippet_ids in plan.items()
        ),
        settings.EMBEDDING_CHORD_CHUNK_SIZE,
    ).group()
    chord(chunked_group)(finalize_dataset_ingest.s(job.id, total))
    return total


def _invalidate_embedding_cache(snippet_set_id: int, embedding_model_id: int) -> None:
    # Lazy: importing app.services.pam_al loads the AL service (torch).
    from app.services.pam_al._embedding_cache import invalidate_embedding_cache

    invalidate_embedding_cache(snippet_set_id, embedding_model_id)


def _record_embedding_failures(db, job: DatasetIngestJob, results: List[dict]) -> Dict[str, int]:
    """Upsert one failure row per recording left with missing snippets; clear fixed ones.

    `results` are the per-recording returns of generate_embeddings_for_recording:
    status "error" means none of its snippets were embedded (`snippets_planned`),
    "success" may still carry `failed_snippets` that BirdNET returned nothing for.
    Returns this run's totals for the job summary.
    """
    totals = {"recordings": 0, "snippets": 0}
    for r in results:
        recording_id = r.get("recording_id")
        if recording_id is None:
            continue
        if r.get("status") == "success":
            missing = int(r.get("failed_snippets", 0) or 0)
            processed = int(r.get("snippets_processed", 0) or 0)
            reason = f"BirdNET returned no embedding for {missing} of {processed} snippets"
        else:
            missing = int(r.get("snippets_planned", 0) or 0)
            reason = str(r.get("message") or "Unknown error")

        row = (
            db.query(RecordingEmbeddingFailure)
            .filter_by(recording_id=recording_id, snippet_set_id=job.snippet_set_id)
            .one_or_none()
        )
        if missing <= 0:
            if row is not None:
                db.delete(row)
            continue

        totals["recordings"] += 1
        totals["snippets"] += missing
        if row is None:
            row = RecordingEmbeddingFailure(
                dataset_id=job.dataset_id,
                snippet_set_id=job.snippet_set_id,
                recording_id=recording_id,
            )
            db.add(row)
        row.snippets_missing = missing
        row.reason = reason
        row.last_job_id = job.id
        row.last_attempt_at = func.now()
        # It failed again on a run that tried it, so it is current, not dismissed.
        row.dismissed_at = None
    db.commit()
    return totals


def _refresh_derived_data(db, job: DatasetIngestJob, *, vectors_added: bool) -> None:
    """Tell everything built from snippets/vectors that the set changed.

    - explore base layers (feed, filters, histograms) list the set's snippets
      that have an embedding, so they are rebuilt when vectors were added. With
      a checkpoint in scope the hub only shows rows that have predictions, so
      new snippets stay hidden there until they are scored.
    - the on-disk embedding matrix used by training/inference is rebuilt when
      vectors were added (its fingerprint would also catch this, but explicit is
      cheaper than a stale read racing a retrain).
    - dataset-level projections are recomputed in the background; that task
      invalidates the explore projection layer itself when it finishes, and the
      old projection keeps serving until then.

    Each step is best-effort: a failure here leaves a stale cache, not lost data,
    so it is logged rather than failing a run whose recordings are already in.
    """
    from app.services.explore.hooks import invalidate_dataset_base
    from app.tasks.embedding_tasks import generate_fpv_for_dataset

    snippet_set = db.get(SnippetSet, job.snippet_set_id)
    if vectors_added:
        invalidate_dataset_base(db, job.dataset_id)
        try:
            _invalidate_embedding_cache(snippet_set.id, snippet_set.embedding_model_id)
        except Exception:
            logger.exception("ingest job %s: embedding cache invalidation failed", job.id)
        try:
            generate_fpv_for_dataset.delay(
                dataset_id=job.dataset_id,
                embedding_model_id=snippet_set.embedding_model_id,
            )
        except Exception:
            logger.exception("ingest job %s: could not enqueue projection rebuild", job.id)


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.ingest_tasks.run_dataset_ingest")
def run_dataset_ingest(self, job_id: int):
    """Entry point dispatched by POST /datasets/{id}/add-recordings.

    Runs scanning and segmenting in-process, then hands embedding to a chord
    whose callback (finalize_dataset_ingest) finishes the job.
    """
    db = SessionLocal()
    try:
        job: Optional[DatasetIngestJob] = db.get(DatasetIngestJob, job_id)
        if job is None:
            return {"status": "error", "message": f"DatasetIngestJob({job_id}) not found"}
        if job.status != IngestJobStatus.PENDING:
            # Already given up on (see dataset_ingest_service.dead_job_reason)
            # before a worker got to it; a newer run may own the dataset now.
            return {"status": "skipped", "job_id": job_id, "reason": job.status.value}
        job.status = IngestJobStatus.RUNNING
        job.started_at = func.now()
        job.celery_task_id = self.request.id
        db.commit()

        if job.mode != IngestMode.RETRY_FAILED:
            _scan(db, job)
            _segment(db, job)
        planned = _dispatch_embedding(db, job)
        return {"status": "scheduled", "job_id": job_id, "snippets_to_embed": planned}
    except Exception as e:
        logger.exception("ingest job %s failed", job_id)
        db.rollback()
        _mark_failed(job_id, str(e))
        return {"status": "error", "job_id": job_id, "message": str(e)}
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.ingest_tasks.finalize_dataset_ingest")
def finalize_dataset_ingest(self, results, job_id: int, planned: int):
    """Chord callback after embedding: record results and finish the job."""
    db = SessionLocal()
    try:
        job = db.get(DatasetIngestJob, job_id)
        if job is None:
            return {"status": "error", "message": f"DatasetIngestJob({job_id}) not found"}

        # Each chord child is a chunk of recordings, so results is a list of lists.
        flat = []
        for r in results or []:
            flat.extend(r if isinstance(r, list) else [r])
        embedded = sum(int(r.get("embeddings_inserted", 0) or 0) for r in flat)
        failures = _record_embedding_failures(db, job, flat)
        _update_summary(
            db, job,
            snippets_embedded=embedded,
            snippets_failed=max(planned - embedded, 0),
            embedding_failures=failures,
        )

        _set_stage(db, job, IngestStage.FINALIZING)
        _refresh_derived_data(db, job, vectors_added=embedded > 0)

        # Scoring needs the model, so it runs on the pam_al worker; that task
        # completes the job. It also picks up snippets an earlier run embedded
        # but failed to score, so it is dispatched even when nothing was embedded.
        _set_stage(db, job, IngestStage.SCORING)
        try:
            from app.tasks.pam_al_tasks import pam_al_score_ingested_snippets

            pam_al_score_ingested_snippets.delay(job_id)
        except Exception as e:
            logger.exception("ingest job %s: could not enqueue scoring", job_id)
            complete_job(job_id, error=f"Scoring could not start: {e}", scoring_skipped_reason="scoring_failed")
        return {"status": "scoring", "job_id": job_id, "snippets_embedded": embedded}
    except Exception as e:
        logger.exception("finalizing ingest job %s failed", job_id)
        db.rollback()
        _mark_failed(job_id, str(e))
        return {"status": "error", "job_id": job_id, "message": str(e)}
    finally:
        db.close()
