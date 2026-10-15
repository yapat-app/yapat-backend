"""
Start and report "add recordings to an existing dataset" runs (yapat-backend#70).

The heavy work (scan -> segment -> embed -> finalize -> score) runs in Celery
(app/tasks/ingest_tasks.py). This service only validates the request, creates the
DatasetIngestJob row and turns a job into the API payload.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.config import settings
from app.models.dataset import Dataset as DatasetModel
from app.models.dataset_ingest_job import (
    ACTIVE_INGEST_STATUSES,
    DatasetIngestJob,
    IngestJobStatus,
    IngestMode,
    IngestStage,
    empty_summary,
)
from app.models.recording import Recording
from app.models.recording_embedding_failure import RecordingEmbeddingFailure
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, SnippetSet, SnippetSetStatus
from app.models.pam_active_learning import ALRetrainJob, ALRetrainStatus
from app.models.user import User
from app.services import ingest_progress
from app.services.dataset_service import DatasetService, ReferenceMetadataNotFound

logger = logging.getLogger(__name__)


class IngestRequestError(Exception):
    """A request the API must refuse. `code` is the `detail` sent to the client."""

    def __init__(self, status_code: int, code: str, job_id: Optional[int] = None):
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.job_id = job_id


def _data_root() -> str:
    return os.path.normpath(settings.DATA_ROOT or "/data")


def _abs_under_data_root(source_uri: str) -> str:
    return os.path.normpath(os.path.join(_data_root(), source_uri))


def _contains(parent_abs: str, child_abs: str) -> bool:
    """True when child_abs is parent_abs or lies below it."""
    try:
        return os.path.commonpath([parent_abs, child_abs]) == parent_abs
    except ValueError:
        return False


class DatasetIngestService:
    def __init__(self, db: Session):
        self.db = db
        self.datasets = DatasetService(db)
        # Dead ingest jobs failed during create_job; their waiting retrains are
        # released once that transaction commits.
        self._failed_dead_ids: list[int] = []

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def create_job(
        self,
        dataset_id: int,
        user: User,
        source_uri: Optional[str] = None,
        retry_failed_only: bool = False,
    ) -> DatasetIngestJob:
        """Validate and create a PENDING job. The caller dispatches the Celery task.

        Raises IngestRequestError for every refusal the start endpoint can return.
        """
        # Row lock on the dataset serialises concurrent starts, so two requests
        # cannot both pass the "no job running" check below.
        dataset = (
            self.db.query(DatasetModel)
            .filter(DatasetModel.id == dataset_id)
            .with_for_update()
            .first()
        )
        if dataset is None:
            raise IngestRequestError(404, "Dataset not found")
        if not self.datasets.user_can_manage_dataset(user, dataset):
            raise IngestRequestError(403, "Only admins and team owners can add recordings")

        self._ensure_nothing_running(dataset.id)
        snippet_set, embedding_job = self._require_embedded(dataset)

        if retry_failed_only:
            if source_uri and source_uri.strip():
                raise IngestRequestError(400, "source_uri_not_allowed_with_retry")
            has_failures = (
                self.db.query(RecordingEmbeddingFailure.id)
                .filter(RecordingEmbeddingFailure.snippet_set_id == snippet_set.id)
                .first()
            )
            if has_failures is None:
                raise IngestRequestError(400, "no_failed_recordings")

        new_source_uri = None if retry_failed_only else self._resolve_source_uri(dataset, source_uri)

        if dataset.is_reference:
            try:
                self.datasets.validate_reference_metadata(
                    new_source_uri or dataset.source_uri,
                    dataset.reference_metadata_path,
                )
            except ReferenceMetadataNotFound:
                raise IngestRequestError(400, "reference_metadata_missing")

        if new_source_uri is not None:
            dataset.source_uri = new_source_uri

        job = DatasetIngestJob(
            dataset_id=dataset.id,
            snippet_set_id=snippet_set.id,
            embedding_job_id=embedding_job.id,
            requested_by_user_id=user.id,
            source_uri=new_source_uri,
            mode=IngestMode.RETRY_FAILED if retry_failed_only else IngestMode.SCAN,
            status=IngestJobStatus.PENDING,
            stage=IngestStage.PENDING,
            summary=empty_summary(),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        _release_retrains_of(self.db, self._failed_dead_ids)
        return job

    def _ensure_nothing_running(self, dataset_id: int) -> None:
        active_ingests = (
            self.db.query(DatasetIngestJob)
            .filter(
                DatasetIngestJob.dataset_id == dataset_id,
                DatasetIngestJob.status.in_(ACTIVE_INGEST_STATUSES),
            )
            .order_by(DatasetIngestJob.created_at.desc())
            .all()
        )
        alive = []
        for job in active_ingests:
            if fail_if_dead(job):
                self._failed_dead_ids.append(job.id)
            else:
                alive.append(job)
        if alive:
            raise IngestRequestError(409, "ingest_job_running", job_id=alive[0].id)

        running_embedding = (
            self.db.query(EmbeddingJob.id)
            .filter(
                EmbeddingJob.dataset_id == dataset_id,
                EmbeddingJob.status.in_([EmbeddingJobStatus.PENDING, EmbeddingJobStatus.RUNNING]),
            )
            .first()
        )
        if running_embedding is not None:
            raise IngestRequestError(409, "embedding_job_running", job_id=running_embedding[0])

        running_retrain = (
            self.db.query(ALRetrainJob.id)
            .filter(
                ALRetrainJob.dataset_id == dataset_id,
                ALRetrainJob.status.in_([ALRetrainStatus.PENDING, ALRetrainStatus.RUNNING]),
            )
            .first()
        )
        if running_retrain is not None:
            raise IngestRequestError(409, "retrain_job_running", job_id=running_retrain[0])

    def _require_embedded(self, dataset: DatasetModel) -> tuple[SnippetSet, EmbeddingJob]:
        """The default snippet set must exist and be READY: new snippets are added to it."""
        if dataset.default_snippet_set_id is None:
            raise IngestRequestError(400, "no_embeddings_yet")
        snippet_set = self.db.get(SnippetSet, dataset.default_snippet_set_id)
        if snippet_set is None:
            raise IngestRequestError(400, "no_embeddings_yet")
        if snippet_set.status != SnippetSetStatus.READY:
            raise IngestRequestError(400, "embeddings_not_ready")
        embedding_job = (
            self.db.query(EmbeddingJob)
            .filter(EmbeddingJob.snippet_set_id == snippet_set.id)
            .order_by(EmbeddingJob.created_at.asc())
            .first()
        )
        if embedding_job is None:
            raise IngestRequestError(400, "no_embeddings_yet")
        return snippet_set, embedding_job

    def _resolve_source_uri(self, dataset: DatasetModel, source_uri: Optional[str]) -> Optional[str]:
        """Validated new source_uri, or None when the current folder is just rescanned."""
        if source_uri is None or not source_uri.strip():
            return None
        candidate = source_uri.strip().strip("/")
        try:
            self.datasets.validate_source_uri(candidate)
        except ValueError:
            raise IngestRequestError(400, "invalid_source_uri")

        new_abs = _abs_under_data_root(candidate)
        current_abs = _abs_under_data_root(dataset.source_uri)
        if new_abs == current_abs:
            return None
        # Existing recordings keep their DATA_ROOT-relative file_path, which stays
        # valid only if the new folder still contains the old one.
        if not _contains(new_abs, current_abs):
            raise IngestRequestError(400, "invalid_source_uri")

        # The scan registers every audio file under the folder, so a folder that
        # holds another dataset's folder would pull that dataset's files in too.
        others = (
            self.db.query(DatasetModel.source_uri)
            .filter(DatasetModel.id != dataset.id)
            .all()
        )
        for (other_uri,) in others:
            if other_uri and _contains(new_abs, _abs_under_data_root(other_uri)):
                raise IngestRequestError(400, "source_uri_overlaps_other_dataset")

        return os.path.relpath(new_abs, _data_root()).replace(os.sep, "/")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_job(self, dataset_id: int, job_id: int) -> Optional[DatasetIngestJob]:
        job = (
            self.db.query(DatasetIngestJob)
            .filter(DatasetIngestJob.id == job_id, DatasetIngestJob.dataset_id == dataset_id)
            .first()
        )
        if job is not None and fail_if_dead(job):
            self.db.commit()
            _release_retrains_of(self.db, [job.id])
        return job

    def active_jobs_by_dataset(self, dataset_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        """Newest pending/running job per dataset, shaped as ActiveIngestJob.

        Dead jobs (see dead_job_reason) are failed here too: the UI disables the
        start button while a job is active, so waiting for the next start request
        to clear them would leave the dataset stuck.
        """
        ids = list(dataset_ids)
        if not ids:
            return {}
        rows = (
            self.db.query(DatasetIngestJob)
            .filter(
                DatasetIngestJob.dataset_id.in_(ids),
                DatasetIngestJob.status.in_(ACTIVE_INGEST_STATUSES),
            )
            .order_by(DatasetIngestJob.created_at.desc())
            .all()
        )
        released = [job for job in rows if fail_if_dead(job)]
        if released:
            self.db.commit()
            _release_retrains_of(self.db, [job.id for job in released])
        result: dict[int, dict[str, Any]] = {}
        for job in rows:
            if job in released:
                continue
            result.setdefault(
                job.dataset_id,
                {
                    "id": job.id,
                    "mode": job.mode.value,
                    "status": job.status.value,
                    "stage": job.stage.value,
                    "started_at": job.started_at,
                },
            )
        return result


    # ------------------------------------------------------------------
    # Embedding failures
    # ------------------------------------------------------------------

    def failure_counts_by_dataset(self, dataset_ids: Iterable[int]) -> dict[int, dict[str, int]]:
        """{dataset_id: {recordings, snippets}} over current (non-dismissed) failures."""
        ids = list(dataset_ids)
        if not ids:
            return {}
        rows = (
            self.db.query(
                RecordingEmbeddingFailure.dataset_id,
                func.count(RecordingEmbeddingFailure.id),
                func.coalesce(func.sum(RecordingEmbeddingFailure.snippets_missing), 0),
            )
            .filter(
                RecordingEmbeddingFailure.dataset_id.in_(ids),
                RecordingEmbeddingFailure.dismissed_at.is_(None),
            )
            .group_by(RecordingEmbeddingFailure.dataset_id)
            .all()
        )
        return {ds: {"recordings": int(n), "snippets": int(s)} for ds, n, s in rows}

    def list_failures(
        self, dataset_id: int, *, skip: int = 0, limit: int = 100, include_dismissed: bool = False
    ) -> dict[str, Any]:
        base = self.db.query(RecordingEmbeddingFailure).filter(
            RecordingEmbeddingFailure.dataset_id == dataset_id
        )
        counts = self.failure_counts_by_dataset([dataset_id]).get(dataset_id, {"recordings": 0, "snippets": 0})
        dismissed = base.filter(RecordingEmbeddingFailure.dismissed_at.isnot(None)).count()

        page = (
            base if include_dismissed else base.filter(RecordingEmbeddingFailure.dismissed_at.is_(None))
        )
        rows = (
            page.join(Recording, Recording.id == RecordingEmbeddingFailure.recording_id)
            .with_entities(RecordingEmbeddingFailure, Recording.file_name, Recording.file_path)
            # Current failures first, then by file so the list is stable across pages.
            .order_by(RecordingEmbeddingFailure.dismissed_at.isnot(None), Recording.file_path)
            .offset(max(skip, 0))
            .limit(max(1, min(limit, 500)))
            .all()
        )
        return {
            **counts,
            "dismissed_recordings": int(dismissed),
            "items": [
                {
                    "recording_id": f.recording_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "snippets_missing": f.snippets_missing,
                    "reason": f.reason,
                    "last_attempt_at": f.last_attempt_at,
                    "last_job_id": f.last_job_id,
                    "dismissed": f.dismissed_at is not None,
                }
                for f, file_name, file_path in rows
            ],
        }

    def dismiss_failures(self, dataset_id: int) -> int:
        """"Continue without them": stop counting and auto-retrying current failures."""
        n = (
            self.db.query(RecordingEmbeddingFailure)
            .filter(
                RecordingEmbeddingFailure.dataset_id == dataset_id,
                RecordingEmbeddingFailure.dismissed_at.is_(None),
            )
            .update({RecordingEmbeddingFailure.dismissed_at: func.now()}, synchronize_session=False)
        )
        self.db.commit()
        return int(n)


# A queued job no worker has picked up after this long is reported as stalled.
PENDING_STALL_AFTER_SECONDS = ingest_progress.STALL_AFTER_SECONDS
# Thresholds after which an active job is treated as dead and failed. PENDING
# waits longer than the stall warning: the `processing` queue is shared with
# dataset scans, so a healthy job can sit queued behind one for a while.
# Generous for RUNNING: one huge file can hash or embed for a long time between
# progress updates, and a falsely-failed job would race the real one.
PENDING_DEAD_AFTER_SECONDS = 30 * 60
RUNNING_DEAD_AFTER_SECONDS = 60 * 60


def dead_job_reason(job: DatasetIngestJob) -> Optional[str]:
    """Why an active job is considered dead, or None while it may still be alive."""
    if job.status == IngestJobStatus.PENDING:
        age = ingest_progress.seconds_since(job.created_at)
        if age is not None and age > PENDING_DEAD_AFTER_SECONDS:
            return (
                "Never started: no Celery worker picked up the task "
                "(is the worker running the current code?)"
            )
        return None
    if job.status == IngestJobStatus.RUNNING:
        _total, _done, last_update = ingest_progress.read(job.id)
        if last_update is not None:
            quiet = ingest_progress.seconds_since_epoch(last_update)
        else:
            quiet = ingest_progress.seconds_since(job.started_at)
        if quiet is not None and quiet > RUNNING_DEAD_AFTER_SECONDS:
            return "No progress for over an hour; the worker probably stopped"
    return None


def _release_retrains_of(db: Session, ingest_job_ids: list[int]) -> None:
    """Start the retrains that waited for these ingest jobs (now failed as dead)."""
    if not ingest_job_ids:
        return
    from app.services.retrain_deferral import release_waiting_retrains

    try:
        release_waiting_retrains(db, ingest_job_ids)
    except Exception:
        # The cleanup beat releases them later.
        logger.exception("releasing retrains waiting for ingest jobs %s failed", ingest_job_ids)


def fail_if_dead(job: DatasetIngestJob) -> bool:
    """Mark a dead active job FAILED (caller commits). True when it was released.

    A dropped Celery message or a killed worker would otherwise keep the job
    active, and with it the dataset locked, forever.
    """
    reason = dead_job_reason(job)
    if reason is None:
        return False
    job.status = IngestJobStatus.FAILED
    job.stage = IngestStage.FAILED
    job.error_message = reason
    job.completed_at = func.now()
    return True


def job_to_response(job: DatasetIngestJob) -> dict[str, Any]:
    """IngestJobResponse payload: DB row for stage/summary, Redis for live counts."""
    total, done, last_update = (None, None, None)
    if job.status == IngestJobStatus.RUNNING:
        total, done, last_update = ingest_progress.read(job.id)

    if job.status == IngestJobStatus.PENDING:
        age = ingest_progress.seconds_since(job.created_at)
        stalled = age is not None and age > PENDING_STALL_AFTER_SECONDS
    else:
        stalled = job.status == IngestJobStatus.RUNNING and ingest_progress.is_stalled(last_update)

    total = total or 0
    done = min(done or 0, total) if total else (done or 0)

    if job.status == IngestJobStatus.COMPLETED:
        percent: Optional[float] = 100.0
    elif total:
        percent = round(100.0 * done / total, 1)
    else:
        percent = None

    summary = {**empty_summary(), **(job.summary or {})}

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "mode": job.mode.value,
        "status": job.status.value,
        "stage": job.stage.value,
        "done": done,
        "total": total,
        "percent": percent,
        "stalled": stalled,
        "error": job.error_message,
        "summary": summary,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
