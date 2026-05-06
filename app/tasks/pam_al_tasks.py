"""
Celery tasks for PAM Active Learning.

Heavy operations (train-from-scratch, manual retrain, auto-retrain from feedback)
are offloaded here so HTTP endpoints return immediately with a job_id the client
can poll via GET /api/pam-al/retrain/jobs/{job_id}.

Flow
----
1. API endpoint calls the corresponding service.setup_*() method synchronously
   to create ALModelCheckpoint (LOADING) + ALRetrainJob (PENDING) records and
   returns both IDs to the caller.
2. The API endpoint dispatches the matching task below with .delay(checkpoint_id, job_id).
3. The API returns ALJobDispatch to the client immediately.
4. The Celery worker picks up the task, calls service.execute_*(), and updates
   the checkpoint/job records to AVAILABLE/COMPLETED (or ERROR/FAILED on failure).
"""

import logging
from datetime import datetime, timezone

from app.celery_app import celery_app
from app.database import SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: get a scoped DB session that is always closed after the task
# ---------------------------------------------------------------------------

def _make_service(db):
    from app.services.pam_active_learning_service import PAMActiveLearningService
    return PAMActiveLearningService(db)


def _ensure_job_failed(job_id: int, error: Exception) -> None:
    """
    Ensure the retrain job is in a terminal FAILED state.

    Uses a fresh DB session so failures can still be recorded after exceptions.
    """
    from app.models.pam_active_learning import ALRetrainJob, ALRetrainStatus

    db2 = SessionLocal()
    try:
        job = db2.query(ALRetrainJob).filter(ALRetrainJob.id == job_id).first()
        if job is None:
            logger.warning(
                "_ensure_job_failed: job %d not found — likely already cascade-deleted "
                "(restart the service to apply the mark-instead-of-delete fix)",
                job_id,
            )
            return
        if job.status in {ALRetrainStatus.COMPLETED, ALRetrainStatus.FAILED}:
            return  # already in a terminal state
        job.status = ALRetrainStatus.FAILED
        job.error_message = str(error)
        job.completed_at = datetime.now(timezone.utc)
        db2.commit()
        logger.info("_ensure_job_failed: marked job %d as FAILED", job_id)
    except Exception:
        db2.rollback()
        logger.exception("_ensure_job_failed: could not mark job %d as FAILED", job_id)
    finally:
        db2.close()


# ---------------------------------------------------------------------------
# Task 1: Cold-start training (train from scratch)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.tasks.pam_al_tasks.pam_al_train_from_scratch",
    max_retries=0,
)
def pam_al_train_from_scratch(self, checkpoint_id: int, job_id: int):
    """
    Run PAM AL cold-start training in the background.

    The checkpoint (LOADING) and job (PENDING) records have already been
    created by the API endpoint before this task was dispatched.
    """
    db = SessionLocal()
    try:
        svc = _make_service(db)
        self.update_state(
            state="RUNNING",
            meta={"checkpoint_id": checkpoint_id, "job_id": job_id},
        )
        svc.execute_train_from_scratch(checkpoint_id=checkpoint_id, job_id=job_id)
        logger.info(
            "pam_al_train_from_scratch completed: checkpoint_id=%d job_id=%d",
            checkpoint_id, job_id,
        )
        return {"status": "completed", "checkpoint_id": checkpoint_id, "job_id": job_id}

    except Exception as e:
        logger.exception(
            "pam_al_train_from_scratch failed: checkpoint_id=%d job_id=%d error=%s",
            checkpoint_id, job_id, str(e),
        )
        _ensure_job_failed(job_id, e)
        return {"status": "failed", "checkpoint_id": checkpoint_id, "job_id": job_id, "error": str(e)}

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2: Manual retrain
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.tasks.pam_al_tasks.pam_al_manual_retrain",
    max_retries=0,
)
def pam_al_manual_retrain(self, checkpoint_id: int, job_id: int):
    """
    Run a manually triggered PAM AL retrain in the background.

    The new checkpoint (LOADING) and job (PENDING) records have already been
    created by the API endpoint before this task was dispatched.
    """
    db = SessionLocal()
    try:
        svc = _make_service(db)
        self.update_state(
            state="RUNNING",
            meta={"checkpoint_id": checkpoint_id, "job_id": job_id},
        )
        svc.execute_manual_retrain(checkpoint_id=checkpoint_id, job_id=job_id)
        logger.info(
            "pam_al_manual_retrain completed: checkpoint_id=%d job_id=%d",
            checkpoint_id, job_id,
        )
        return {"status": "completed", "checkpoint_id": checkpoint_id, "job_id": job_id}

    except Exception as e:
        logger.exception(
            "pam_al_manual_retrain failed: checkpoint_id=%d job_id=%d error=%s",
            checkpoint_id, job_id, str(e),
        )
        _ensure_job_failed(job_id, e)
        return {"status": "failed", "checkpoint_id": checkpoint_id, "job_id": job_id, "error": str(e)}

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 3: Auto-retrain triggered by feedback threshold
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.tasks.pam_al_tasks.pam_al_auto_retrain",
    max_retries=0,
)
def pam_al_auto_retrain(self, checkpoint_id: int, job_id: int):
    """
    Run an automatically triggered PAM AL retrain in the background.

    Called when the feedback count since the last retrain reaches RETRAIN_AFTER.
    The new checkpoint (LOADING) and job (PENDING) records have already been
    created inside submit_feedback → setup_auto_retrain() before this task
    was dispatched.
    """
    db = SessionLocal()
    try:
        svc = _make_service(db)
        self.update_state(
            state="RUNNING",
            meta={"checkpoint_id": checkpoint_id, "job_id": job_id},
        )
        svc.execute_auto_retrain(checkpoint_id=checkpoint_id, job_id=job_id)
        logger.info(
            "pam_al_auto_retrain completed: checkpoint_id=%d job_id=%d",
            checkpoint_id, job_id,
        )
        return {"status": "completed", "checkpoint_id": checkpoint_id, "job_id": job_id}

    except Exception as e:
        logger.exception(
            "pam_al_auto_retrain failed: checkpoint_id=%d job_id=%d error=%s",
            checkpoint_id, job_id, str(e),
        )
        _ensure_job_failed(job_id, e)
        return {"status": "failed", "checkpoint_id": checkpoint_id, "job_id": job_id, "error": str(e)}

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 4: Inference / prediction creation (async)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.tasks.pam_al_tasks.pam_al_create_predictions",
    max_retries=0,
)
def pam_al_create_predictions(self, job_id: int, inference_body: dict):
    """Run inference and (re)create predictions asynchronously.

    This is used by POST /api/pam-al/inference/get-or-create so the API doesn't
    synchronously load embeddings/models and serialize large prediction payloads.
    """
    from app.models.pam_active_learning import ALRetrainJob, ALRetrainStatus
    from app.services.pam_al.service import PAMActiveLearningService

    db = SessionLocal()
    try:
        svc = PAMActiveLearningService(db)

        job = db.query(ALRetrainJob).filter(ALRetrainJob.id == job_id).one_or_none()
        if job is None:
            raise ValueError(f"Inference job {job_id} not found")

        job.status = ALRetrainStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        self.update_state(state="RUNNING", meta={"job_id": job_id})

        # Force refresh so the job is deterministic.
        inference_body = {**(inference_body or {}), "force_refresh": True}

        # Service method expects a request object with attribute access.
        from app.schemas.pam_active_learning import ALRunInferenceRequest

        req = ALRunInferenceRequest(**inference_body)
        svc.get_or_create_predictions(req)

        job.status = ALRetrainStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        logger.info("pam_al_create_predictions completed: job_id=%d", job_id)
        return {"status": "completed", "job_id": job_id}

    except Exception as e:
        logger.exception("pam_al_create_predictions failed: job_id=%d error=%s", job_id, str(e))
        _ensure_job_failed(job_id, e)
        return {"status": "failed", "job_id": job_id, "error": str(e)}

    finally:
        db.close()
