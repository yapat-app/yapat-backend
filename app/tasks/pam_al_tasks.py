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

from app.celery_app import celery_app
from app.database import SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: get a scoped DB session that is always closed after the task
# ---------------------------------------------------------------------------

def _make_service(db):
    from app.services.pam_active_learning_service import PAMActiveLearningService
    return PAMActiveLearningService(db)


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
        return {"status": "failed", "checkpoint_id": checkpoint_id, "job_id": job_id, "error": str(e)}

    finally:
        db.close()
