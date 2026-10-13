"""
Celery tasks for WSSED operations

Handles asynchronous training and detection tasks.
"""

from celery import shared_task
from contextlib import contextmanager
import logging

from app.config import settings
from app.database import SessionLocal
from app.models.wssed import TrainingStatus, WSSEDTrainingJob
from app.services.wssed_service import WSSEDService

logger = logging.getLogger(__name__)

TERMINAL_TRAINING_STATUSES = (TrainingStatus.COMPLETED, TrainingStatus.FAILED)


def schedule_training_poll(job_id: int) -> None:
    """Queue the next poll_training_status for a job sent to the GPU server."""
    poll_training_status.apply_async(
        args=[job_id], countdown=settings.WSSED_TRAINING_POLL_INTERVAL
    )


@contextmanager
def session_scope():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@shared_task(bind=True, name="app.tasks.wssed_tasks.trigger_wssed_training", acks_late=False)
def trigger_wssed_training(self, job_id: int):
    """
    Trigger WSSED training on GPU server.
    
    Args:
        job_id: Training job ID
    
    Returns:
        Dict with status and result
    """
    logger.info(f"Starting WSSED training task for job {job_id}")
    
    with session_scope() as db:
        service = WSSEDService(db)
        
        try:
            # This will make an async HTTP call to GPU server
            import asyncio
            task_id = asyncio.run(service.trigger_remote_training(job_id))
            
            logger.info(f"WSSED training triggered for job {job_id}, task_id: {task_id}")

            try:
                schedule_training_poll(job_id)
            except Exception as e:
                # The beat safety net still syncs the job, just less often.
                logger.warning("Could not schedule status polling for WSSED job %s: %s", job_id, e)

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": task_id,
                "message": "Training started on GPU server"
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger WSSED training for job {job_id}: {e}", exc_info=True)
            job = db.query(WSSEDTrainingJob).filter(WSSEDTrainingJob.id == job_id).first()
            if job is not None:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                db.commit()
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }


@shared_task(bind=True, name="app.tasks.wssed_tasks.trigger_wssed_detection")
def trigger_wssed_detection(self, job_id: int, threshold: float = 0.5):
    """
    Trigger WSSED detection on GPU server.
    
    Args:
        job_id: Training job ID
        threshold: Detection confidence threshold
    
    Returns:
        Dict with status and result
    """
    logger.info(f"Starting WSSED detection task for job {job_id}, threshold={threshold}")
    
    with session_scope() as db:
        service = WSSEDService(db)
        
        try:
            # Trigger detection on GPU server
            import asyncio
            task_id = asyncio.run(service.trigger_detection(job_id, threshold))
            
            logger.info(f"WSSED detection triggered for job {job_id}, task_id: {task_id}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "task_id": task_id,
                "threshold": threshold,
                "message": "Detection started on GPU server"
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger WSSED detection for job {job_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }


@shared_task(bind=True, name="app.tasks.wssed_tasks.poll_training_status")
def poll_training_status(self, job_id: int):
    """Sync one job's status from the GPU server; re-queues itself until COMPLETED/FAILED."""
    with session_scope() as db:
        service = WSSEDService(db)

        try:
            import asyncio
            job = asyncio.run(service.update_training_status(job_id))
            status = job.status
            result = {
                "status": "success",
                "job_id": job_id,
                "training_status": status.value,
                "model_path": job.model_path,
                "model_paths": job.model_paths,
                "completed": status in TERMINAL_TRAINING_STATUSES,
            }
        except Exception as e:
            logger.error(f"Failed to poll training status for job {job_id}: {e}")
            # Keep polling through transient errors while the job is unfinished.
            db.rollback()
            job = db.query(WSSEDTrainingJob).filter(WSSEDTrainingJob.id == job_id).first()
            status = job.status if job is not None else None
            result = {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }

    if status is not None and status not in TERMINAL_TRAINING_STATUSES:
        schedule_training_poll(job_id)
    return result


@shared_task(name="app.tasks.wssed_tasks.sync_wssed_training_jobs")
def sync_wssed_training_jobs():
    """Beat safety net: sync TRAINING jobs whose poll chain was lost (e.g. worker restart)."""
    from app.models.wssed import TrainingStatus

    with session_scope() as db:
        active_jobs = (
            db.query(WSSEDTrainingJob)
            .filter(WSSEDTrainingJob.status == TrainingStatus.TRAINING)
            .all()
        )

        if not active_jobs:
            return {"synced": 0}

        service = WSSEDService(db)
        synced, failed = 0, 0
        for job in active_jobs:
            try:
                import asyncio
                asyncio.run(service.update_training_status(job.id))
                synced += 1
            except Exception as e:
                failed += 1
                logger.warning("Failed to sync WSSED job %s: %s", job.id, e)

        logger.info("WSSED sync: %s synced, %s failed", synced, failed)
        return {"synced": synced, "failed": failed}


@shared_task(bind=True, name="app.tasks.wssed_tasks.store_detection_results")
def store_detection_results(self, job_id: int, predictions: list):
    """
    Store detection results in database.
    
    This task is called by the GPU server webhook after detection completes.
    
    Args:
        job_id: Training job ID
        predictions: List of prediction dicts
    
    Returns:
        Dict with count of stored predictions
    """
    logger.info(f"Storing detection results for job {job_id}, count={len(predictions)}")
    
    with session_scope() as db:
        service = WSSEDService(db)
        
        try:
            count = service.store_predictions(job_id, predictions)
            
            logger.info(f"Stored {count} predictions for job {job_id}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "predictions_stored": count
            }
            
        except Exception as e:
            logger.error(f"Failed to store detection results for job {job_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }
