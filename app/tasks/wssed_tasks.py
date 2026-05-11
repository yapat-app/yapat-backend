"""
Celery tasks for WSSED operations

Handles asynchronous training and detection tasks.
"""

from celery import shared_task
from contextlib import contextmanager
import logging

from app.database import SessionLocal
from app.models.wssed import TrainingStatus, WSSEDTrainingJob
from app.services.wssed_service import WSSEDService

logger = logging.getLogger(__name__)


@contextmanager
def session_scope():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@shared_task(bind=True, name="app.tasks.wssed_tasks.trigger_wssed_training")
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
    """
    Poll GPU server for training status update.
    
    This task can be scheduled to periodically check training progress.
    
    Args:
        job_id: Training job ID
    
    Returns:
        Dict with current status
    """
    with session_scope() as db:
        service = WSSEDService(db)
        
        try:
            import asyncio
            job = asyncio.run(service.update_training_status(job_id))
            
            return {
                "status": "success",
                "job_id": job_id,
                "training_status": job.status.value,
                "model_path": job.model_path,
                "completed": job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED),
            }
            
        except Exception as e:
            logger.error(f"Failed to poll training status for job {job_id}: {e}")
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }


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
