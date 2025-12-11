"""
Celery tasks for dataset scanning and orchestration.
"""

from contextlib import contextmanager
from celery import shared_task

from app.database import SessionLocal
from app.services.dataset_service import DatasetService
from app.celery_app import celery_app


# --------------------------------------------------------------------
# Context manager for DB session handling
# --------------------------------------------------------------------

@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------
# Task: Scan dataset for recordings
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.processing_tasks.scan_dataset")
def scan_dataset(self, dataset_id: int):
    """
    Discover recordings in dataset.source_uri.
    Uses DatasetService.scan_recordings().
    """
    import logging
    logger = logging.getLogger(__name__)
    
    with session_scope() as db:
        svc = DatasetService(db)

        dataset = svc.get_dataset(dataset_id)
        if dataset is None:
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        self.update_state(state="SCANNING", meta={"dataset_id": dataset_id})
        logger.info(f"Starting scan for dataset {dataset_id} (source_uri: {dataset.source_uri})")

        try:
            new_recs = svc.scan_recordings(dataset)
            logger.info(f"Scan completed for dataset {dataset_id}: {len(new_recs)} recordings created")
        except Exception as e:
            error_msg = f"Scan failed for dataset {dataset_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "recordings_created": len(new_recs),
        }


# --------------------------------------------------------------------
# Orchestration: dataset processing = scan only
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.processing_tasks.process_dataset")
def process_dataset(self, dataset_id: int):
    """
    Trigger the dataset scanning pipeline.

    New architecture:
    - Only scanning runs at dataset creation
    - Snippet generation is triggered by embedding jobs, not datasets
    """
    result = scan_dataset.delay(dataset_id)

    return {
        "status": "submitted",
        "dataset_id": dataset_id,
        "scan_task_id": result.id,
    }

@shared_task(bind=True)
def generate_snippets(self, dataset_id: int, snippet_set_id: int):
    """
    Placeholder for snippet segmentation logic.

    In the new SnippetSet architecture, snippet generation
    happens inside embedding jobs (run_embedding).
    """
    return {"status": "not_implemented"}
