"""
Celery tasks for dataset scanning and orchestration.
"""

from contextlib import contextmanager
from celery import shared_task

from app.database import SessionLocal
from app.services.dataset_service import DatasetService


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

@shared_task(bind=True)
def scan_dataset(self, dataset_id: int):
    """
    Discover recordings in dataset.source_uri.
    Uses DatasetService.scan_recordings().
    """
    with session_scope() as db:
        svc = DatasetService(db)

        dataset = svc.get_dataset(dataset_id)
        if dataset is None:
            return {"status": "error", "message": "dataset_not_found"}

        self.update_state(state="SCANNING", meta={"dataset_id": dataset_id})

        try:
            new_recs = svc.scan_recordings(dataset)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "recordings_created": len(new_recs),
        }


# --------------------------------------------------------------------
# Orchestration: dataset processing = scan only
# --------------------------------------------------------------------

@shared_task(bind=True)
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
