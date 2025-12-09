"""
Celery tasks for recording and snippet processing.

These tasks wrap DatasetService + SnippetService.
The tasks DO NOT implement business logic themselves.
"""

from contextlib import contextmanager
from celery import shared_task

from app.database import SessionLocal
from app.models.dataset import Dataset
from app.models.snippet import SnippetConfig
from app.services.dataset_service import DatasetService


# ---------------------------------------------------------
# Safe session helper
# ---------------------------------------------------------

@contextmanager
def session_scope():
    """Provide a transactional scope around task operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------
# Scan dataset (async)
# ---------------------------------------------------------

@shared_task(bind=True, name="tasks.scan_dataset")
def scan_dataset(self, dataset_id: int):
    """
    Asynchronously scan recordings under a dataset.
    """
    with session_scope() as db:
        svc = DatasetService(db)
        dataset = svc.get_dataset(dataset_id)

        if dataset is None:
            return {"status": "error", "message": "dataset_not_found"}

        # Optional metadata update for task monitors
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


# ---------------------------------------------------------
# Snippet generation (placeholder until segmentation is implemented)
# ---------------------------------------------------------

@shared_task(bind=True, name="tasks.generate_snippets")
def generate_snippets(self, dataset_id: int, snippet_config_id: int):
    """
    Placeholder snippet-generation task.
    Actual segmentation MUST be implemented in SnippetService or a dedicated class.
    """
    with session_scope() as db:
        dataset = db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return {"status": "error", "message": "dataset_not_found"}

        cfg = db.query(SnippetConfig).filter_by(id=snippet_config_id).first()
        if not cfg:
            return {"status": "error", "message": "snippet_config_not_found"}

        # TODO: SnippetService integration
        return {
            "status": "not_implemented",
            "dataset_id": dataset_id,
            "snippet_config_id": snippet_config_id,
        }


# ---------------------------------------------------------
# Orchestration: scan dataset → (future) generate snippets
# ---------------------------------------------------------

@shared_task(bind=True, name="tasks.process_dataset")
def process_dataset(self, dataset_id: int):
    """
    High-level pipeline:
    1. Scan for recordings
    2. Kick off snippet generation for all configs (future)
    """
    scan_result = scan_dataset.delay(dataset_id)

    return {
        "status": "submitted",
        "dataset_id": dataset_id,
        "scan_task_id": scan_result.id,
    }
