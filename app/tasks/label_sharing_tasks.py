"""
Celery task: share a dataset's labels between its snippet sets.

Runs after a snippet set is embedded, and on request for labels made before
label sharing existed. See app/services/label_sharing.py for the rules.
"""

import logging

from app import database
from app.celery_app import celery_app
from app.models.embedding import SnippetSet
from app.services import label_sharing

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.label_sharing_tasks.share_labels_for_dataset")
def share_labels_for_dataset(
    dataset_id: int,
    target_snippet_set_id: int | None = None,
    source_snippet_set_id: int | None = None,
) -> dict:
    """Share a dataset's labels.

    ``target_snippet_set_id`` alone: onto that (new) set. With
    ``source_snippet_set_id`` too: a one-way transfer from that set.
    """
    db = database.SessionLocal()
    try:
        set_count = db.query(SnippetSet.id).filter(SnippetSet.dataset_id == dataset_id).count()
        if set_count < 2:
            return {"dataset_id": dataset_id, "skipped": "only one snippet set"}
        totals = label_sharing.share_dataset(
            db,
            dataset_id,
            target_snippet_set_id=target_snippet_set_id,
            source_snippet_set_id=source_snippet_set_id,
        )
        return {
            "dataset_id": dataset_id,
            "source_snippet_set_id": source_snippet_set_id,
            "target_snippet_set_id": target_snippet_set_id,
            **totals,
        }
    except Exception:
        db.rollback()
        logger.exception("Label sharing failed for dataset %s", dataset_id)
        raise
    finally:
        db.close()
