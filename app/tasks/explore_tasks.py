"""
Celery task that builds explore-store layers (see app/services/explore).
"""

from __future__ import annotations

import logging

from app.celery_app import celery_app
from app.database import SessionLocal

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.explore_tasks.build_explore_layer", acks_late=False)
def build_explore_layer(self, kind: str, params: dict):
    from app.services.explore import registry
    from app.services.explore.builders import base_key, model_key, projection_key

    if kind == "base":
        key = base_key(params["snippet_set_id"])
    elif kind == "model":
        key = model_key(params["checkpoint_id"], params["snippet_set_id"])
    elif kind == "projection":
        key = projection_key(params["dataset_id"], params["embedding_model_id"])
    else:
        return {"status": "error", "message": f"unknown kind {kind!r}"}

    client = registry._redis_client()
    db = SessionLocal()
    try:
        registry.build_inline(db, kind, key, params)
        return {"status": "success", "kind": kind, "key": key}
    except Exception as exc:
        logger.exception("explore: building %s/%s failed", kind, key)
        if client is not None:
            try:
                client.set(
                    registry.error_key(kind, key),
                    f"Building the {kind} explore layer failed: {exc}"[:2000],
                    ex=registry.BUILD_ERROR_TTL_SECONDS,
                )
            except Exception:
                pass
        return {"status": "error", "kind": kind, "key": key, "message": str(exc)}
    finally:
        db.close()
        if client is not None:
            try:
                client.delete(registry.guard_key(kind, key))
            except Exception:
                pass
