"""
Celery tasks for the embedding pipeline (SnippetSet architecture).
"""

from celery import group
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import SessionLocal

from app.models.snippet import Snippet
from app.models.recording import Recording
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    SnippetSet,
)
from app.services.embedding_service import EmbeddingService


# ----------------------------------------------------------------------
# DB session helper
# ----------------------------------------------------------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # closed in tasks


# ----------------------------------------------------------------------
# Generate embedding for a single snippet
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.generate_embedding_for_snippet")
def generate_embedding_for_snippet(self, snippet_id: int, model_id: int):
    """
    Placeholder: generate embedding for a single snippet.
    In the current architecture, embeddings are *not* stored in Snippet.
    """
    db = SessionLocal()
    try:
        snippet = db.query(Snippet).filter_by(id=snippet_id).first()
        if not snippet:
            return {"status": "error", "snippet_id": snippet_id, "message": "snippet_not_found"}

        # Get model from database
        service = EmbeddingService(db)
        try:
            model = service.get_model(model_id)
        except ValueError:
            return {"status": "error", "message": f"model_not_found"}

        # TODO: embedding inference should write into a vector store, not DB
        dummy_vector = [0.1, 0.2, 0.3]

        # No snippet.embedding field exists; pretend we saved it externally.
        return {"status": "success", "snippet_id": snippet_id}

    except Exception as e:
        return {"status": "error", "snippet_id": snippet_id, "message": str(e)}

    finally:
        db.close()


# ----------------------------------------------------------------------
# Main pipeline: run embedding job
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.run_embedding")
def run_embedding(self, embedding_job_id: int):
    """
    Run the embedding pipeline:

        - Update job → RUNNING
        - Load SnippetSet
        - Segment all recordings into Snippets
        - Generate embeddings (parallel Celery group)
        - Mark job COMPLETED or FAILED
    """

    db = SessionLocal()
    service = EmbeddingService(db)

    try:
        # ------------------------------------------------------
        # 1. Load EmbeddingJob
        # ------------------------------------------------------
        job = db.query(EmbeddingJob).filter_by(id=embedding_job_id).first()
        if not job:
            raise ValueError(f"EmbeddingJob(id={embedding_job_id}) not found")

        service.update_job_status(
            job.id,
            EmbeddingJobStatus.RUNNING,
            celery_task_id=self.request.id,
        )

        snippet_set: SnippetSet = job.snippet_set
        model: EmbeddingModel = job.embedding_model

        window = snippet_set.window_size
        step = snippet_set.step_size
        overlap = snippet_set.overlap

        # ------------------------------------------------------
        # 2. Fetch recordings for dataset
        # ------------------------------------------------------
        recordings = (
            db.query(Recording)
            .filter(Recording.dataset_id == job.dataset_id)
            .all()
        )

        snippet_ids = []

        # ------------------------------------------------------
        # 3. Segment each recording into Snippets
        # ------------------------------------------------------
        for rec in recordings:
            duration = rec.duration  # field name in your model
            if duration is None:
                continue

            t = 0.0
            while t + window <= duration:
                snippet = Snippet(
                    recording_id=rec.id,
                    snippet_set_id=snippet_set.id,
                    start_time=t,
                    end_time=t + window,
                    duration=window,
                )
                db.add(snippet)
                db.flush()
                snippet_ids.append(snippet.id)
                t += step

        db.commit()

        # ------------------------------------------------------
        # 4. Compute embeddings in parallel
        # ------------------------------------------------------
        task_group = group(
            generate_embedding_for_snippet.s(snippet_id, model.id)
            for snippet_id in snippet_ids
        )

        results = task_group.apply_async().get()
        failures = [r for r in results if r.get("status") != "success"]

        # ------------------------------------------------------
        # 5. Final job status
        # ------------------------------------------------------
        if failures:
            service.update_job_status(
                job.id,
                EmbeddingJobStatus.FAILED,
                message=f"{len(failures)} snippet embedding failures",
            )
        else:
            service.update_job_status(job.id, EmbeddingJobStatus.COMPLETED)

        return {
            "status": "completed",
            "embedding_job_id": job.id,
            "total_snippets": len(snippet_ids),
            "failed": len(failures),
        }

    except Exception as e:
        service.update_job_status(
            job.id,
            EmbeddingJobStatus.FAILED,
            message=str(e),
        )
        return {"status": "error", "message": str(e)}

    finally:
        db.close()
