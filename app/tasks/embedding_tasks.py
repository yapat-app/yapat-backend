"""
Celery tasks for the embedding pipeline.

This implements the new architecture:

    run_embedding(embedding_job_id):
        - loads job + model + snippet_config
        - segments all recordings for the dataset
        - creates snippet rows
        - generates embeddings for each snippet
        - marks job COMPLETE or FAILED
"""

from typing import List
from celery import group
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import SessionLocal

from app.models.snippet import Snippet, SnippetConfig
from app.models.recording import Recording
from app.models.embedding import EmbeddingJob, EmbeddingModel, EmbeddingJobStatus

from app.services.embedding_service import EmbeddingService


# ----------------------------------------------------------------------
# Helper: DB session
# ----------------------------------------------------------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # closed in tasks


# ----------------------------------------------------------------------
# Core task: generate embedding for a single snippet
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.generate_embedding_for_snippet")
def generate_embedding_for_snippet(self, snippet_id: int, model_id: int):
    """
    Generate embedding for a single snippet.
    """
    db = SessionLocal()
    try:
        snippet = db.query(Snippet).filter_by(id=snippet_id).first()
        if not snippet:
            return {"status": "error", "snippet_id": snippet_id, "message": "Not found"}

        model = db.query(EmbeddingModel).filter_by(id=model_id).first()
        if not model:
            return {"status": "error", "message": f"EmbeddingModel(id={model_id}) not found"}

        # TODO: Actual embedding code
        dummy_embedding = [0.1, 0.2, 0.3]  # placeholder

        snippet.embedding = dummy_embedding
        db.commit()

        return {"status": "success", "snippet_id": snippet_id}

    except Exception as e:
        db.rollback()
        return {"status": "error", "snippet_id": snippet_id, "message": str(e)}

    finally:
        db.close()


# ----------------------------------------------------------------------
# Main pipeline: run_embedding(job)
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.run_embedding")
def run_embedding(self, embedding_job_id: int):
    """
    Run the entire embedding pipeline for an EmbeddingJob:
        - update job status
        - segment recordings
        - create snippets
        - generate embeddings for each snippet (parallel group)
    """

    db = SessionLocal()
    service = EmbeddingService(db)

    try:
        # ----------------------------------------------------------
        # 1. Load job and mark RUNNING
        # ----------------------------------------------------------
        job = db.query(EmbeddingJob).filter_by(id=embedding_job_id).first()
        if not job:
            raise ValueError(f"EmbeddingJob(id={embedding_job_id}) not found")

        service.update_job_status(job.id, EmbeddingJobStatus.RUNNING, celery_task_id=self.request.id)

        dataset = job.dataset
        config = job.snippet_config
        model = job.embedding_model

        window = config.window_size
        step = config.step_size
        overlap = config.overlap   # might be redundant but kept for clarity

        # ----------------------------------------------------------
        # 2. Load recordings for the dataset
        # ----------------------------------------------------------
        recordings = (
            db.query(Recording)
            .filter_by(dataset_id=dataset.id)
            .all()
        )

        snippet_ids = []

        # ----------------------------------------------------------
        # 3. Segment each recording → create Snippet rows
        # ----------------------------------------------------------
        for rec in recordings:
            duration = rec.duration_seconds
            if duration is None:
                # You may want to compute it from the audio or skip
                continue

            t = 0.0
            while t + window <= duration:
                snippet = Snippet(
                    recording_id=rec.id,
                    embedding_job_id=job.id,
                    start_time=t,
                    end_time=t + window,
                )
                db.add(snippet)
                db.flush()
                snippet_ids.append(snippet.id)
                t += step

        db.commit()

        # ----------------------------------------------------------
        # 4. Generate embeddings (parallel)
        # ----------------------------------------------------------
        task_group = group(
            generate_embedding_for_snippet.s(snippet_id, model.id)
            for snippet_id in snippet_ids
        )

        results = task_group.apply_async().get()

        # ----------------------------------------------------------
        # 5. Final job status
        # ----------------------------------------------------------
        failures = [r for r in results if r.get("status") != "success"]

        if failures:
            service.update_job_status(
                job.id,
                EmbeddingJobStatus.FAILED,
                message=f"{len(failures)} snippet embedding failures"
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
            embedding_job_id,
            EmbeddingJobStatus.FAILED,
            message=str(e)
        )
        return {"status": "error", "message": str(e)}

    finally:
        db.close()
