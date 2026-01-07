"""
Celery tasks for the embedding pipeline (SnippetSet architecture).
"""
import os

from celery import chord, group

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.services.embedding_service import EmbeddingService, VectorStore


# from sqlalchemy.orm import Session


# ----------------------------------------------------------------------
# Generate embedding for a single snippet
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.generate_embedding_for_snippet")
def generate_embedding_for_snippet(self, snippet_id: int, model_id: int):
    db = SessionLocal()
    try:
        # --- Load snippet ---
        snippet = db.query(Snippet).filter_by(id=snippet_id).first()
        if not snippet:
            return {"status": "error", "snippet_id": snippet_id, "message": "snippet_not_found"}

        service = EmbeddingService(db)

        # --- BirdNET embedding ---
        from app.services.birdnet_model import BirdNetEmbedder

        vector = BirdNetEmbedder.embed(
            audio_path=os.path.join(os.getenv("DATA_ROOT", "/data"), snippet.recording.file_path),
            start_time=snippet.start_time
        )

        if vector is None:
            raise RuntimeError(f"BirdNET returned no vector for snippet {snippet_id}")

        # --- Find matching EmbeddingJob ---
        job = (
            db.query(EmbeddingJob)
            .filter_by(
                embedding_model_id=model_id,
                snippet_set_id=snippet.snippet_set_id,
            )
            .first()
        )
        if not job:
            raise RuntimeError(
                f"No EmbeddingJob found for snippet_set={snippet.snippet_set_id} model={model_id}"
            )

        # --- Store in vector store ---
        VectorStore(db).insert(
            snippet_id=snippet_id,
            job_id=job.id,
            model_id=model_id,
            vector=vector,
        )

        return {"status": "success", "snippet_id": snippet_id}

    except Exception as e:
        return {"status": "error", "snippet_id": snippet_id, "message": str(e)}

    finally:
        db.close()


# ----------------------------------------------------------------------
# Main embedding job
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.run_embedding")
def run_embedding(self, embedding_job_id: int):
    db = SessionLocal()
    service = EmbeddingService(db)

    try:
        # --- Load job ---
        job = db.query(EmbeddingJob).filter_by(id=embedding_job_id).first()
        if not job:
            raise ValueError(f"EmbeddingJob(id={embedding_job_id}) not found")

        # Mark job as running
        service.update_job_status(
            job.id,
            EmbeddingJobStatus.RUNNING,
            celery_task_id=self.request.id,
        )

        snippet_set: SnippetSet = job.snippet_set
        model: EmbeddingModel = job.embedding_model

        window = snippet_set.window_size
        step = snippet_set.step_size

        # --- Fetch dataset recordings ---
        recordings = db.query(Recording).filter(
            Recording.dataset_id == job.dataset_id
        ).all()

        snippet_ids = []

        # --- Segmentation ---
        for rec in recordings:
            duration = rec.duration or 0.0
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

        # --- Mark SnippetSet as READY after snippet materialization ---
        snippet_set.status = SnippetSetStatus.READY
        db.commit()

        # --- Set as default if no default exists ---
        dataset = db.query(Dataset).filter_by(id=job.dataset_id).first()
        if dataset and dataset.default_snippet_set_id is None:
            dataset.default_snippet_set_id = snippet_set.id
            db.commit()

        # --- Parallel embedding tasks ---
        task_group = group(
            generate_embedding_for_snippet.s(snippet_id, model.id)
            for snippet_id in snippet_ids
        )

        # Use a chord instead of blocking join
        finalize = finalize_embedding_job.s(job.id)

        chord(task_group)(finalize)

        return {
            "status": "scheduled",
            "embedding_job_id": job.id,
            "total_snippets": len(snippet_ids),
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


@celery_app.task(bind=True, name="app.tasks.embedding.finalize_embedding_job")
def finalize_embedding_job(self, results, embedding_job_id):
    db = SessionLocal()
    service = EmbeddingService(db)

    try:
        failures = [r for r in results if r.get("status") != "success"]

        if failures:
            service.update_job_status(
                embedding_job_id,
                EmbeddingJobStatus.FAILED,
                message=f"{len(failures)} snippet failures",
            )
        else:
            service.update_job_status(
                embedding_job_id,
                EmbeddingJobStatus.COMPLETED
            )

    except Exception as e:
        service.update_job_status(
            embedding_job_id,
            EmbeddingJobStatus.FAILED,
            message=str(e)
        )
        raise

    finally:
        db.close()
