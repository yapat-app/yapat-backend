"""
Celery tasks for the embedding pipeline (SnippetSet architecture).
"""
import os
from typing import List

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
# Generate embeddings for all snippets in a recording (BATCHED)
# ----------------------------------------------------------------------
@celery_app.task(bind=True, name="app.tasks.embedding.generate_embeddings_for_recording")
def generate_embeddings_for_recording(self, recording_id: int, snippet_ids: List[int], model_id: int, job_id: int):
    """
    Generate embeddings for all snippets in a recording using batch processing.
    
    Args:
        recording_id: Recording to process
        snippet_ids: List of snippet IDs to embed (must all belong to this recording)
        model_id: Embedding model ID
        job_id: EmbeddingJob ID for tracking
        
    Returns:
        dict with status and stats
    """
    db = SessionLocal()
    try:
        # --- Load recording ---
        recording = db.query(Recording).filter_by(id=recording_id).first()
        if not recording:
            return {
                "status": "error",
                "recording_id": recording_id,
                "message": "recording_not_found"
            }
        
        # --- Load all snippets for this recording ---
        snippets = (
            db.query(Snippet)
            .filter(Snippet.id.in_(snippet_ids))
            .filter(Snippet.recording_id == recording_id)
            .order_by(Snippet.start_time)
            .all()
        )
        
        if not snippets:
            return {
                "status": "success",
                "recording_id": recording_id,
                "snippets_processed": 0,
                "message": "no_snippets_found"
            }
        
        # --- Prepare snippet windows ---
        snippet_windows = [(s.start_time, s.end_time) for s in snippets]
        
        # --- Batch embedding with BirdNET ---
        from app.services.birdnet_model import BirdNetEmbedder
        
        audio_path = os.path.join(os.getenv("DATA_ROOT", "/data"), recording.file_path)
        embeddings = BirdNetEmbedder.embed_batch_from_recording(audio_path, snippet_windows)
        
        # --- Prepare bulk insert data ---
        bulk_data = []
        failed_count = 0
        
        for snippet, embedding in zip(snippets, embeddings):
            if embedding is None:
                failed_count += 1
                continue
            
            bulk_data.append({
                "snippet_id": snippet.id,
                "job_id": job_id,
                "model_id": model_id,
                "vector": embedding,
            })
        
        # --- Bulk insert embeddings ---
        inserted_count = 0
        if bulk_data:
            vector_store = VectorStore(db)
            inserted_count = vector_store.bulk_insert(bulk_data)
        
        return {
            "status": "success",
            "recording_id": recording_id,
            "snippets_processed": len(snippets),
            "embeddings_inserted": inserted_count,
            "failed_snippets": failed_count,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "recording_id": recording_id,
            "message": str(e)
        }
    
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

        # --- Segmentation: Create snippets grouped by recording ---
        recording_snippet_map = {}  # recording_id -> list of snippet_ids
        total_snippets = 0

        for rec in recordings:
            duration = rec.duration or 0.0
            t = 0.0
            recording_snippets = []

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
                recording_snippets.append(snippet.id)
                t += step

            if recording_snippets:
                recording_snippet_map[rec.id] = recording_snippets
                total_snippets += len(recording_snippets)

        db.commit()

        # --- Mark SnippetSet as READY after snippet materialization ---
        snippet_set.status = SnippetSetStatus.READY
        db.commit()

        # --- Set as default if no default exists ---
        dataset = db.query(Dataset).filter_by(id=job.dataset_id).first()
        if dataset and dataset.default_snippet_set_id is None:
            dataset.default_snippet_set_id = snippet_set.id
            db.commit()

        # --- OPTIMIZED: One task per RECORDING instead of per snippet ---
       
        task_group = group(
            generate_embeddings_for_recording.s(
                recording_id=recording_id,
                snippet_ids=snippet_ids,
                model_id=model.id,
                job_id=job.id
            )
            for recording_id, snippet_ids in recording_snippet_map.items()
        )

        # Use a chord to finalize after all recording tasks complete
        finalize = finalize_embedding_job.s(job.id)

        chord(task_group)(finalize)

        return {
            "status": "scheduled",
            "embedding_job_id": job.id,
            "total_recordings": len(recording_snippet_map),
            "total_snippets": total_snippets,
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
