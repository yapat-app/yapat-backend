"""
Celery tasks for the embedding pipeline (SnippetSet architecture).
"""
import logging
import os
import time
from typing import Dict, List, Union

from celery import chord

from app.celery_app import celery_app
from app.config import settings
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
from app.schemas.visualisation import FPVDatasetRequest
from benchmarks.stage_timer import stage_timer, write_csv_row

logger = logging.getLogger(__name__)


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

        # --- Fetch dataset recordings (only the columns segmentation needs) ---
        recordings = db.query(
            Recording.id, Recording.duration
        ).filter(
            Recording.dataset_id == job.dataset_id
        ).all()

        # --- Check if snippets already exist for this snippet_set ---
        existing_snippets = (
            db.query(
                Snippet.id, Snippet.recording_id, Snippet.start_time, Snippet.end_time
            )
            .filter(Snippet.snippet_set_id == snippet_set.id)
            .all()
        )

        # Create a dictionary mapping (recording_id, start_time, end_time) -> snippet_id for O(1) lookup
        existing_snippet_map = {
            (s.recording_id, s.start_time, s.end_time): s.id
            for s in existing_snippets
        }

        # --- Segmentation: Create snippets grouped by recording (only if they don't exist) ---
        # Snippet objects for keys not already in existing_snippet_map are collected and
        # flushed once at the end (instead of once per snippet) so a dataset with tens of
        # thousands of recordings doesn't pay for a DB round-trip per snippet. Slots hold
        # either an existing snippet_id (int) or the pending Snippet object itself, which is
        # resolved to a real id right after the single flush below.
        recording_snippet_map: Dict[int, List[Union[int, Snippet]]] = {}
        new_snippets: List[Snippet] = []
        total_snippets = 0

        with stage_timer("snippet_gen", "cpu", str(job.dataset_id)) as _snip_timer:
            for rec in recordings:
                duration = rec.duration or 0.0
                t = 0.0
                recording_slots: List[Union[int, Snippet]] = []

                while t + window <= duration:
                    snippet_key = (rec.id, t, t + window)

                    existing_id = existing_snippet_map.get(snippet_key)
                    if existing_id is not None:
                        # Reuse existing snippet
                        recording_slots.append(existing_id)
                    else:
                        # Create new snippet only if it doesn't exist
                        snippet = Snippet(
                            recording_id=rec.id,
                            snippet_set_id=snippet_set.id,
                            start_time=t,
                            end_time=t + window,
                            duration=window,
                        )
                        new_snippets.append(snippet)
                        recording_slots.append(snippet)
                        # Avoid creating duplicates for the same key within this run
                        existing_snippet_map[snippet_key] = snippet

                    t += step

                if recording_slots:
                    recording_snippet_map[rec.id] = recording_slots
                    total_snippets += len(recording_slots)

            # Single batched round-trip for every new snippet in this run instead of
            # one flush per snippet.
            if new_snippets:
                db.add_all(new_snippets)
                db.flush()

            # Resolve pending Snippet objects to their now-populated ids.
            for rec_id, slots in recording_snippet_map.items():
                recording_snippet_map[rec_id] = [
                    slot.id if isinstance(slot, Snippet) else slot for slot in slots
                ]

            _snip_timer.n = total_snippets

        db.commit()

        # --- SnippetSet stays PENDING here on purpose ---
        # Segmentation completing does not mean embeddings exist yet — the actual
        # per-recording embedding generation happens asynchronously below via the
        # chunked chord, and can fail/be killed partway through. Marking READY this
        # early made `is_ready_for_feed` (and the "Generate Embeddings" UI) lie about
        # datasets whose embedding job never actually finished. The real READY/FAILED
        # transition now happens in finalize_embedding_job, once every chunk's result
        # is actually known.

        # --- Set as default if no default exists ---
        dataset = db.query(Dataset).filter_by(id=job.dataset_id).first()
        if dataset and dataset.default_snippet_set_id is None:
            dataset.default_snippet_set_id = snippet_set.id
            db.commit()

        # --- One chord child per CHUNK of recordings instead of per recording ---
        # A dataset with tens of thousands of recordings would otherwise dispatch tens of
        # thousands of individual chord entries, which is dominated by Redis chord-counter
        # and broker overhead rather than actual worker concurrency (celery-worker runs a
        # small fixed --concurrency). generate_embeddings_for_recording.chunks(...) batches
        # several recordings into each task message; each chunk task still calls the same
        # per-recording function once per recording, in-process.
        chunked_group = generate_embeddings_for_recording.chunks(
            (
                (recording_id, snippet_ids, model.id, job.id)
                for recording_id, snippet_ids in recording_snippet_map.items()
            ),
            settings.EMBEDDING_CHORD_CHUNK_SIZE,
        ).group()

        # Use a chord to finalize after all recording tasks complete
        finalize = finalize_embedding_job.s(job.id)

        chord(chunked_group)(finalize)

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
        # Each chord child is now a chunk of several recordings (see run_embedding), so
        # `results` is a list of per-chunk result lists rather than a flat list of
        # per-recording dicts. Flatten before inspecting status.
        flat_results = []
        for r in results:
            if isinstance(r, list):
                flat_results.extend(r)
            else:
                flat_results.append(r)

        failures = [r for r in flat_results if r.get("status") != "success"]

        job = db.query(EmbeddingJob).filter_by(id=embedding_job_id).first()
        snippet_set = job.snippet_set if job is not None else None

        if failures:
            service.update_job_status(
                embedding_job_id,
                EmbeddingJobStatus.FAILED,
                message=f"{len(failures)} snippet failures",
            )
            # Embeddings didn't actually finish — don't let the SnippetSet claim
            # READY regardless of how far segmentation got in run_embedding.
            if snippet_set is not None:
                snippet_set.status = SnippetSetStatus.FAILED
                db.commit()
        else:
            service.update_job_status(
                embedding_job_id,
                EmbeddingJobStatus.COMPLETED
            )
            # Only now, with every chunk confirmed successful, is the SnippetSet
            # genuinely ready for feed generation.
            if snippet_set is not None:
                snippet_set.status = SnippetSetStatus.READY
                db.commit()
            if job is not None:
                if job.started_at and job.completed_at:
                    elapsed = (job.completed_at - job.started_at).total_seconds()
                    write_csv_row({
                        "operation": "embedding",
                        "device": os.getenv("PAM_DEFAULT_DEVICE", "cpu"),
                        "dataset": str(job.dataset_id),
                        "N": None,
                        "time_s": round(elapsed, 3),
                        "peak_mem_mb": None,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })

                from app.services.pam_al._embedding_cache import invalidate_embedding_cache

                invalidate_embedding_cache(job.snippet_set_id, job.embedding_model_id)
                # Trigger dataset-level FPV generation asynchronously. This makes projections
                # available instantly on the Active Learning page and decouples them from inference.
                # Skip datasets too large for the current no-subsampling pipeline so we don't
                # auto-queue a job that would OOM the worker right after embedding completes.
                from app.services.visualisation_service import count_fpv_points

                n_fpv = count_fpv_points(db, job.dataset_id, job.embedding_model_id)
                if n_fpv > settings.FPV_MAX_POINTS:
                    logger.warning(
                        "Skipping auto FPV generation for dataset_id=%s: %s snippets "
                        "exceeds FPV_MAX_POINTS=%s (projections over very large datasets "
                        "are not yet supported).",
                        job.dataset_id, n_fpv, settings.FPV_MAX_POINTS,
                    )
                else:
                    generate_fpv_for_dataset.delay(
                        dataset_id=job.dataset_id,
                        embedding_model_id=job.embedding_model_id,
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


@celery_app.task(bind=True, name="app.tasks.embedding.generate_fpv_for_dataset")
def generate_fpv_for_dataset(self, dataset_id: int, embedding_model_id: int, run_3d: bool = False):
    """
    Compute and cache dataset-level FPV projections from EmbeddingVector.
    Stored in fpv_vis with model_checkpoint_id = NULL and embedding_model_id set.

    run_3d defaults to False to match the historical auto-trigger behavior
    (called from finalize_embedding_job above); the manual-generation API
    endpoint passes through whatever the caller actually requested.
    """
    db = SessionLocal()
    try:
        from app.services.visualisation_service import VISService

        service = VISService(db)
        body = FPVDatasetRequest(
            dataset_id=dataset_id,
            embedding_model_id=embedding_model_id,
            run_3d=run_3d,
        )
        service.generate_fpv_for_dataset_embeddings(body)
        return {"status": "success", "dataset_id": dataset_id, "embedding_model_id": embedding_model_id}
    except Exception as e:
        return {"status": "error", "dataset_id": dataset_id, "embedding_model_id": embedding_model_id, "message": str(e)}
    finally:
        db.close()
