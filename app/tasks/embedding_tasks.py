"""
Celery tasks for embedding generation and similarity computation
"""

from typing import List, Optional
from celery import group, chord
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.snippet import Snippet
from app.models.recording import Recording
from app.services.embedding_service import EmbeddingService


def get_db():
    """Get database session for tasks"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Will be closed in task


@celery_app.task(bind=True, name="app.tasks.embedding_tasks.generate_embedding_for_snippet")
def generate_embedding_for_snippet(self, snippet_id: int, embedding_model: Optional[str] = None):
    """
    Generate embedding for a single snippet
    
    Args:
        snippet_id: ID of the snippet to process
        embedding_model: Optional model name to use
        
    Returns:
        dict: Result with snippet_id and status
    """
    db = SessionLocal()
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'snippet_id': snippet_id})
        
        # Get snippet
        snippet = db.query(Snippet).filter(Snippet.id == snippet_id).first()
        if not snippet:
            return {"status": "error", "message": f"Snippet {snippet_id} not found"}
        
        # Initialize embedding service
        embedding_service = EmbeddingService(model_name=embedding_model)
        
        # TODO: In actual implementation, this would:
        # 1. Load audio file from snippet.file_path or recording
        # 2. Generate embedding using the model
        # 3. Store embedding vector in snippet.embedding
        
        # Placeholder implementation
        embedding = embedding_service.generate_embedding(b"dummy_audio_data")
        snippet.embedding = embedding
        db.commit()
        
        return {
            "status": "success",
            "snippet_id": snippet_id,
            "embedding_dim": len(embedding)
        }
    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "snippet_id": snippet_id,
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.embedding_tasks.generate_embeddings_batch")
def generate_embeddings_batch(self, snippet_ids: List[int], embedding_model: Optional[str] = None):
    """
    Generate embeddings for a batch of snippets in parallel
    
    Args:
        snippet_ids: List of snippet IDs to process
        embedding_model: Optional model name to use
        
    Returns:
        dict: Summary of batch processing
    """
    self.update_state(
        state='PROCESSING',
        meta={'total': len(snippet_ids), 'processed': 0}
    )
    
    # Create parallel tasks for each snippet
    job = group(
        generate_embedding_for_snippet.s(snippet_id, embedding_model)
        for snippet_id in snippet_ids
    )
    
    result = job.apply_async()
    results = result.get()
    
    # Summarize results
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    return {
        "status": "completed",
        "total": len(snippet_ids),
        "successful": successful,
        "failed": failed,
        "results": results
    }


@celery_app.task(bind=True, name="app.tasks.embedding_tasks.regenerate_embeddings_for_dataset")
def regenerate_embeddings_for_dataset(self, dataset_id: int, embedding_model: Optional[str] = None):
    """
    Regenerate embeddings for all snippets in a dataset
    
    Args:
        dataset_id: ID of the dataset
        embedding_model: Optional model name to use
        
    Returns:
        dict: Summary of regeneration process
    """
    db = SessionLocal()
    try:
        self.update_state(
            state='PROCESSING',
            meta={'dataset_id': dataset_id, 'status': 'fetching_snippets'}
        )
        
        # Get all snippets for the dataset through recordings
        snippets = db.query(Snippet).join(Recording).filter(
            Recording.dataset_id == dataset_id
        ).all()
        
        snippet_ids = [s.id for s in snippets]
        
        if not snippet_ids:
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "message": "No snippets found in dataset",
                "total": 0
            }
        
        # Trigger batch processing
        result = generate_embeddings_batch.delay(snippet_ids, embedding_model)
        
        return {
            "status": "started",
            "dataset_id": dataset_id,
            "total_snippets": len(snippet_ids),
            "batch_task_id": result.id
        }
    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "message": str(e)
        }
    finally:
        db.close()

