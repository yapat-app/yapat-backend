"""
Celery tasks package
"""

from app.tasks.embedding_tasks import (
    generate_embedding_for_snippet,
    generate_embeddings_batch,
    regenerate_embeddings_for_dataset,
)
from app.tasks.processing_tasks import (
    process_recording,
    generate_snippets_for_recording,
    scan_and_process_dataset,
)

__all__ = [
    # Embedding tasks
    "generate_embedding_for_snippet",
    "generate_embeddings_batch",
    "regenerate_embeddings_for_dataset",
    # Processing tasks
    "process_recording",
    "generate_snippets_for_recording",
    "scan_and_process_dataset",
]

