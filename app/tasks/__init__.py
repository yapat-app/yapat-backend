"""
Celery tasks package
"""

from app.tasks.embedding_tasks import (
    generate_embedding_for_snippet,
    generate_embeddings_batch,
    regenerate_embeddings_for_dataset,
)
from app.tasks.processing_tasks import (
    scan_dataset,
    generate_snippets,
    process_dataset,
)

__all__ = [
    # Embedding tasks
    "generate_embedding_for_snippet",
    "generate_embeddings_batch",
    "regenerate_embeddings_for_dataset",

    # Processing tasks (new architecture)
    "scan_dataset",
    "generate_snippets",
    "process_dataset",
]
