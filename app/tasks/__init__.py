"""
Celery tasks package
"""

from app.tasks.embedding_tasks import (
    run_embedding,
    generate_embedding_for_snippet,
)
from app.tasks.processing_tasks import (
    scan_dataset,
    process_dataset,
)

__all__ = [
    # Embedding pipeline
    "run_embedding",
    "generate_embedding_for_snippet",

    # Processing tasks
    "scan_dataset",
    "process_dataset",
]
