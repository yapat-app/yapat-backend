"""
Snippet retrieval and utility service
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.annotation import Annotation
from app.models.recording import Recording


class SnippetService:
    """Service for querying snippet metadata"""

    def __init__(self, db: Session):
        self.db = db

    def list_snippets(
        self,
        dataset_id: int,
        snippet_config_id: int,
        recording_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Snippet]:
        """
        Retrieve snippets belonging to a dataset + snippet configuration.

        Optionally restrict to a single recording.
        """
        query = (
            self.db.query(Snippet)
            .join(Snippet.recording)
            .filter(Recording.dataset_id == dataset_id)
            .filter(Snippet.snippet_config_id == snippet_config_id)
        )

        if recording_id is not None:
            query = query.filter(Snippet.recording_id == recording_id)

        return query.offset(skip).limit(limit).all()

    def annotation_count(self, snippet_id: int) -> int:
        """Return number of annotations for a snippet."""
        return (
            self.db.query(Annotation)
            .filter(Annotation.snippet_id == snippet_id)
            .count()
        )

    def get_similar_snippets(
        self,
        snippet_id: int,
        embedding_model_id: int,
        limit: int = 10,
    ) -> List[Snippet]:
        """
        Retrieve similar snippets based on embedding vectors.

        This function delegates to an embedding/vector search backend.
        Snippets no longer store embeddings themselves.
        """
        # TODO: Replace with EmbeddingService.get_similar(snippet_id, embedding_model_id)
        return []
