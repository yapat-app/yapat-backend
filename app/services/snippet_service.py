"""
Snippet retrieval and utility service (for SnippetSet-based architecture)
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.annotation import Annotation
from app.models.recording import Recording
from app.models.embedding import SnippetSet


class SnippetService:
    """Service for querying snippet metadata and statistics."""

    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Snippet Listing
    # ---------------------------------------------------------

    def list_snippets(
        self,
        dataset_id: int,
        snippet_set_id: int,
        recording_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Snippet]:
        """
        Retrieve snippets belonging to a dataset and snippet_set.

        Optionally filter by recording.
        """

        query = (
            self.db.query(Snippet)
            .join(Snippet.recording)
            .join(Snippet.snippet_set)
            .filter(SnippetSet.dataset_id == dataset_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )

        if recording_id is not None:
            query = query.filter(Snippet.recording_id == recording_id)

        return query.order_by(Snippet.start_time).offset(skip).limit(limit).all()

    # ---------------------------------------------------------
    # Annotation Utility
    # ---------------------------------------------------------

    def annotation_count(self, snippet_id: int) -> int:
        """Return the number of annotations attached to a snippet."""
        return (
            self.db.query(Annotation)
            .filter(Annotation.snippet_id == snippet_id)
            .count()
        )

    # ---------------------------------------------------------
    # Similarity Search Placeholder
    # ---------------------------------------------------------

    def get_similar_snippets(
        self,
        snippet_id: int,
        embedding_model_id: int,
        limit: int = 10,
    ) -> List[Snippet]:
        """
        Retrieve similar snippets based on embedding vectors.

        Placeholder — real implementation will delegate
        to EmbeddingService or a vector-search backend.
        """
        return []
