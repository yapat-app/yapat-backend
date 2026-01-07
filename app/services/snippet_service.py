"""
Snippet retrieval and utility service (for SnippetSet-based architecture)
"""

import random
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.annotation import Annotation
from app.models.recording import Recording
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.models.dataset import Dataset


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

    # ---------------------------------------------------------
    # Feed Generation
    # ---------------------------------------------------------

    def _resolve_and_validate_snippet_set(
        self,
        dataset_id: Optional[int],
        snippet_set_id: Optional[int],
    ) -> int:
        """
        Resolve snippet_set_id from parameters or dataset default, and validate it's READY.
        
        Args:
            dataset_id: Optional dataset ID (required if snippet_set_id not provided)
            snippet_set_id: Optional snippet set ID (if not provided, uses dataset's default)
            
        Returns:
            Resolved snippet_set_id (int)
            
        Raises:
            ValueError: If dataset_id is required but not provided, SnippetSet not found,
                       has no default, or is not READY
        """
        # Resolve snippet_set_id
        if snippet_set_id is None:
            if dataset_id is None:
                raise ValueError("Either dataset_id or snippet_set_id must be provided")
            # Get default SnippetSet from dataset
            dataset = self.db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset(id={dataset_id}) not found")
            if dataset.default_snippet_set_id is None:
                raise ValueError(f"Dataset(id={dataset_id}) has no default SnippetSet")
            snippet_set_id = dataset.default_snippet_set_id
        
        # Validate SnippetSet is READY
        snippet_set = self.db.query(SnippetSet).filter_by(id=snippet_set_id).first()
        if not snippet_set:
            raise ValueError(f"SnippetSet(id={snippet_set_id}) not found")
        if snippet_set.status != SnippetSetStatus.READY:
            raise ValueError(
                f"SnippetSet(id={snippet_set_id}) is not READY (status: {snippet_set.status.value})"
            )
        
        return snippet_set_id

    def get_feed(
        self,
        dataset_id: Optional[int] = None,
        snippet_set_id: Optional[int] = None,
        recording_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Snippet]:
        """
        Get feed of snippets for annotation workflow.
        
        Returns random snippets as a placeholder.
        Optionally filters by dataset_id, snippet_set_id, and/or recording_id.
        
        Args:
            dataset_id: Optional dataset ID to filter snippets (required if snippet_set_id not provided)
            snippet_set_id: Optional snippet set ID (if not provided, uses dataset's default)
            recording_id: Optional recording ID to filter snippets
            skip: Number of snippets to skip (for pagination)
            limit: Maximum number of snippets to return
            
        Returns:
            List of Snippet objects in random order
            
        Raises:
            ValueError: If dataset_id is required but not provided, or if SnippetSet is not READY
        """
        # Resolve and validate snippet_set_id
        snippet_set_id = self._resolve_and_validate_snippet_set(dataset_id, snippet_set_id)
        
        # Build base query
        query = (
            self.db.query(Snippet)
            .join(Snippet.recording)
            .join(Snippet.snippet_set)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )
        
        # Filter by dataset_id if provided (for validation)
        if dataset_id is not None:
            query = query.filter(SnippetSet.dataset_id == dataset_id)
        
        # Filter by recording_id if provided
        if recording_id is not None:
            query = query.filter(Snippet.recording_id == recording_id)
        
        # Return random snippets as placeholder
        all_snippets = query.all()
        random.shuffle(all_snippets)
        
        # Apply pagination
        return all_snippets[skip:skip + limit]

    def get_feed_random(
        self,
        dataset_id: Optional[int] = None,
        snippet_set_id: Optional[int] = None,
        recording_id: Optional[int] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Snippet]:
        """
        Randomly sample snippets from the dataset without bias.
        
        Useful for initial exploration or broad manual labeling before model initialization.
        
        Args:
            dataset_id: Optional dataset ID to filter snippets (required if snippet_set_id not provided)
            snippet_set_id: Optional snippet set ID (if not provided, uses dataset's default)
            recording_id: Optional recording ID to filter snippets
            status: Optional filter by snippet status (ignored for now)
            skip: Number of snippets to skip (for pagination)
            limit: Maximum number of snippets to return (default 50)
            
        Returns:
            List of Snippet objects in random order
            
        Raises:
            ValueError: If dataset_id is required but not provided, or if SnippetSet is not READY
        """
        # Resolve and validate snippet_set_id
        snippet_set_id = self._resolve_and_validate_snippet_set(dataset_id, snippet_set_id)
        
        # Build base query
        query = (
            self.db.query(Snippet)
            .join(Snippet.recording)
            .join(Snippet.snippet_set)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )
        
        # Filter by dataset_id if provided (for validation)
        if dataset_id is not None:
            query = query.filter(SnippetSet.dataset_id == dataset_id)
        
        # Filter by recording_id if provided
        if recording_id is not None:
            query = query.filter(Snippet.recording_id == recording_id)
        
        # Return random snippets
        import random
        all_snippets = query.all()
        random.shuffle(all_snippets)
        
        # Apply pagination
        return all_snippets[skip:skip + limit]

    def get_feed_similarity(
        self,
        dataset_id: int,
        snippet_set_id: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        query_snippet_id: Optional[int] = None,
        embedding_model_id: Optional[int] = None,
        crop_start_sec: Optional[float] = None,
        crop_end_sec: Optional[float] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Snippet]:
        """
        Find snippets acoustically similar to a query example.
        
        Uses cosine similarity in the embedding space. The query is embedded on the fly
        and is ephemeral (not stored).
        
        Args:
            dataset_id: Required dataset ID
            snippet_set_id: Optional snippet set ID (if not provided, uses dataset's default)
            query_embedding: Optional pre-computed embedding vector for the query
            query_snippet_id: Optional snippet ID to use as query (alternative to query_embedding)
            embedding_model_id: Optional embedding model ID (defaults to dataset's current model)
            crop_start_sec: Optional crop start time for query audio
            crop_end_sec: Optional crop end time for query audio
            skip: Number of snippets to skip (for pagination)
            limit: Maximum number of similar snippets to return (default 50)
            
        Returns:
            List of Snippet objects ranked by similarity to query
            
        Raises:
            ValueError: If SnippetSet is not READY
            
        TODO: Implement embedding-based similarity search
        """
        # Resolve and validate snippet_set_id
        snippet_set_id = self._resolve_and_validate_snippet_set(dataset_id, snippet_set_id)
        
        # Placeholder implementation - delegates to existing method if query_snippet_id provided
        if query_snippet_id:
            return self.get_similar_snippets(
                snippet_id=query_snippet_id,
                embedding_model_id=embedding_model_id or 1,  # TODO: Get default from dataset
                limit=limit
            )
        return []
