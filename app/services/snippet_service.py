"""
Snippet retrieval and utility service (for SnippetSet-based architecture)
"""

from typing import List, Optional
from sqlalchemy import func
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
    # Similarity Search
    # ---------------------------------------------------------

    def get_similar_snippets(
        self,
        snippet_id: int,
        embedding_model_id: int,
        limit: int = 10,
    ) -> List[Snippet]:
        """
        Retrieve similar snippets based on embedding vectors using cosine similarity.

        Args:
            snippet_id: The ID of the query snippet
            embedding_model_id: The embedding model ID to use for similarity search
            limit: Maximum number of similar snippets to return (default 10)

        Returns:
            List of Snippet objects ranked by similarity (most similar first)
            
        Raises:
            ValueError: If query snippet not found or has no embedding vector
        """
        from app.services.embedding_service import VectorStore
        
        # Validate query snippet exists
        query_snippet = self.db.query(Snippet).filter_by(id=snippet_id).first()
        if not query_snippet:
            raise ValueError(f"Query snippet with id={snippet_id} not found")
        
        # Get query vector
        vector_store = VectorStore(self.db)
        query_vector_obj = vector_store.get(snippet_id=snippet_id, model_id=embedding_model_id)
        
        if not query_vector_obj:
            raise ValueError(
                f"No embedding vector found for snippet_id={snippet_id} "
                f"with embedding_model_id={embedding_model_id}"
            )
        
        # Perform similarity search (returns list of (snippet_id, similarity_score) tuples)
        # Request limit + 1 to exclude the query snippet itself if it appears in results
        similar_results = vector_store.search(
            model_id=embedding_model_id,
            query_vector=query_vector_obj.vector,
            k=limit + 1
        )
        
        if not similar_results:
            return []
        
        # Extract snippet IDs and filter out the query snippet itself
        similar_snippet_ids = [
            sid for sid, score in similar_results 
            if sid != snippet_id
        ][:limit]
        
        if not similar_snippet_ids:
            return []
        
        # Fetch snippets in the order of similarity
        # Build a mapping for efficient ordering
        id_to_snippet = {
            s.id: s 
            for s in self.db.query(Snippet).filter(Snippet.id.in_(similar_snippet_ids)).all()
        }
        
        # Return snippets in similarity order
        return [id_to_snippet[sid] for sid in similar_snippet_ids if sid in id_to_snippet]

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

        return query.order_by(func.random()).offset(skip).limit(limit).all()

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

        return query.order_by(func.random()).offset(skip).limit(limit).all()

    def get_feed_filter(
        self,
        dataset_id: Optional[int] = None,
        snippet_set_id: Optional[int] = None,
        recording_id: Optional[int] = None,
        annotation_status: Optional[str] = None,
        location: Optional[str] = None,
        user_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Snippet]:
        """
        Return a randomly-ordered, SQL-filtered feed.

        Args:
            dataset_id: Optional dataset ID (required when snippet_set_id not given)
            snippet_set_id: Optional snippet set ID (uses dataset default if absent)
            recording_id: Optional recording ID to restrict snippets to one recording
            annotation_status: "annotated" | "unannotated" | "any" (default).
                               Annotation scope is any user annotation on the snippet.
            location: Optional site/locality filter. Supports one value or a comma-separated
                      list for multi-select filtering.
            user_id: Unused for annotation_status filtering (kept for API compatibility).
            skip: Pagination offset
            limit: Maximum snippets to return (default 50)

        Returns:
            List of Snippet objects in random order

        Raises:
            ValueError: If parameters are invalid or SnippetSet is not READY
        """
        from sqlalchemy import cast, func, exists, select, String

        snippet_set_id = self._resolve_and_validate_snippet_set(dataset_id, snippet_set_id)

        query = (
            self.db.query(Snippet)
            .join(Snippet.recording)
            .join(Snippet.snippet_set)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )

        if dataset_id is not None:
            query = query.filter(SnippetSet.dataset_id == dataset_id)

        if recording_id is not None:
            query = query.filter(Snippet.recording_id == recording_id)

        if location is not None and location.strip():
            loc_values = [p.strip() for p in location.split(",") if p.strip()]
            if not loc_values:
                loc_values = []
            # Parse & persist location from file names for rows that still lack it
            backfill_dataset_id = dataset_id
            if backfill_dataset_id is None:
                ss_row = (
                    self.db.query(SnippetSet)
                    .filter(SnippetSet.id == snippet_set_id)
                    .first()
                )
                if ss_row is not None:
                    backfill_dataset_id = ss_row.dataset_id
            if backfill_dataset_id is not None:
                from app.services.dataset_service import DatasetService

                DatasetService(self.db).backfill_recording_locations(
                    backfill_dataset_id
                )

            bind = self.db.get_bind()
            dialect = bind.dialect.name
            # PG: cast(json['k'], String) is wrong for string JSON values; use ->> for text.
            if loc_values:
                if dialect == "postgresql":
                    query = query.filter(
                        Recording.extra_metadata.op("->>")("location").in_(loc_values)
                    )
                elif dialect == "sqlite":
                    query = query.filter(
                        func.json_extract(Recording.extra_metadata, "$.location")
                        .in_(loc_values)
                    )
                else:
                    query = query.filter(
                        cast(Recording.extra_metadata["location"], String).in_(loc_values)
                    )

        # Apply annotation status filter based on any annotation on the snippet
        status = (annotation_status or "any").lower()
        if status in ("annotated", "unannotated"):
            ann_subq = (
                select(Annotation.id)
                .where(
                    Annotation.snippet_id == Snippet.id,
                )
                .correlate(Snippet)
                .exists()
            )
            if status == "annotated":
                query = query.filter(ann_subq)
            else:
                query = query.filter(~ann_subq)

        # Push randomisation + pagination into the DB — avoids loading the full set in Python
        return query.order_by(func.random()).offset(skip).limit(limit).all()

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
        
        Uses cosine similarity in the embedding space. Supports:
        - Query by snippet ID (uses existing embedding)
        - Query by pre-computed embedding vector
        - Query by cropped audio (TODO: future enhancement)
        
        Args:
            dataset_id: Required dataset ID to scope the search
            snippet_set_id: Optional snippet set ID (if not provided, uses dataset's default)
            query_embedding: Optional pre-computed embedding vector for the query
            query_snippet_id: Optional snippet ID to use as query (alternative to query_embedding)
            embedding_model_id: Optional embedding model ID (required if query_embedding provided)
            crop_start_sec: Optional crop start time for query audio (future enhancement)
            crop_end_sec: Optional crop end time for query audio (future enhancement)
            skip: Number of snippets to skip for pagination
            limit: Maximum number of similar snippets to return (default 50)
            
        Returns:
            List of Snippet objects ranked by similarity to query (paginated)
            
        Raises:
            ValueError: If parameters are invalid or SnippetSet is not READY
        """
        from app.services.embedding_service import VectorStore
        
        # Validate input: must provide either query_snippet_id or query_embedding
        if query_snippet_id is None and query_embedding is None:
            raise ValueError("Either query_snippet_id or query_embedding must be provided")
        
        if query_embedding is not None and query_snippet_id is not None:
            raise ValueError("Cannot provide both query_snippet_id and query_embedding")
        
        # Resolve and validate snippet_set_id
        snippet_set_id = self._resolve_and_validate_snippet_set(dataset_id, snippet_set_id)
        
        # Get the snippet_set to determine the embedding_model_id if not provided
        snippet_set = self.db.query(SnippetSet).filter_by(id=snippet_set_id).first()
        if not snippet_set:
            raise ValueError(f"SnippetSet(id={snippet_set_id}) not found")
        
        # Determine the embedding model to use
        if embedding_model_id is None:
            # Use the snippet_set's embedding model
            embedding_model_id = snippet_set.embedding_model_id
        
        # Get query vector
        vector_store = VectorStore(self.db)
        query_vector = None
        
        if query_snippet_id is not None:
            # Retrieve embedding for the query snippet
            query_vector_obj = vector_store.get(
                snippet_id=query_snippet_id,
                model_id=embedding_model_id
            )
            if not query_vector_obj:
                raise ValueError(
                    f"No embedding vector found for query_snippet_id={query_snippet_id} "
                    f"with embedding_model_id={embedding_model_id}"
                )
            query_vector = query_vector_obj.vector
        else:
            # Use provided query_embedding
            query_vector = query_embedding
        
        # Perform similarity search with dataset filtering for security and performance
        # Request skip + limit to get enough results for pagination
        search_limit = skip + limit + (1 if query_snippet_id else 0)  # +1 to exclude query itself
        
        similar_results = vector_store.search(
            model_id=embedding_model_id,
            query_vector=query_vector,
            k=search_limit,
            snippet_set_id=snippet_set_id  # Filter by snippet_set for efficiency and security
        )
        
        if not similar_results:
            return []
        
        # Extract snippet IDs and filter out the query snippet itself
        similar_snippet_ids = [
            sid for sid, score in similar_results
            if sid != query_snippet_id  # Filter query snippet if present
        ]
        
        if not similar_snippet_ids:
            return []
        
        # Filter snippets by snippet_set_id to ensure they belong to the correct dataset/snippet_set
        # This is critical for multi-tenant scenarios
        query = (
            self.db.query(Snippet)
            .filter(Snippet.id.in_(similar_snippet_ids))
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )
        
        # Fetch all matching snippets
        snippets = query.all()
        
        # Build a mapping for efficient ordering
        id_to_snippet = {s.id: s for s in snippets}
        
        # Return snippets in similarity order with pagination
        ordered_snippets = [
            id_to_snippet[sid] 
            for sid in similar_snippet_ids 
            if sid in id_to_snippet
        ]
        
        # Apply pagination
        return ordered_snippets[skip:skip + limit]
