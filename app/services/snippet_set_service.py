"""
SnippetSet service — manages creation and retrieval of snippet sets.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.embedding import (
    EmbeddingModel,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.dataset import Dataset
from app.models.snippet import Snippet
from app.models.annotation import Annotation


class SnippetSetService:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Get / List
    # ---------------------------------------------------------

    def get(self, snippet_set_id: int) -> SnippetSet:
        ss = (
            self.db.query(SnippetSet)
            .filter(SnippetSet.id == snippet_set_id)
            .first()
        )
        if not ss:
            raise ValueError(f"SnippetSet(id={snippet_set_id}) not found")
        return ss

    def list_for_dataset(self, dataset_id: int) -> List[SnippetSet]:
        return (
            self.db.query(SnippetSet)
            .filter(SnippetSet.dataset_id == dataset_id)
            .order_by(SnippetSet.created_at)
            .all()
        )

    def list_for_model(
        self,
        dataset_id: int,
        embedding_model_id: int
    ) -> List[SnippetSet]:
        return (
            self.db.query(SnippetSet)
            .filter(
                SnippetSet.dataset_id == dataset_id,
                SnippetSet.embedding_model_id == embedding_model_id,
            )
            .order_by(SnippetSet.created_at)
            .all()
        )

    # ---------------------------------------------------------
    # Create
    # (Primarily called through EmbeddingService)
    # ---------------------------------------------------------

    def create(
        self,
        dataset: Dataset,
        model: EmbeddingModel,
        *,
        window_size: float,
        step_size: float,
        overlap: float,
    ) -> SnippetSet:
        """
        Create a new SnippetSet with given parameters.
        Does not check for duplicates — callers should use get_or_create().
        """

        ss = SnippetSet(
            dataset_id=dataset.id,
            embedding_model_id=model.id,
            window_size=window_size,
            step_size=step_size,
            overlap=overlap,
            status=SnippetSetStatus.PENDING,
        )

        self.db.add(ss)
        self.db.commit()
        self.db.refresh(ss)
        return ss

    # ---------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------

    def assert_belongs_to_dataset(self, snippet_set_id: int, dataset_id: int) -> SnippetSet:
        """
        Fetch snippet_set and ensure it belongs to the given dataset.
        """
        ss = self.get(snippet_set_id)
        if ss.dataset_id != dataset_id:
            raise ValueError(
                f"SnippetSet(id={snippet_set_id}) does not belong to Dataset(id={dataset_id})"
            )
        return ss

    def assert_for_model(self, snippet_set_id: int, model_id: int) -> SnippetSet:
        """
        Ensure snippet set is tied to the correct embedding model.
        """
        ss = self.get(snippet_set_id)
        if ss.embedding_model_id != model_id:
            raise ValueError(
                f"SnippetSet(id={snippet_set_id}) does not belong to EmbeddingModel(id={model_id})"
            )
        return ss

    # ---------------------------------------------------------
    # Annotation protection
    # ---------------------------------------------------------

    def count_annotations(self, snippet_set_id: int) -> int:
        """
        Count total annotations across all snippets in this snippet set.
        Used for data loss prevention.
        """
        count = (
            self.db.query(func.count(Annotation.id))
            .join(Snippet, Annotation.snippet_id == Snippet.id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .scalar()
        )
        return count or 0

    def get_annotation_stats(self, snippet_set_id: int) -> Dict[str, Any]:
        """
        Get detailed annotation statistics for a snippet set.
        Returns dict with annotation_count, annotated_snippet_count, etc.
        """
        # Total annotations
        total_annotations = self.count_annotations(snippet_set_id)
        
        # Number of snippets with at least one annotation
        annotated_snippets = (
            self.db.query(func.count(func.distinct(Snippet.id)))
            .join(Annotation, Snippet.id == Annotation.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .scalar()
        ) or 0
        
        # Total snippets in set
        total_snippets = (
            self.db.query(func.count(Snippet.id))
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .scalar()
        ) or 0
        
        return {
            "annotation_count": total_annotations,
            "annotated_snippet_count": annotated_snippets,
            "total_snippet_count": total_snippets,
            "has_annotations": total_annotations > 0,
        }

    # ---------------------------------------------------------
    # Delete with protection
    # ---------------------------------------------------------

    def safe_delete(
        self, 
        snippet_set_id: int, 
        *, 
        force: bool = False,
        allow_with_annotations: bool = False
    ) -> Dict[str, Any]:
        """
        Safely delete a snippet set with annotation loss prevention.
        
        Args:
            snippet_set_id: ID of the snippet set to delete
            force: If True, bypass annotation checks (dangerous!)
            allow_with_annotations: If True, allows deletion even with annotations
                                   (requires explicit acknowledgment)
        
        Returns:
            Dict with deletion info including stats on what was deleted
            
        Raises:
            ValueError: If snippet set not found
            RuntimeError: If annotations exist and protection is active
        """
        ss = self.get(snippet_set_id)
        
        # Get stats before deletion
        stats = self.get_annotation_stats(snippet_set_id)
        
        # Protection: prevent deletion if annotations exist
        if stats["annotation_count"] > 0:
            if not (force or allow_with_annotations):
                raise RuntimeError(
                    f"Cannot delete SnippetSet(id={snippet_set_id}): "
                    f"It contains {stats['annotation_count']} annotation(s) "
                    f"across {stats['annotated_snippet_count']} snippet(s). "
                    f"Deleting this would result in permanent data loss. "
                    f"To proceed, you must explicitly acknowledge this by setting "
                    f"allow_with_annotations=True."
                )
        
        # Perform deletion
        self.db.delete(ss)
        self.db.commit()
        
        return {
            "deleted_snippet_set_id": snippet_set_id,
            "deleted_annotation_count": stats["annotation_count"],
            "deleted_snippet_count": stats["total_snippet_count"],
            "message": "SnippetSet deleted successfully" + 
                      (f" (with {stats['annotation_count']} annotations)" 
                       if stats["annotation_count"] > 0 else ""),
        }
