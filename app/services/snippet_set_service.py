"""
SnippetSet service — manages creation and retrieval of snippet sets.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.embedding import (
    SnippetSet,
    SnippetSetStatus,
    EmbeddingModel,
)
from app.models.dataset import Dataset


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
