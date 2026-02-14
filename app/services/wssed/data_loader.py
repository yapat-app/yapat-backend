"""
Data Loader

Handles loading and saving of embeddings and labels for active learning.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.wssed import WSSEDSnippetLabel, FeedbackType
from app.models.embedding import EmbeddingVector
from app.models.snippet import Snippet

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and persistence for active learning."""

    def __init__(self, db: Session):
        self.db = db

    def load_embedding_pool(
        self, snippet_set_id: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """
        Load embeddings for a snippet set.

        Args:
            snippet_set_id: Snippet set ID

        Returns:
            Tuple containing:
                - X_pool: np.ndarray [N, D] - embeddings
                - Z_pool: np.ndarray [N, d] or None - additional features (if any)
                - snippet_ids: List[int] - maps pool index to snippet_id
        """
        # Get all snippets in the snippet set with their embeddings
        results = (
            self.db.query(Snippet.id, EmbeddingVector.vector)
            .join(EmbeddingVector, Snippet.id == EmbeddingVector.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .order_by(Snippet.id)
            .all()
        )

        if not results:
            raise ValueError(f"No embeddings found for snippet_set_id={snippet_set_id}")

        snippet_ids = [r[0] for r in results]
        embeddings = [r[1] for r in results]

        # Convert to numpy array
        X_pool = np.array(embeddings, dtype=np.float32)
        Z_pool = None  # No additional features for now

        logger.info(
            f"Loaded {len(snippet_ids)} embeddings for snippet_set {snippet_set_id}, "
            f"shape={X_pool.shape}"
        )

        return X_pool, Z_pool, snippet_ids

    def load_labels(
        self, snippet_set_id: int, species_model_id: int
    ) -> Dict[int, int]:
        """
        Load existing labels for a species model and snippet set.

        Args:
            snippet_set_id: Snippet set ID
            species_model_id: Species model ID

        Returns:
            Dictionary mapping snippet_id to label (0 or 1)
        """
        # Get all labeled snippets for this species model in the snippet set
        results = (
            self.db.query(WSSEDSnippetLabel.snippet_id, WSSEDSnippetLabel.user_label)
            .join(Snippet, WSSEDSnippetLabel.snippet_id == Snippet.id)
            .filter(
                and_(
                    WSSEDSnippetLabel.species_model_id == species_model_id,
                    Snippet.snippet_set_id == snippet_set_id,
                    WSSEDSnippetLabel.user_label.isnot(None)
                )
            )
            .all()
        )

        # Convert feedback to binary labels: ACCEPTED=1, REJECTED=0
        labels = {}
        for snippet_id, user_label in results:
            if user_label == FeedbackType.ACCEPTED:
                labels[snippet_id] = 1
            elif user_label == FeedbackType.REJECTED:
                labels[snippet_id] = 0

        logger.info(
            f"Loaded {len(labels)} labels for species_model {species_model_id}, "
            f"snippet_set {snippet_set_id}"
        )

        return labels

    def save_labels(
        self,
        species_model_id: int,
        snippet_id_to_label: Dict[int, int]
    ):
        """
        Persist labels to the database.

        Args:
            species_model_id: Species model ID
            snippet_id_to_label: Dictionary mapping snippet_id to label (0 or 1)
        """
        for snippet_id, label in snippet_id_to_label.items():
            # Convert binary label to feedback type
            user_label = FeedbackType.ACCEPTED if label == 1 else FeedbackType.REJECTED

            # Check if label already exists
            existing_label = self.db.query(WSSEDSnippetLabel).filter(
                and_(
                    WSSEDSnippetLabel.species_model_id == species_model_id,
                    WSSEDSnippetLabel.snippet_id == snippet_id
                )
            ).first()

            if existing_label:
                # Update existing label
                existing_label.user_label = user_label
                existing_label.labeled_at = datetime.utcnow()
            else:
                # This shouldn't happen normally, but handle it
                logger.warning(
                    f"Creating new label entry for snippet {snippet_id} "
                    f"(should have been created during prediction)"
                )
                new_label = WSSEDSnippetLabel(
                    species_model_id=species_model_id,
                    snippet_id=snippet_id,
                    predicted_label=0.5,  # Unknown
                    user_label=user_label,
                    labeled_at=datetime.utcnow()
                )
                self.db.add(new_label)

        self.db.commit()
        logger.info(f"Saved {len(snippet_id_to_label)} labels for species_model {species_model_id}")
