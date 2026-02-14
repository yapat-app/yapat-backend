"""
Prediction Handler

Handles model predictions and prediction storage for active learning.
"""

from typing import List, Optional
import numpy as np
import torch
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.wssed import WSSEDSnippetLabel

logger = logging.getLogger(__name__)

# Species to class index mapping for multi-class models
# Based on WSSED training with 4 species: DENMIN, BOARAN, DENNAN, LEPFUS
SPECIES_TO_CLASS_IDX = {
    "Dendropsophus_minutus": 0,
    "Boana_raniceps": 1,
    "Dendropsophus_nanus": 2,
    "Leptodactylus_fuscus": 3,
}


class PredictionHandler:
    """Handles predictions and prediction storage."""

    def __init__(self, db: Session):
        self.db = db

    def predict_probs_for_species(
        self, 
        model, 
        X_np: np.ndarray, 
        species_name: str, 
        device: str = "cpu", 
        batch_size: int = 2048
    ) -> np.ndarray:
        """
        Predict probabilities for a specific species from a multi-class model.
        
        Args:
            model: PyTorch model
            X_np: Input embeddings [N, D]
            species_name: Target species name
            device: Device for computation
            batch_size: Batch size for inference
            
        Returns:
            probs: Probabilities for the target species [N]
        """
        model.eval()
        model.to(device)
        
        # Get class index for this species
        class_idx = SPECIES_TO_CLASS_IDX.get(species_name)
        
        out = []
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                x = torch.from_numpy(X_np[i:i+batch_size]).to(device)
                logits = model(x)
                
                # Check if model is multi-class or binary
                if logits.dim() == 1:
                    # Binary classification (single output per sample)
                    p = torch.sigmoid(logits).cpu().numpy()
                elif logits.dim() == 2:
                    # Multi-class classification (multiple outputs per sample)
                    if class_idx is not None:
                        # Extract probability for the target species class
                        probs_all = torch.softmax(logits, dim=1)
                        p = probs_all[:, class_idx].cpu().numpy()
                    else:
                        # Species not in mapping, use first class or error
                        logger.warning(
                            f"Species '{species_name}' not in class mapping. "
                            f"Available: {list(SPECIES_TO_CLASS_IDX.keys())}"
                        )
                        # Use average probability across all classes as fallback
                        probs_all = torch.softmax(logits, dim=1)
                        p = probs_all.mean(dim=1).cpu().numpy()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                out.append(p)
        
        return np.concatenate(out, axis=0)

    def store_predictions(
        self,
        species_model_id: int,
        snippet_ids: List[int],
        probs: List[float],
        confidences: List[Optional[float]]
    ):
        """
        Store predictions in the database.

        Args:
            species_model_id: Species model ID
            snippet_ids: List of snippet IDs
            probs: List of predicted probabilities
            confidences: List of confidence scores
        """
        for snippet_id, prob, confidence in zip(snippet_ids, probs, confidences):
            # Check if prediction already exists
            existing = self.db.query(WSSEDSnippetLabel).filter(
                and_(
                    WSSEDSnippetLabel.species_model_id == species_model_id,
                    WSSEDSnippetLabel.snippet_id == snippet_id
                )
            ).first()

            if existing:
                # Update existing prediction
                existing.predicted_label = float(prob)
                existing.confidence_score = float(confidence) if confidence is not None else None
            else:
                # Create new prediction
                new_label = WSSEDSnippetLabel(
                    species_model_id=species_model_id,
                    snippet_id=snippet_id,
                    predicted_label=float(prob),
                    confidence_score=float(confidence) if confidence is not None else None
                )
                self.db.add(new_label)

        self.db.commit()
        logger.info(f"Stored {len(snippet_ids)} predictions for species_model {species_model_id}")
