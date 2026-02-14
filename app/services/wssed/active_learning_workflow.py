"""
Active Learning Workflow

Main orchestration service for active learning with species-specific models.
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from sqlalchemy.orm import Session
import torch
import numpy as np

from app.models.wssed import WSSEDSpeciesModel, WSSEDSnippetLabel, FeedbackType
from app.models.snippet import Snippet
from app.services.species_model_store import get_species_model_store, SpeciesModelStore
from app.services.wssed.species_model_manager import SpeciesModelManager
from app.services.wssed.data_loader import DataLoader
from app.services.wssed.prediction_handler import PredictionHandler
from active_learning.active_learning import ActiveLearning

logger = logging.getLogger(__name__)


class ActiveLearningService:
    """
    Service for managing active learning workflow with species-specific models.
    
    This service orchestrates the active learning process by coordinating
    model management, data loading, predictions, and training.
    """

    def __init__(self, db: Session, model_store: Optional[SpeciesModelStore] = None):
        self.db = db
        self.model_store = model_store or get_species_model_store()
        
        # Initialize component services
        self.model_manager = SpeciesModelManager(db)
        self.data_loader = DataLoader(db)
        self.prediction_handler = PredictionHandler(db)

    # ========== Species Model Management (Delegated) ==========

    def register_species_model(
        self,
        species_name: str,
        dataset_id: int,
        base_model_directory: str,
        metric_type: str = "macro",
        prediction_level: str = "segment",
        model_version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> WSSEDSpeciesModel:
        """
        Register or update a species-specific model.
        
        See SpeciesModelManager.register_model for details.
        """
        return self.model_manager.register_model(
            species_name=species_name,
            dataset_id=dataset_id,
            base_model_directory=base_model_directory,
            metric_type=metric_type,
            prediction_level=prediction_level,
            model_version=model_version,
            hyperparameters=hyperparameters
        )

    def get_species_model(self, species_model_id: int) -> Optional[WSSEDSpeciesModel]:
        """Get a species model by ID."""
        return self.model_manager.get_by_id(species_model_id)

    def get_species_model_by_name(
        self, species_name: str, dataset_id: int
    ) -> Optional[WSSEDSpeciesModel]:
        """Get a species model by name and dataset."""
        return self.model_manager.get_by_name(species_name, dataset_id)

    def list_species_models(self, dataset_id: Optional[int] = None) -> List[WSSEDSpeciesModel]:
        """List all species models, optionally filtered by dataset."""
        return self.model_manager.list_models(dataset_id)

    # ========== Active Learning API ==========

    def get_suggestions(
        self,
        snippet_set_id: int,
        species_name: str,
        dataset_id: int,
        strategy: str = "uncertainty",
        k: int = 20,
        device: str = "cpu",
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Get active learning suggestions for labeling.

        Args:
            snippet_set_id: Snippet set ID
            species_name: Species name
            dataset_id: Dataset ID
            strategy: Query strategy ("uncertainty", "margin", "diversity", "hybrid")
            k: Number of suggestions to return
            device: Device for computation ("cpu" or "cuda")
            seed: Random seed

        Returns:
            Dictionary with:
                - snippet_ids: List of suggested snippet IDs
                - probs: List of predicted probabilities
                - confidences: List of confidence scores
                - n_labeled: Number of already labeled snippets
                - model_info: Information about the model used
        """
        # Get species model
        species_model = self.get_species_model_by_name(species_name, dataset_id)
        if not species_model:
            raise ValueError(
                f"No species model found for species '{species_name}' in dataset {dataset_id}"
            )

        # Load embeddings
        X_pool, Z_pool, snippet_ids = self.data_loader.load_embedding_pool(snippet_set_id)

        # Load model checkpoint
        model = self.model_store.load_model(
            model_directory=species_model.model_directory,
            metric_type=species_model.metric_type,
            prediction_level=species_model.prediction_level
        )

        # Create ActiveLearning object
        al = ActiveLearning(X_pool=X_pool, Z_pool=Z_pool)

        # Load existing labels
        labels = self.data_loader.load_labels(snippet_set_id, species_model.id)
        sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids)}
        idx_to_label = {
            sid_to_idx[sid]: lab
            for sid, lab in labels.items()
            if sid in sid_to_idx
        }
        al.apply_new_annotations(idx_to_label)

        # Get predictions for this specific species (handles multi-class models)
        if strategy == "uncertainty":
            p_np = self.prediction_handler.predict_probs_for_species(
                model, X_pool, species_name, device=device
            )
        else:
            p_np = None
        
        # Use ActiveLearning's select_topk for sample selection
        is_labeled_np = al.is_labeled_mask()
        chosen_idx = ActiveLearning.select_topk(
            strategy=strategy,
            k=k,
            is_labeled_np=is_labeled_np,
            p_np=p_np,
            Z_np=Z_pool,
            seed=seed
        )
        
        chosen_snippet_ids = [snippet_ids[i] for i in chosen_idx]
        
        # Get probabilities for chosen samples
        probs = []
        confidences = []
        if p_np is not None:
            for idx in chosen_idx:
                prob = float(p_np[idx])
                probs.append(prob)
                # Confidence is inverse of uncertainty (distance from 0.5)
                uncertainty = abs(prob - 0.5) * 2  # Scale to [0, 1]
                confidence = 1.0 - uncertainty  # Invert: high uncertainty = low confidence
                confidences.append(confidence)
        else:
            probs = [None] * len(chosen_snippet_ids)
            confidences = [None] * len(chosen_snippet_ids)

        # Store predictions in the database
        self.prediction_handler.store_predictions(
            species_model_id=species_model.id,
            snippet_ids=chosen_snippet_ids,
            probs=probs,
            confidences=confidences
        )

        return {
            "snippet_ids": chosen_snippet_ids,
            "probs": probs,
            "confidences": confidences,
            "n_labeled": int(is_labeled_np.sum()),
            "model_info": {
                "species_model_id": species_model.id,
                "species_name": species_model.species_name,
                "metric_type": species_model.metric_type,
                "prediction_level": species_model.prediction_level,
                "model_version": species_model.model_version,
            }
        }

    def submit_labels_and_maybe_retrain(
        self,
        snippet_set_id: int,
        species_name: str,
        dataset_id: int,
        snippet_id_to_label: Dict[int, int],  # {snippet_id: 0/1}
        retrain_every: int = 5,
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Submit user labels and optionally trigger retraining.

        Args:
            snippet_set_id: Snippet set ID
            species_name: Species name
            dataset_id: Dataset ID
            snippet_id_to_label: Dictionary mapping snippet_id to label (0 or 1)
            retrain_every: Retrain after every N labels (default 5)
            device: Device for training ("cpu" or "cuda")
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Dictionary with:
                - added: Number of new labels added
                - labeled_count: Total number of labeled snippets
                - retrained: Whether retraining was triggered
                - train_stats: Training statistics (if retrained)
                - checkpoint: Path to saved checkpoint (if retrained)
        """
        # Get species model
        species_model = self.get_species_model_by_name(species_name, dataset_id)
        if not species_model:
            raise ValueError(
                f"No species model found for species '{species_name}' in dataset {dataset_id}"
            )

        # Load embeddings
        X_pool, Z_pool, snippet_ids = self.data_loader.load_embedding_pool(snippet_set_id)

        # Load model
        model = self.model_store.load_model(
            model_directory=species_model.model_directory,
            metric_type=species_model.metric_type,
            prediction_level=species_model.prediction_level
        )

        # Create ActiveLearning object
        al = ActiveLearning(X_pool=X_pool, Z_pool=Z_pool)

        # Load existing labels
        existing = self.data_loader.load_labels(snippet_set_id, species_model.id)
        sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids)}
        al.apply_new_annotations({
            sid_to_idx[sid]: lab
            for sid, lab in existing.items()
            if sid in sid_to_idx
        })

        # Apply new labels
        idx_to_label = {
            sid_to_idx[sid]: int(lab)
            for sid, lab in snippet_id_to_label.items()
            if sid in sid_to_idx
        }
        added = al.apply_new_annotations(idx_to_label)

        # Save labels to database
        self.data_loader.save_labels(species_model.id, snippet_id_to_label)

        # Check if retraining is needed
        labeled_count = int(al.is_labeled_mask().sum())
        do_retrain = (added > 0) and (labeled_count % retrain_every == 0)

        result = {
            "added": added,
            "labeled_count": labeled_count,
            "retrained": False,
            "species_model_id": species_model.id
        }

        if do_retrain:
            logger.info(
                f"Triggering retraining for species '{species_name}' "
                f"after {labeled_count} labels"
            )
            result.update(self._retrain_model(
                species_model=species_model,
                al=al,
                model=model,
                labeled_count=labeled_count,
                device=device,
                epochs=epochs,
                lr=lr,
                version_suffix=""
            ))

        return result

    def manual_retrain(
        self,
        snippet_set_id: int,
        species_name: str,
        dataset_id: int,
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Manually trigger retraining for a species model.

        Args:
            snippet_set_id: Snippet set ID
            species_name: Species name
            dataset_id: Dataset ID
            device: Device for training ("cpu" or "cuda")
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Dictionary with:
                - labeled_count: Total number of labeled snippets
                - retrained: Whether retraining was successful
                - train_stats: Training statistics
                - checkpoint: Path to saved checkpoint
        """
        # Get species model
        species_model = self.get_species_model_by_name(species_name, dataset_id)
        if not species_model:
            raise ValueError(
                f"No species model found for species '{species_name}' in dataset {dataset_id}"
            )

        # Load embeddings
        X_pool, Z_pool, snippet_ids = self.data_loader.load_embedding_pool(snippet_set_id)

        # Load model
        model = self.model_store.load_model(
            model_directory=species_model.model_directory,
            metric_type=species_model.metric_type,
            prediction_level=species_model.prediction_level
        )

        # Create ActiveLearning object
        al = ActiveLearning(X_pool=X_pool, Z_pool=Z_pool)

        # Load existing labels
        existing = self.data_loader.load_labels(snippet_set_id, species_model.id)
        sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids)}
        al.apply_new_annotations({
            sid_to_idx[sid]: lab
            for sid, lab in existing.items()
            if sid in sid_to_idx
        })

        # Check if we have labels
        labeled_count = int(al.is_labeled_mask().sum())
        if labeled_count == 0:
            raise ValueError("No labels found. Cannot retrain without labels.")

        logger.info(
            f"Manual retraining triggered for species '{species_name}' "
            f"with {labeled_count} labels"
        )
        
        result = self._retrain_model(
            species_model=species_model,
            al=al,
            model=model,
            labeled_count=labeled_count,
            device=device,
            epochs=epochs,
            lr=lr,
            version_suffix="_manual"
        )
        result["added"] = 0  # Manual retraining doesn't add new labels
        
        return result

    def _retrain_model(
        self,
        species_model: WSSEDSpeciesModel,
        al: ActiveLearning,
        model,
        labeled_count: int,
        device: str,
        epochs: int,
        lr: float,
        version_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Internal method to perform model retraining and save checkpoint.
        
        Args:
            species_model: Species model instance
            al: ActiveLearning object with labels
            model: PyTorch model to retrain
            labeled_count: Number of labeled samples
            device: Device for training
            epochs: Number of training epochs
            lr: Learning rate
            version_suffix: Suffix for version string (e.g., "_manual")
            
        Returns:
            Dictionary with retraining results
        """
        # Perform retraining
        stats = al.retrain(model, device=device, epochs=epochs, lr=lr)
        
        # Save the updated model
        model_path = self.model_store.get_model_path(
            model_directory=species_model.model_directory,
            metric_type=species_model.metric_type,
            prediction_level=species_model.prediction_level
        )
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_stats': stats,
            'labeled_count': labeled_count,
        }, str(model_path))
        logger.info(f"Checkpoint saved to {model_path.resolve()}")
        
        # Clear cache to force reload next time
        self.model_store.clear_cache(species_model.model_directory)
        
        # Update model version
        version = f"v{labeled_count}{version_suffix}"
        self.model_manager.update_version(species_model, version)
        
        return {
            "retrained": True,
            "species_model_id": species_model.id,
            "train_stats": stats,
            "checkpoint": str(model_path.resolve()),
            "labeled_count": labeled_count
        }

    def get_statistics(
        self, species_model_id: int, snippet_set_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a species model.

        Args:
            species_model_id: Species model ID
            snippet_set_id: Optional snippet set to filter by

        Returns:
            Dictionary with statistics
        """
        query = self.db.query(WSSEDSnippetLabel).filter(
            WSSEDSnippetLabel.species_model_id == species_model_id
        )

        if snippet_set_id:
            query = query.join(Snippet).filter(Snippet.snippet_set_id == snippet_set_id)

        all_labels = query.all()

        total = len(all_labels)
        labeled = sum(1 for label in all_labels if label.user_label is not None)
        accepted = sum(
            1 for label in all_labels
            if label.user_label == FeedbackType.ACCEPTED
        )
        rejected = sum(
            1 for label in all_labels
            if label.user_label == FeedbackType.REJECTED
        )

        return {
            "species_model_id": species_model_id,
            "snippet_set_id": snippet_set_id,
            "total_predictions": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "accepted": accepted,
            "rejected": rejected,
        }
