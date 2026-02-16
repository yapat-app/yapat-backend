"""
Active Learning Workflow

Main orchestration service for active learning with species-specific models.
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
import torch
import numpy as np

import os

from app.config import settings
from app.models.wssed import WSSEDSpeciesModel, WSSEDSnippetLabel, FeedbackType
from app.models.snippet import Snippet
from app.models.embedding import SnippetSet
from app.services.species_model_store import get_species_model_store, SpeciesModelStore
from app.services.wssed.species_model_manager import SpeciesModelManager
from app.services.wssed.data_loader import DataLoader
from app.services.wssed.prediction_handler import PredictionHandler
from active_learning.active_learning import ActiveLearning

logger = logging.getLogger(__name__)


def _resolve_model_directory(species_model: WSSEDSpeciesModel) -> str:
    """
    Resolve the filesystem path for a species model so it works across deployments.

    The database stores an absolute path (e.g. /model_AL/boaran) from when the model
    was registered. On another server, that path may not exist. If ACTIVE_LEARNING_MODELS_DIR
    is set, we resolve to <ACTIVE_LEARNING_MODELS_DIR>/<species_subdir> so the same
    DB can be used with a different models directory (e.g. /srv/DATA01/.../models_AL).
    """
    stored = species_model.model_directory
    if not settings.ACTIVE_LEARNING_MODELS_DIR:
        return stored
    # Use current base + last path component of stored path (the species subdir name)
    species_subdir = os.path.basename(stored.rstrip(os.sep))
    resolved = os.path.join(settings.ACTIVE_LEARNING_MODELS_DIR, species_subdir)
    return resolved


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

        # Load model checkpoint (resolve path so it works across deployments)
        model = self.model_store.load_model(
            model_directory=_resolve_model_directory(species_model),
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

        # Load model (resolve path so it works across environments)
        model = self.model_store.load_model(
            model_directory=_resolve_model_directory(species_model),
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

        # Load model (resolve path so it works across deployments)
        model = self.model_store.load_model(
            model_directory=_resolve_model_directory(species_model),
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

        Saves the main checkpoint (overwritten each time, used for loading) and
        a timestamped copy under checkpoints/ for history.
        
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
            Dictionary with retraining results (checkpoint, checkpoint_history, train_stats, ...)
        """
        # Perform retraining
        stats = al.retrain(model, device=device, epochs=epochs, lr=lr)

        resolved_dir = _resolve_model_directory(species_model)
        # Save the updated model (main file used for loading)
        model_path = self.model_store.get_model_path(
            model_directory=resolved_dir,
            metric_type=species_model.metric_type,
            prediction_level=species_model.prediction_level
        )
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_payload = {
            'model_state_dict': model.state_dict(),
            'train_stats': stats,
            'labeled_count': labeled_count,
        }

        # Save main checkpoint (overwrites; this is what we load)
        torch.save(checkpoint_payload, str(model_path))
        logger.info(f"Checkpoint saved to {model_path.resolve()}")

        # Save a copy at each retrain for history (checkpoints/checkpoint_n{labeled_count}_{timestamp}{suffix}.pt)
        checkpoint_dir = model_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_suffix = version_suffix.replace(" ", "_") if version_suffix else ""
        history_name = f"checkpoint_n{labeled_count}_{ts}{safe_suffix}.pt"
        history_path = checkpoint_dir / history_name
        checkpoint_payload['saved_at'] = datetime.utcnow().isoformat() + "Z"
        torch.save(checkpoint_payload, str(history_path))
        logger.info(f"Checkpoint copy saved to {history_path.resolve()}")

        # Clear cache to force reload next time
        self.model_store.clear_cache(resolved_dir)
        
        # Update model version
        version = f"v{labeled_count}{version_suffix}"
        self.model_manager.update_version(species_model, version)
        
        return {
            "retrained": True,
            "species_model_id": species_model.id,
            "train_stats": stats,
            "checkpoint": str(model_path.resolve()),
            "checkpoint_history": str(history_path.resolve()),
            "labeled_count": labeled_count
        }

    def _count_snippets_for_species(
        self,
        species_model_id: int,
        snippet_set_id: Optional[int] = None,
    ) -> int:
        """
        Count total snippets for this species (dataset), optionally restricted to a snippet set.
        This is the stable count that does not change on retraining.
        """
        species_model = self.get_species_model(species_model_id)
        if not species_model:
            return 0
        q = (
            self.db.query(func.count(Snippet.id))
            .join(SnippetSet, Snippet.snippet_set_id == SnippetSet.id)
            .filter(SnippetSet.dataset_id == species_model.dataset_id)
        )
        if snippet_set_id is not None:
            q = q.filter(Snippet.snippet_set_id == snippet_set_id)
        return (q.scalar()) or 0

    def get_statistics(
        self, species_model_id: int, snippet_set_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a species model.

        total_snippets: all snippets in the species' dataset (or snippet set); stable across retrains.
        total_predictions: snippets that have a prediction (WSSEDSnippetLabel row); can change on retrain.
        """
        total_snippets = self._count_snippets_for_species(
            species_model_id, snippet_set_id
        )

        query = self.db.query(WSSEDSnippetLabel).filter(
            WSSEDSnippetLabel.species_model_id == species_model_id
        )

        if snippet_set_id:
            query = query.join(Snippet).filter(Snippet.snippet_set_id == snippet_set_id)

        all_labels = query.all()

        total_predictions = len(all_labels)
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
            "total_snippets": total_snippets,
            "total_predictions": total_predictions,
            "labeled": labeled,
            "unlabeled": total_predictions - labeled,
            "accepted": accepted,
            "rejected": rejected,
        }

    def get_prediction_histogram(
        self,
        species_model_id: int,
        snippet_set_id: Optional[int] = None,
        num_bins: int = 10,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Build a histogram of model predictions (0-1) for the species (or snippet set).

        When snippet_set_id is provided: runs the model on the full embedding pool for that
        set so the histogram reflects all snippets in the set (not just previously stored
        predictions), and stores those predictions for consistency.
        When snippet_set_id is None: uses only already-stored predictions (may be partial).

        total_snippets is always the full count of snippets for this species/set (stable).

        Args:
            species_model_id: Species model ID.
            snippet_set_id: Optional snippet set. If set, we run model on full pool for that set.
            num_bins: Number of bins in [0, 1] (default 10).
            device: Device for inference ("cpu" or "cuda").

        Returns:
            Dict with bin_edges, counts, total_snippets, snippets_with_predictions.
        """
        species_model = self.get_species_model(species_model_id)
        if not species_model:
            raise ValueError(f"Species model not found: {species_model_id}")

        total_snippets = self._count_snippets_for_species(
            species_model_id, snippet_set_id
        )

        predictions = np.array([], dtype=np.float64)
        # When a snippet set is specified, run model on full pool so histogram is for entire set
        if snippet_set_id is not None:
            try:
                X_pool, _, snippet_ids = self.data_loader.load_embedding_pool(
                    snippet_set_id
                )
                model = self.model_store.load_model(
                    model_directory=_resolve_model_directory(species_model),
                    metric_type=species_model.metric_type,
                    prediction_level=species_model.prediction_level,
                )
                p_np = self.prediction_handler.predict_probs_for_species(
                    model, X_pool, species_model.species_name, device=device
                )
                predictions = p_np
                # Store all predictions so stats and future histogram use full set
                probs = [float(p) for p in predictions]
                confidences = [None] * len(probs)
                self.prediction_handler.store_predictions(
                    species_model_id=species_model_id,
                    snippet_ids=snippet_ids,
                    probs=probs,
                    confidences=confidences,
                )
            except ValueError as e:
                # No embeddings for this set (e.g. pool not ready): fall back to stored only
                logger.debug(
                    "Could not load embedding pool for histogram, using stored predictions: %s",
                    e,
                )
                pass

        if len(predictions) == 0:
            # Use stored predictions only (snippet_set_id was None or pool load failed)
            query = self.db.query(WSSEDSnippetLabel.predicted_label).filter(
                WSSEDSnippetLabel.species_model_id == species_model_id
            )
            if snippet_set_id is not None:
                query = query.join(Snippet).filter(
                    Snippet.snippet_set_id == snippet_set_id
                )
            rows = query.all()
            predictions = np.array([float(r[0]) for r in rows], dtype=np.float64)

        snippets_with_predictions = len(predictions)

        if num_bins < 1:
            num_bins = 10
        bins = np.linspace(0, 1, num_bins + 1)
        predictions = np.clip(predictions, 0.0, 1.0)
        counts, bin_edges = np.histogram(predictions, bins=bins)
        bin_edges = bin_edges.tolist()
        counts = counts.tolist()

        return {
            "species_model_id": species_model_id,
            "species_name": species_model.species_name,
            "snippet_set_id": snippet_set_id,
            "bin_edges": bin_edges,
            "counts": counts,
            "total_snippets": total_snippets,
            "snippets_with_predictions": snippets_with_predictions,
        }
