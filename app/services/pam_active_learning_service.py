"""
PAM Active Learning Service

Orchestrates the full PAM active-learning flow:
  dataset selection → model checkout → inference → scoring/ranking
  → user feedback → auto/manual retrain

All DB access goes through this service; the ``active_learning.*``
modules are pure-compute (no SQLAlchemy dependency).

Model file resolution:
  • **First-time inference** — the checkpoint has no ``checkpoint_path``;
    the checkout layer falls back to the physical base model file
    (``Settings.PAM_BASE_MODEL_PATH``).
  • **After retrain** — a new versioned checkpoint file is written to
    ``Settings.PAM_CHECKPOINTS_DIR`` and a *new* ``PAMModelCheckpoint``
    DB row is created with the path, linked to its parent via
    ``parent_checkpoint_id``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.snippet import Snippet
from app.models.embedding import EmbeddingVector, SnippetSet
from app.models.pam_active_learning import (
    PAMModelCheckpoint,
    PAMModelStatus,
    PAMPrediction,
    PAMFeedbackEvent,
    PAMFeedbackAction,
    PAMRetrainJob,
    PAMRetrainStatus,
)
from active_learning.pam_model_checkout import checkout_model, PAMModelHandle
from active_learning.pam_classifier import load_pam_classifier
from active_learning.pam_scoring import combined_score, select_top_k
from active_learning.pam_retrain import (
    get_interaction_counter,
    run_retrain,
    AUTO_RETRAIN_THRESHOLD,
)

import numpy as np

logger = logging.getLogger(__name__)


class PAMActiveLearningService:
    """
    High-level service wiring together dataset access, model checkout,
    classifier inference, combined scoring, feedback persistence, and
    retrain orchestration for the PAM active-learning flow.
    """

    def __init__(self, db: Session):
        self.db = db

    # ================================================================
    # 1. Dataset selection
    # ================================================================

    def get_pam_dataset(self, dataset_id: int) -> Dataset:
        """
        Fetch a PAM dataset by ID.

        Raises ValueError if the dataset does not exist or is not of type PAM.
        """
        ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        if ds.dataset_type != DatasetType.PAM:
            raise ValueError(
                f"Dataset {dataset_id} is of type '{ds.dataset_type.value}', "
                f"expected 'PAM'"
            )
        return ds

    # ================================================================
    # 2. Model checkout / selection
    # ================================================================

    def register_checkpoint(
        self,
        dataset_id: int,
        name: str,
        version: str = "v0",
        checkpoint_path: Optional[str] = None,
        model_type: str = "pam_classifier",
        hyperparameters: Optional[Dict[str, Any]] = None,
        is_base: bool = False,
        parent_checkpoint_id: Optional[int] = None,
    ) -> PAMModelCheckpoint:
        """
        Create or update a model checkpoint record for a PAM dataset.

        When *is_base* is ``True`` and no *checkpoint_path* is supplied the
        checkpoint is treated as a "base model entry" — the checkout layer
        will automatically resolve the physical base model file at
        ``Settings.PAM_BASE_MODEL_PATH``.
        """
        self.get_pam_dataset(dataset_id)  # validate

        existing = (
            self.db.query(PAMModelCheckpoint)
            .filter(
                and_(
                    PAMModelCheckpoint.dataset_id == dataset_id,
                    PAMModelCheckpoint.name == name,
                    PAMModelCheckpoint.version == version,
                )
            )
            .first()
        )

        if existing:
            existing.checkpoint_path = checkpoint_path
            existing.model_type = model_type
            existing.hyperparameters = hyperparameters
            existing.is_base = int(is_base)
            existing.parent_checkpoint_id = parent_checkpoint_id
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            logger.info("Updated PAM checkpoint id=%d", existing.id)
            return existing

        ckpt = PAMModelCheckpoint(
            dataset_id=dataset_id,
            name=name,
            version=version,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            hyperparameters=hyperparameters,
            is_base=int(is_base),
            parent_checkpoint_id=parent_checkpoint_id,
            status=PAMModelStatus.AVAILABLE,
        )
        self.db.add(ckpt)
        self.db.commit()
        self.db.refresh(ckpt)
        logger.info("Registered PAM checkpoint id=%d name=%s is_base=%s", ckpt.id, name, is_base)
        return ckpt

    def get_checkpoint(self, checkpoint_id: int) -> PAMModelCheckpoint:
        ckpt = (
            self.db.query(PAMModelCheckpoint)
            .filter(PAMModelCheckpoint.id == checkpoint_id)
            .first()
        )
        if ckpt is None:
            raise ValueError(f"PAMModelCheckpoint {checkpoint_id} not found")
        return ckpt

    def list_checkpoints(
        self, dataset_id: Optional[int] = None
    ) -> List[PAMModelCheckpoint]:
        q = self.db.query(PAMModelCheckpoint)
        if dataset_id is not None:
            q = q.filter(PAMModelCheckpoint.dataset_id == dataset_id)
        return q.order_by(PAMModelCheckpoint.created_at.desc()).all()

    def _checkout(self, ckpt: PAMModelCheckpoint) -> PAMModelHandle:
        """
        Internal: wrap a DB row into a PAMModelHandle.

        Passes ``Settings.PAM_BASE_MODEL_PATH`` so the checkout layer
        can resolve the base model file when no retrained checkpoint
        exists yet.
        """
        return checkout_model(
            checkpoint_id=ckpt.id,
            dataset_id=ckpt.dataset_id,
            name=ckpt.name,
            version=ckpt.version,
            checkpoint_path=ckpt.checkpoint_path,
            model_type=ckpt.model_type,
            hyperparameters=ckpt.hyperparameters or {},
            is_base=bool(ckpt.is_base),
            parent_checkpoint_id=ckpt.parent_checkpoint_id,
            base_model_path_setting=settings.PAM_BASE_MODEL_PATH,
        )

    # ================================================================
    # 3. Inference + scoring
    # ================================================================

    def run_inference(
        self,
        model_checkpoint_id: int,
        snippet_set_id: int,
        k: int = 20,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Run the full inference → scoring → ranking pipeline.

        Steps:
          1. Load embeddings for the snippet set.
          2. Check out the model and load the classifier.
          3. Run classifier inference (labels + confidences).
          4. Compute combined ranking scores.
          5. Select top-k and persist predictions.

        Returns:
            dict with ``predictions``, ``total_scored``, ``model_info``.
        """
        ckpt = self.get_checkpoint(model_checkpoint_id)
        handle = self._checkout(ckpt)

        # Load embeddings
        X_pool, snippet_ids = self._load_embeddings(snippet_set_id)

        # Load classifier
        classifier = load_pam_classifier(
            checkpoint_path=handle.effective_path,
            model_type=handle.model_type,
            device=device,
        )

        # Inference
        labels, confidences = classifier.predict(X_pool)

        # Combined scoring
        scores = combined_score(confidences)

        # Already-labeled mask (predictions with feedback)
        labeled_snippet_ids = set(
            r[0]
            for r in self.db.query(PAMFeedbackEvent.prediction_id)
            .join(PAMPrediction)
            .filter(PAMPrediction.model_checkpoint_id == model_checkpoint_id)
            .all()
        )
        labeled_mask = np.array(
            [sid in labeled_snippet_ids for sid in snippet_ids], dtype=bool
        )

        # Select top-k
        top_indices = select_top_k(scores, k=k, exclude_mask=labeled_mask)

        # Persist predictions (upsert)
        predictions_out = []
        for rank, idx in enumerate(top_indices):
            pred = self._upsert_prediction(
                model_checkpoint_id=model_checkpoint_id,
                snippet_id=snippet_ids[idx],
                predicted_label=labels[idx],
                confidence=float(confidences[idx]),
                ranking_score=float(scores[idx]),
            )
            predictions_out.append(pred)

        self.db.commit()

        return {
            "predictions": predictions_out,
            "total_scored": len(X_pool),
            "model_info": {
                "checkpoint_id": handle.checkpoint_id,
                "name": handle.name,
                "version": handle.version,
                "model_type": handle.model_type,
            },
        }

    def _load_embeddings(self, snippet_set_id: int):
        """Load embeddings for a snippet set. Returns (X_pool, snippet_ids)."""
        rows = (
            self.db.query(Snippet.id, EmbeddingVector.vector)
            .join(EmbeddingVector, Snippet.id == EmbeddingVector.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .order_by(Snippet.id)
            .all()
        )
        if not rows:
            raise ValueError(f"No embeddings found for snippet_set_id={snippet_set_id}")

        snippet_ids = [r[0] for r in rows]
        X_pool = np.array([r[1] for r in rows], dtype=np.float32)
        return X_pool, snippet_ids

    def _upsert_prediction(
        self,
        model_checkpoint_id: int,
        snippet_id: int,
        predicted_label: str,
        confidence: float,
        ranking_score: float,
    ) -> PAMPrediction:
        existing = (
            self.db.query(PAMPrediction)
            .filter(
                and_(
                    PAMPrediction.model_checkpoint_id == model_checkpoint_id,
                    PAMPrediction.snippet_id == snippet_id,
                )
            )
            .first()
        )
        if existing:
            existing.predicted_label = predicted_label
            existing.confidence = confidence
            existing.ranking_score = ranking_score
            return existing

        pred = PAMPrediction(
            model_checkpoint_id=model_checkpoint_id,
            snippet_id=snippet_id,
            predicted_label=predicted_label,
            confidence=confidence,
            ranking_score=ranking_score,
        )
        self.db.add(pred)
        return pred

    # ================================================================
    # 4. Feedback (accept / reject / modify)
    # ================================================================

    def submit_feedback(
        self,
        prediction_id: int,
        action: str,
        user_id: Optional[int] = None,
        modified_label: Optional[str] = None,
        notes: Optional[str] = None,
        retrain_threshold: int = AUTO_RETRAIN_THRESHOLD,
        retrain_epochs: int = 5,
        retrain_lr: float = 1e-3,
        retrain_device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Record a feedback event and, if the auto-retrain threshold is
        reached, trigger retraining.

        Returns dict with the feedback record + retrain status.
        """
        # Validate action
        try:
            action_enum = PAMFeedbackAction(action)
        except ValueError:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of: "
                f"{[a.value for a in PAMFeedbackAction]}"
            )

        if action_enum == PAMFeedbackAction.MODIFY and not modified_label:
            raise ValueError("modified_label is required when action=MODIFY")

        # Validate prediction
        pred = (
            self.db.query(PAMPrediction)
            .filter(PAMPrediction.id == prediction_id)
            .first()
        )
        if pred is None:
            raise ValueError(f"Prediction {prediction_id} not found")

        # Create feedback event
        event = PAMFeedbackEvent(
            prediction_id=prediction_id,
            user_id=user_id,
            action=action_enum,
            modified_label=modified_label,
            notes=notes,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)

        # Count feedback since last completed retrain for this checkpoint
        checkpoint_id = pred.model_checkpoint_id
        fb_count = self._feedback_count_since_retrain(checkpoint_id)

        # Interaction counter (in-process helper)
        counter = get_interaction_counter()
        counter.increment(checkpoint_id)

        # Check auto-retrain
        retrain_triggered = False
        if fb_count >= retrain_threshold:
            retrain_triggered = self._trigger_retrain(
                checkpoint_id=checkpoint_id,
                trigger="auto",
                feedback_count=fb_count,
                epochs=retrain_epochs,
                learning_rate=retrain_lr,
                device=retrain_device,
            )

        return {
            "feedback_id": event.id,
            "prediction_id": prediction_id,
            "action": action_enum.value,
            "modified_label": modified_label,
            "created_at": event.created_at,
            "feedback_count_since_retrain": fb_count,
            "retrain_triggered": retrain_triggered,
        }

    def _feedback_count_since_retrain(self, checkpoint_id: int) -> int:
        """Count feedback events after the most recent completed retrain."""
        last_retrain = (
            self.db.query(PAMRetrainJob.completed_at)
            .filter(
                PAMRetrainJob.model_checkpoint_id == checkpoint_id,
                PAMRetrainJob.status == PAMRetrainStatus.COMPLETED,
            )
            .order_by(PAMRetrainJob.completed_at.desc())
            .first()
        )
        cutoff = last_retrain[0] if last_retrain else datetime.min

        count = (
            self.db.query(func.count(PAMFeedbackEvent.id))
            .join(PAMPrediction)
            .filter(
                PAMPrediction.model_checkpoint_id == checkpoint_id,
                PAMFeedbackEvent.created_at > cutoff,
            )
            .scalar()
        )
        return count or 0

    # ================================================================
    # 5. Retrain (auto + manual)
    # ================================================================

    def manual_retrain(
        self,
        model_checkpoint_id: int,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> PAMRetrainJob:
        """
        Manually trigger a retrain regardless of interaction count.
        """
        fb_count = self._feedback_count_since_retrain(model_checkpoint_id)
        self._trigger_retrain(
            checkpoint_id=model_checkpoint_id,
            trigger="manual",
            feedback_count=fb_count,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
        # Return the latest job
        return (
            self.db.query(PAMRetrainJob)
            .filter(PAMRetrainJob.model_checkpoint_id == model_checkpoint_id)
            .order_by(PAMRetrainJob.created_at.desc())
            .first()
        )

    def _next_version(self, checkpoint_id: int) -> str:
        """
        Compute the next version tag for a checkpoint lineage.

        Inspects existing versions for the same (dataset, name) and
        returns "v{max+1}".
        """
        ckpt = self.get_checkpoint(checkpoint_id)
        siblings = (
            self.db.query(PAMModelCheckpoint.version)
            .filter(
                PAMModelCheckpoint.dataset_id == ckpt.dataset_id,
                PAMModelCheckpoint.name == ckpt.name,
            )
            .all()
        )
        max_num = 0
        for (v,) in siblings:
            # Parse "v0", "v1", … "vN"
            try:
                num = int(v.lstrip("v"))
                max_num = max(max_num, num)
            except (ValueError, AttributeError):
                pass
        return f"v{max_num + 1}"

    def _trigger_retrain(
        self,
        checkpoint_id: int,
        trigger: str,
        feedback_count: int,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> bool:
        """
        Create a PAMRetrainJob, invoke the training entrypoint, and on
        success persist a **new versioned checkpoint** to disk and DB.

        Returns True on success, False on error.
        """
        ckpt = self.get_checkpoint(checkpoint_id)
        handle = self._checkout(ckpt)
        new_version = self._next_version(checkpoint_id)

        job = PAMRetrainJob(
            model_checkpoint_id=checkpoint_id,
            trigger=trigger,
            feedback_count=feedback_count,
            status=PAMRetrainStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        try:
            metrics = run_retrain(
                checkpoint_id=checkpoint_id,
                trigger=trigger,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                feedback_count=feedback_count,
                model_name=ckpt.name,
                new_version=new_version,
                parent_checkpoint_path=handle.effective_path,
                checkpoints_dir=settings.PAM_CHECKPOINTS_DIR,
            )
            job.status = PAMRetrainStatus.COMPLETED
            job.result_metrics = metrics
            job.completed_at = datetime.utcnow()

            # ── Create a new checkpoint record for the retrained version ──
            new_checkpoint_path = metrics.get("new_checkpoint_path")
            new_ckpt = PAMModelCheckpoint(
                dataset_id=ckpt.dataset_id,
                name=ckpt.name,
                version=new_version,
                checkpoint_path=new_checkpoint_path,
                model_type=ckpt.model_type,
                hyperparameters=ckpt.hyperparameters,
                is_base=0,
                parent_checkpoint_id=checkpoint_id,
                status=PAMModelStatus.AVAILABLE,
            )
            self.db.add(new_ckpt)
            self.db.flush()  # get new_ckpt.id

            # Store references in the job metrics for the API layer
            metrics["new_checkpoint_id"] = new_ckpt.id
            job.result_metrics = metrics

            # Reset interaction counter
            counter = get_interaction_counter()
            counter.reset(checkpoint_id)

            logger.info(
                "PAM retrain job %d completed → new checkpoint id=%d version=%s path=%s",
                job.id, new_ckpt.id, new_version, new_checkpoint_path,
            )
        except Exception as exc:
            job.status = PAMRetrainStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            logger.error("PAM retrain job %d failed: %s", job.id, exc)

        self.db.commit()
        self.db.refresh(job)
        return job.status == PAMRetrainStatus.COMPLETED

    # ================================================================
    # 6. Statistics
    # ================================================================

    def get_stats(self, model_checkpoint_id: int) -> Dict[str, Any]:
        """Aggregate statistics for a checkpoint."""
        total_preds = (
            self.db.query(func.count(PAMPrediction.id))
            .filter(PAMPrediction.model_checkpoint_id == model_checkpoint_id)
            .scalar()
        ) or 0

        total_fb = (
            self.db.query(func.count(PAMFeedbackEvent.id))
            .join(PAMPrediction)
            .filter(PAMPrediction.model_checkpoint_id == model_checkpoint_id)
            .scalar()
        ) or 0

        accepted = (
            self.db.query(func.count(PAMFeedbackEvent.id))
            .join(PAMPrediction)
            .filter(
                PAMPrediction.model_checkpoint_id == model_checkpoint_id,
                PAMFeedbackEvent.action == PAMFeedbackAction.ACCEPT,
            )
            .scalar()
        ) or 0

        rejected = (
            self.db.query(func.count(PAMFeedbackEvent.id))
            .join(PAMPrediction)
            .filter(
                PAMPrediction.model_checkpoint_id == model_checkpoint_id,
                PAMFeedbackEvent.action == PAMFeedbackAction.REJECT,
            )
            .scalar()
        ) or 0

        modified = (
            self.db.query(func.count(PAMFeedbackEvent.id))
            .join(PAMPrediction)
            .filter(
                PAMPrediction.model_checkpoint_id == model_checkpoint_id,
                PAMFeedbackEvent.action == PAMFeedbackAction.MODIFY,
            )
            .scalar()
        ) or 0

        fb_since = self._feedback_count_since_retrain(model_checkpoint_id)

        retrain_jobs = (
            self.db.query(func.count(PAMRetrainJob.id))
            .filter(PAMRetrainJob.model_checkpoint_id == model_checkpoint_id)
            .scalar()
        ) or 0

        return {
            "model_checkpoint_id": model_checkpoint_id,
            "total_predictions": total_preds,
            "total_feedback": total_fb,
            "accepted": accepted,
            "rejected": rejected,
            "modified": modified,
            "feedback_since_last_retrain": fb_since,
            "retrain_jobs": retrain_jobs,
        }
