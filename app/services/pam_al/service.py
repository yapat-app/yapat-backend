"""
PAMActiveLearningService — thin orchestrator.

Public methods (called directly by API endpoints) are defined here.
All heavy logic is delegated to the helper modules in this package.
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.core import taxonomy
from app.models.annotation import Annotation as AnnotationModel
from app.models.custom_taxonomy import CustomTaxonomy
from app.models.team import TeamMembership
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALPrediction,
    ALFeedbackEvent,
    ALRetrainJob,
    ALModelStatus,
    ALRetrainStatus,
    ALAnnotationSource,
)
from app.models.snippet import Snippet
from app.schemas.pam_active_learning import (
    ALTrainFromScratchRequest,
    ALFeedbackSubmit,
    ALPredictionResponse,
    ALModelType,
    SamplingMode,
)

from active_learning.model_zoo.mlp_multilabel_classifier import MultiLabelMLPClassifier
from active_learning.model_zoo.linear_multilabel_classifier import MultiLabelLinearClassifier
from active_learning.config import RETRAIN_AFTER
from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
    RETRAIN_AFTER,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DROPOUT,
)
from app.services.pam_al import _checkpoint_helpers as ckpt_h
from app.services.pam_al import _data_helpers as data_h
from app.services.pam_al import _annotation_helpers as ann_h
from app.services.pam_al import _inference_helpers as inf_h
from app.services.pam_al import _feedback_helpers as fb_h


logger = logging.getLogger(__name__)

DATA_ROOT = settings.DATA_ROOT or "/data"


def _resolve_device(*candidates: Any) -> str:
    """Resolve PAM AL device from request overrides or PAM_DEFAULT_DEVICE."""
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value:
            return value
    return str(settings.PAM_DEFAULT_DEVICE or "cpu")


class PAMActiveLearningService:
    """Thin orchestrator that wires helper functions together."""

    def __init__(self, db: Session):
        self.db = db

    def _cleanup_failed_training_checkpoint(
        self,
        checkpoint: ALModelCheckpoint,
        job: ALRetrainJob,
        error: Exception,
    ) -> None:
        """
        Mark a training checkpoint and its associated retrain job as failed.

        This keeps the job record available for polling/inspection while
        ensuring both entities are in terminal error states.
        """
        checkpoint_id = checkpoint.id
        job_id = job.id
        error_message = str(error)
        try:
            checkpoint.status = ALModelStatus.ERROR
            job.status = ALRetrainStatus.FAILED
            job.error_message = error_message
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            logger.info(
                "Marked checkpoint and job as failed checkpoint_id=%d job_id=%d error=%s",
                checkpoint_id,
                job_id,
                error_message,
            )
        except Exception:
            self.db.rollback()
            logger.exception(
                "Failed to mark checkpoint/job as failed checkpoint_id=%d job_id=%d",
                checkpoint_id,
                job_id,
            )
            raise

    # ==================================================================
    # Checkpoint management
    # ==================================================================

    def register_checkpoint(self, **kwargs) -> ALModelCheckpoint:
        return ckpt_h.register_checkpoint(self.db, **kwargs)

    def list_active_family_checkpoints(self, dataset_id: Optional[int] = None) -> List[ALModelCheckpoint]:
        return ckpt_h.list_active_family_checkpoints(self.db, dataset_id=dataset_id)

    def list_checkpoints(self, dataset_id: Optional[int] = None) -> List[ALModelCheckpoint]:
        return ckpt_h.list_checkpoints(self.db, dataset_id=dataset_id)

    def _get_checkpoint(self, checkpoint_id: int) -> Optional[ALModelCheckpoint]:
        return ckpt_h.get_checkpoint(self.db, checkpoint_id)

    # ==================================================================
    # Train from scratch  (sync — kept for backward compat)
    # ==================================================================

    def train_from_scratch(self, body: ALTrainFromScratchRequest) -> ALModelCheckpoint:
        ds = ckpt_h.get_pam_dataset(self.db, body.dataset_id)
        device = _resolve_device(body.device)

        snippet_set_id = body.snippet_set_id or ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError("No snippet_set_id provided and dataset has no default_snippet_set_id.")

        use_metadata_labels = bool(body.metadata_path and body.label_config_path)
        if use_metadata_labels:
            metadata_path = os.path.join(DATA_ROOT, body.metadata_path)
            label_config_path = os.path.join(DATA_ROOT, body.label_config_path)
            species_list = ckpt_h.load_species_from_label_config(label_config_path)
        else:
            metadata_path = None
            label_config_path = None

            annotations_by_snippet = ann_h.get_trusted_annotations(
                self.db,
                body.dataset_id,
            )

            if not annotations_by_snippet:
                raise ValueError("No user annotations available for bootstrap training.")

            species_list = sorted({
                label
                for labels in annotations_by_snippet.values()
                for label in labels
            })

            if not species_list:
                raise ValueError("No labels found in user annotations.")

        model = ckpt_h.make_model(body.model_type)

        # Hyperparameters not valid for linear classifiers
        is_mlp = body.model_type == ALModelType.PAM_MLP_MULTILABEL
        hidden_dim = body.hidden_dim if is_mlp else None
        dropout = body.dropout if is_mlp else None

        initial_label_config_path = label_config_path or ""
        model_ckpt = ALModelCheckpoint(
            dataset_id=body.dataset_id, model_family_name=body.model_family_name,
            version=body.version, checkpoint_path="", label_config_path=initial_label_config_path,
            model_type=body.model_type.value if hasattr(body.model_type, "value") else body.model_type,
            hyperparameters={
                "training_mode": "cold_start", "embedding_model_id": body.embedding_model_id,
                "metadata_path": metadata_path, "label_config_path": initial_label_config_path,
                "min_samples_per_class": body.min_samples_per_class,
                "max_samples_per_class": body.max_samples_per_class,
                "epochs": body.epochs, "learning_rate": body.learning_rate,
                "batch_size": body.batch_size, "hidden_dim": hidden_dim,
                "dropout": dropout, "device": device,
            },
            is_base=1, parent_checkpoint_id=None, status=ALModelStatus.LOADING,
        )
        self.db.add(model_ckpt)
        self.db.flush()

        job = ALRetrainJob(
            model_checkpoint_id=model_ckpt.id, dataset_id=body.dataset_id,
            trigger="cold_start", feedback_count=0, status=ALRetrainStatus.PENDING,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(model_ckpt)
        self.db.refresh(job)

        try:
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            X, snippet_rows = data_h.load_embeddings(self.db, snippet_set_id, body.embedding_model_id)
            if use_metadata_labels:
                gt_index = data_h.load_ground_truth_metadata(metadata_path, species_list, allowed_subsets=["train"])
                X_train, y_train, used_snippet_ids = data_h.align_embeddings_and_labels(X, snippet_rows, gt_index, species_list)
            else:
                snippet_ids = [row["snippet_id"] for row in snippet_rows]

                keep_indices = [
                    i for i, sid in enumerate(snippet_ids)
                    if sid in annotations_by_snippet
                ]

                if not keep_indices:
                    raise ValueError("No embeddings found for user-annotated snippets.")

                X_train = X[keep_indices]
                used_snippet_ids = [snippet_ids[i] for i in keep_indices]

                y_train = ann_h.build_multihot_from_annotations(
                    used_snippet_ids,
                    species_list,
                    annotations_by_snippet,
                )

            X_train, y_train, labeled_snippet_ids, used_species, excluded_species, class_counts = (
                model.filter_and_balance_classes(
                    X=X_train, y=y_train, snippet_ids=used_snippet_ids,
                    species_list=species_list,
                    min_samples_per_class=body.min_samples_per_class,
                    max_samples_per_class=body.max_samples_per_class,
                )
            )

            if y_train.shape[0] == 0:
                raise ValueError("No training samples remain after alignment.")
            if y_train.shape[1] == 0:
                raise ValueError("No species remain after min_samples_per_class filtering.")

            n_dim, num_classes = X_train.shape[1], y_train.shape[1]
            model.create_classifier(n_dim=n_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
            model.to(device)

            train_metrics = model.fit(X=X_train, y=y_train, epochs=body.epochs, learning_rate=body.learning_rate, batch_size=body.batch_size, device=device)

            checkpoint_path = ckpt_h.make_checkpoint_path(ds.id, body.model_family_name, body.version, model_ckpt.id)
            resolved_lcp = ckpt_h.make_label_config_path(ds.id, body.model_family_name, body.version, model_ckpt.id)

            ckpt_h.save_label_config(resolved_lcp, used_species)
            ckpt_h.save_classifier_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                hidden_dim=hidden_dim,
                dropout=dropout,
                label_order=used_species,
            )

            model_ckpt.checkpoint_path = checkpoint_path
            model_ckpt.label_config_path = resolved_lcp
            model_ckpt.status = ALModelStatus.AVAILABLE
            model_ckpt.hyperparameters = {
                **(model_ckpt.hyperparameters or {}),
                "resolved_snippet_set_id": snippet_set_id, "n_dim": n_dim, "num_classes": num_classes,
                "train_samples": int(X_train.shape[0]), "label_order": used_species,
                "used_species": used_species, "excluded_species": excluded_species, "class_counts": class_counts,
            }
            ann_h.store_snippet_annotations(self.db, body.dataset_id, labeled_snippet_ids, y_train, used_species, ALAnnotationSource.GROUND_TRUTH, model_ckpt.id)

            inference_metrics = None
            if body.run_inference:
                labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(self.db, body.dataset_id)
                inference_metrics = inf_h.run_and_store_inference(
                    self.db, body.dataset_id, model_ckpt, model, X, snippet_rows, used_species, labeled_ids,
                    body.threshold, body.density_k, body.composite_wu, body.composite_wd, body.composite_wr,
                )

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {
                "new_checkpoint_id": model_ckpt.id, "new_checkpoint_path": checkpoint_path,
                "label_config_path": body.label_config_path, "aligned_snippet_count": len(used_snippet_ids),
                "train_samples": int(X_train.shape[0]), "num_classes": int(num_classes),
                "used_species": used_species, "excluded_species": excluded_species,
                "class_counts": class_counts, **train_metrics,
            }
            ckpt_h.set_active_family_checkpoint(self.db, body.dataset_id, body.model_family_name, model_ckpt.id)
            self.db.commit()
            self.db.refresh(model_ckpt)
            return model_ckpt

        except Exception as e:
            logger.exception("Cold-start training failed.")
            self._cleanup_failed_training_checkpoint(model_ckpt, job, e)
            raise

    # ==================================================================
    # Feedback
    # ==================================================================

    def submit_feedback(self, body: ALFeedbackSubmit) -> dict:
        model_ckpt = ckpt_h.get_active_checkpoint_for_model_family(self.db, body.dataset_id, body.model_family_name)
        if model_ckpt is None:
            logger.info("No active checkpoint found. Submitting bootstrap feedback.")
            return self._submit_bootstrap_feedback(body)

        if model_ckpt.dataset_id != body.dataset_id:
            raise ValueError(f"Checkpoint {model_ckpt.id} does not belong to dataset {body.dataset_id}.")

        predictions = (
            self.db.query(ALPrediction)
            .filter(ALPrediction.model_checkpoint_id == model_ckpt.id, ALPrediction.snippet_id == body.snippet_id)
            .all()
        )
        if not predictions:
            raise ValueError(f"No prediction found for checkpoint={model_ckpt.id}, snippet={body.snippet_id}.")

        predicted_labels = fb_h.collect_predicted_labels_for_snippet(predictions)
        action_value = body.action.value if hasattr(body.action, "value") else body.action
        normalized_labels = fb_h.normalize_feedback_labels(body.labels)

        if action_value == "MODIFY" and not normalized_labels:
            raise ValueError("labels are required when action=MODIFY")

        final_labels = fb_h.resolve_feedback_labels(action_value, predicted_labels, normalized_labels)

        feedback = ALFeedbackEvent(
            dataset_id=body.dataset_id, model_checkpoint_id=model_ckpt.id,
            snippet_id=body.snippet_id, user_id=body.user_id,
            action=action_value, final_labels=final_labels, notes=body.notes,
        )
        self.db.add(feedback)
        self.db.flush()

        if action_value in {"ACCEPT", "MODIFY"} and final_labels:
            ann_h.store_user_labels_for_snippet(self.db, body.dataset_id, body.snippet_id, final_labels, model_ckpt.id, body.user_id)
            if getattr(body, "persist_annotations", True):
                self._try_store_final_annotations(
                    dataset_id=body.dataset_id,
                    snippet_id=body.snippet_id,
                    labels=final_labels,
                    user_id=body.user_id,
                    notes=body.notes,
                )
        elif action_value == "REJECT":
            # Rejection means: remove the user's AL labels for this snippet so it
            # becomes "unlabeled" again (no border) unless it has ground-truth.
            ann_h.delete_user_labels_for_snippet(
                self.db,
                dataset_id=body.dataset_id,
                snippet_id=body.snippet_id,
                user_id=body.user_id,
            )
            self._delete_final_annotations_from_active_learning(
                snippet_id=body.snippet_id,
                user_id=body.user_id,
            )

        self.db.commit()
        self.db.refresh(feedback)

        feedback_count = fb_h.feedback_count_since_retrain(self.db, model_ckpt.id)
        retrain_triggered = False
        auto_retrain_checkpoint_id = None
        auto_retrain_job_id = None
        last_retrain_failed = fb_h.has_failed_child_retrain(self.db, model_ckpt.id)

        # Trigger whenever the threshold is met and no job is currently active or
        # pending for a child checkpoint.  Also skip if the last child retrain
        # already failed — that would create an infinite auto-retry loop and the
        # user should investigate and trigger manually instead.
        if (
            feedback_count >= RETRAIN_AFTER
            and not fb_h.has_active_retrain_job(self.db, model_ckpt.id)
            and not fb_h.has_pending_child_retrain(self.db, model_ckpt.id)
            and not last_retrain_failed
        ):
            retrain_triggered = True
            new_ckpt, retrain_job = self.setup_auto_retrain(model_ckpt.id)
            auto_retrain_checkpoint_id = new_ckpt.id
            auto_retrain_job_id = retrain_job.id
            # Reset to 0: the retrain job was just queued, so the "since-last-retrain"
            # window has restarted. Re-reading the counter here would still return the
            # old high value (old checkpoint has no new COMPLETED job yet).
            feedback_count = 0

        return {
            "id": feedback.id, "model_family_name": model_ckpt.model_family_name,
            "model_checkpoint_id": feedback.model_checkpoint_id, "active_checkpoint_id": model_ckpt.id,
            "snippet_id": feedback.snippet_id, "action": feedback.action,
            "final_labels": feedback.final_labels, "notes": feedback.notes,
            "created_at": feedback.created_at, "feedback_count_since_retrain": feedback_count,
            "retrain_triggered": retrain_triggered,
            "last_retrain_failed": last_retrain_failed,
            "auto_retrain_checkpoint_id": auto_retrain_checkpoint_id,
            "auto_retrain_job_id": auto_retrain_job_id,
        }

    def get_feedback_count_since_retrain(self, dataset_id: int, model_family_name: str) -> dict:
        """
        Return the feedback counter used to gate auto-retrain.

        Counts distinct snippets with feedback since the last completed retrain.
        Returns 0 while a retrain is already pending/running for this model family.
        """
        model_ckpt = ckpt_h.get_active_checkpoint_for_model_family(self.db, dataset_id, model_family_name)
        if model_ckpt is None:
            feedback_count = fb_h.feedback_count_since_retrain(self.db, checkpoint_id=None, dataset_id=dataset_id)
            return {
                "dataset_id": dataset_id,
                "model_family_name": model_family_name,
                "active_checkpoint_id": None,
                "feedback_count_since_retrain": feedback_count,
                "retrain_after": RETRAIN_AFTER,
                "retrain_pending": False,
            }

        # Reset the counter once a child retrain is queued/running.
        if fb_h.has_pending_child_retrain(self.db, model_ckpt.id):
            return {
                "dataset_id": dataset_id,
                "model_family_name": model_family_name,
                "active_checkpoint_id": model_ckpt.id,
                "feedback_count_since_retrain": 0,
                "retrain_after": RETRAIN_AFTER,
                "retrain_pending": True,
            }

        feedback_count = fb_h.feedback_count_since_retrain(self.db, model_ckpt.id)
        return {
            "dataset_id": dataset_id,
            "model_family_name": model_family_name,
            "active_checkpoint_id": model_ckpt.id,
            "feedback_count_since_retrain": feedback_count,
            "retrain_after": RETRAIN_AFTER,
            "retrain_pending": False,
        }

    def _try_store_final_annotations(
        self,
        dataset_id: int,
        snippet_id: int,
        labels: list[str],
        user_id: int | None,
        notes: str | None = None,
    ) -> None:
        """
        Persistence of confirmed labels into the canonical `annotations` table.

        This only works when AL labels are resolvable as either:
          - a namespaced taxon id (e.g. gbif:123 / custom:uuid), or
          - a species name resolvable via the taxonomy matcher.

        If labels are project-specific codes (common in PAM), they are still
        stored in `ALSnippetAnnotation` but may not be representable as
        `Annotation.taxon_id` without a mapping.
        """
        if user_id is None:
            return

        resolved_items: list[dict[str, str]] = []
        for label in labels:
            label = (label or "").strip()
            if not label:
                continue

            # Case 1: label already is a taxon_id
            if taxonomy.parse_taxon_id(label):
                resolved = taxonomy.resolve_taxon_id(label, db_session=self.db)
                if resolved:
                    resolved_items.append(
                        {
                            "taxon_id": label,
                            "resolved_name_snapshot": resolved.get("canonical_name")
                            or resolved.get("scientific_name")
                            or label,
                        }
                    )
                continue

            # Case 2: try to resolve as a species name
            matched = taxonomy.match_species_name(label)
            if matched and matched.get("taxon_id"):
                resolved_items.append(
                    {
                        "taxon_id": matched["taxon_id"],
                        "resolved_name_snapshot": matched.get("canonical_name")
                        or matched.get("scientific_name")
                        or label,
                    }
                )
                continue

            # Case 3: fall back to dataset-specific label codes by creating (or reusing)
            # a minimal custom taxonomy entry *per code*. This keeps `Annotation.taxon_id`
            # valid (custom:<uuid>) even when labels are not GBIF-resolvable.
            custom_taxon_id = self._get_or_create_custom_taxon_id_for_code(
                dataset_id=dataset_id,
                code=label,
                user_id=user_id,
            )
            if custom_taxon_id:
                resolved_items.append(
                    {
                        "taxon_id": custom_taxon_id,
                        "resolved_name_snapshot": label,
                    }
                )

        if not resolved_items:
            return

        # Replace the user's annotations for this snippet with the confirmed set.
        self.db.query(AnnotationModel).filter(
            AnnotationModel.snippet_id == snippet_id,
            AnnotationModel.user_id == user_id,
        ).delete(synchronize_session=False)

        for item in resolved_items:
            self.db.add(
                AnnotationModel(
                    snippet_id=snippet_id,
                    user_id=user_id,
                    taxon_id=item["taxon_id"],
                    resolved_name_snapshot=item["resolved_name_snapshot"],
                    notes=notes,
                    extra_metadata={
                        "source": "active_learning",
                        "dataset_id": dataset_id,
                    },
                )
            )

    def _delete_final_annotations_from_active_learning(
        self,
        snippet_id: int,
        user_id: int | None,
    ) -> None:
        """Delete canonical annotations that were created by the AL flow."""
        if user_id is None:
            return
        rows = (
            self.db.query(AnnotationModel)
            .filter(
                AnnotationModel.snippet_id == snippet_id,
                AnnotationModel.user_id == user_id,
            )
            .all()
        )
        for row in rows:
            meta = getattr(row, "extra_metadata", None) or {}
            if isinstance(meta, dict) and meta.get("source") == "active_learning":
                self.db.delete(row)

    def _get_or_create_custom_taxon_id_for_code(
        self,
        dataset_id: int,
        code: str,
        user_id: int,
    ) -> str | None:
        """
        Create or reuse a CustomTaxonomy row to represent a single label code.

        Why per-code taxonomy?
        - The canonical `Annotation.taxon_id` only accepts `custom:<uuid>`.
        - The current custom taxonomy subsystem does not expose stable per-node IDs
          that fit that constraint.
        - Creating a minimal CustomTaxonomy per code yields a valid, resolvable ID
          and keeps dataset stats consistent.
        """
        code = (code or "").strip()
        if not code:
            return None

        # Resolve a team_id to attach this custom taxonomy to.
        ds = ckpt_h.get_pam_dataset(self.db, dataset_id)
        team_id = getattr(ds, "team_id", None)
        if team_id is None:
            membership = (
                self.db.query(TeamMembership)
                .filter(TeamMembership.user_id == user_id)
                .order_by(TeamMembership.id.asc())
                .first()
            )
            team_id = membership.team_id if membership else None
        # If we cannot associate this dataset/user with a team (e.g. admin-created
        # dataset + user has no team memberships), we still return a valid custom
        # taxon id so we can store canonical final annotations and dataset stats
        # reflect AL progress. In that edge case, taxonomy resolution will not
        # be available (no CustomTaxonomy row), but stats and exports work.
        if team_id is None:
            return f"custom:{uuid.uuid4()}"

        existing = (
            self.db.query(CustomTaxonomy)
            .filter(CustomTaxonomy.team_id == team_id, CustomTaxonomy.name == code)
            .first()
        )
        if existing:
            return existing.taxonomy_id

        taxonomy_id = f"custom:{uuid.uuid4()}"
        taxonomy_data: dict[str, Any] = {
            "nodes": [
                {
                    "id": code,
                    "name": code,
                    "rank": "label_code",
                    "metadata": {"source": "al_auto", "dataset_id": dataset_id},
                    "children": [],
                }
            ]
        }

        row = CustomTaxonomy(
            taxonomy_id=taxonomy_id,
            team_id=team_id,
            created_by_user_id=user_id,
            name=code,
            description=f"Auto-created label code from AL for dataset {dataset_id}",
            taxonomy_data=taxonomy_data,
            status="active",
            is_global=False,
        )
        self.db.add(row)
        self.db.flush()
        return taxonomy_id

    # ==================================================================
    # Inference / predictions
    # ==================================================================

    def get_or_create_predictions(self, body):
        model_ckpt = ckpt_h.get_active_checkpoint_for_model_family(self.db, body.dataset_id, body.model_family_name)

        if model_ckpt is None:
            return self._build_random_snippet_suggestions(body)

        hyper = model_ckpt.hyperparameters or {}
        device = _resolve_device(body.device)
        embedding_model_id = hyper.get("embedding_model_id")
        if embedding_model_id is None:
            # Backward-compat / manual checkpoint registration:
            # if embedding_model_id wasn't stored on the checkpoint, infer it from the most
            # recent COMPLETED embedding job for this dataset + snippet_set.
            from app.models.embedding import EmbeddingJob, EmbeddingJobStatus

            job = (
                self.db.query(EmbeddingJob)
                .filter(
                    EmbeddingJob.dataset_id == body.dataset_id,
                    EmbeddingJob.snippet_set_id == body.snippet_set_id,
                    EmbeddingJob.status == EmbeddingJobStatus.COMPLETED,
                )
                .order_by(EmbeddingJob.created_at.desc())
                .first()
            )
            if job is None:
                raise ValueError(
                    f"Checkpoint {model_ckpt.id} is missing embedding_model_id in hyperparameters, "
                    f"and no COMPLETED embedding job found for dataset_id={body.dataset_id} "
                    f"snippet_set_id={body.snippet_set_id}. Generate embeddings first, or re-register "
                    f"the checkpoint with hyperparameters.embedding_model_id set."
                )

            embedding_model_id = job.embedding_model_id
            # Persist the inferred embedding_model_id so future requests don't need the lookup.
            model_ckpt.hyperparameters = {**hyper, "embedding_model_id": int(embedding_model_id)}
            self.db.commit()

        threshold, density_k, wu, wd, wr = inf_h.resolve_inference_params(
            body.threshold, body.density_k, body.composite_wu, body.composite_wd, body.composite_wr,
        )

        predictions = inf_h.get_predictions_for_checkpoint_and_snippet_set(self.db, model_ckpt.id, body.snippet_set_id)

        if not predictions or body.force_refresh:
            X, snippet_rows = data_h.load_embeddings(self.db, body.snippet_set_id, embedding_model_id)

            # Resolve label order *before* loading the model so we can fall back to
            # rebuilding the classifier when legacy checkpoint metadata is missing.
            label_order = hyper.get("label_order")
            if not label_order and getattr(model_ckpt, "label_config_path", None):
                try:
                    label_order = ckpt_h.load_species_from_label_config(model_ckpt.label_config_path)
                    model_ckpt.hyperparameters = {**(model_ckpt.hyperparameters or {}), "label_order": label_order}
                    self.db.commit()
                except Exception as e:
                    raise ValueError(
                        f"Failed to load label_config_path='{model_ckpt.label_config_path}' "
                        f"for checkpoint {model_ckpt.id}: {e}"
                    )

            # Load the model; if legacy checkpoint is missing metadata like n_dim,
            # rebuild the architecture from embeddings + label_order and load weights.
            try:
                model = ckpt_h.load_model_from_checkpoint(model_ckpt, device=device)
            except KeyError as e:
                if str(e).strip("'\"") != "n_dim":
                    raise
                if not label_order:
                    raise ValueError(
                        f"Checkpoint {model_ckpt.id} is missing n_dim metadata and no label_order is available. "
                        "Set checkpoint.label_config_path (recommended) or hyperparameters.label_order."
                    )
                import torch

                payload = torch.load(model_ckpt.checkpoint_path, map_location=device)
                state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        f"Legacy checkpoint {model_ckpt.id} missing 'state_dict'; cannot rebuild classifier."
                    )

                model = ckpt_h.make_model(model_ckpt.model_type)
                model.create_classifier(
                    n_dim=int(X.shape[1]),
                    num_classes=len(label_order),
                    hidden_dim=int((model_ckpt.hyperparameters or {}).get("hidden_dim"))
                    if (model_ckpt.hyperparameters or {}).get("hidden_dim") is not None
                    else None,
                    dropout=float((model_ckpt.hyperparameters or {}).get("dropout"))
                    if (model_ckpt.hyperparameters or {}).get("dropout") is not None
                    else None,
                )
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                model.label_order = label_order

            # If the checkpoint stored label order in the .pt file, prefer it (and persist).
            label_order = getattr(model, "label_order", None) or label_order
            if not label_order:
                raise ValueError(
                    f"No label_order found for checkpoint {model_ckpt.id}. "
                    "Provide hyperparameters.label_order or set label_config_path."
                )
            if not hyper.get("label_order"):
                model_ckpt.hyperparameters = {**(model_ckpt.hyperparameters or {}), "label_order": label_order}
                self.db.commit()

            labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(self.db, model_ckpt.dataset_id)
            inf_h.run_and_store_inference(
                self.db, model_ckpt.dataset_id, model_ckpt, model, X, snippet_rows, label_order, labeled_ids,
                threshold, density_k, wu, wd, wr,
            )
            self.db.commit()

            predictions = inf_h.get_predictions_for_checkpoint_and_snippet_set(self.db, model_ckpt.id, body.snippet_set_id)

        if not body.sample_suggestion:
            return {
                "mode": "predictions", "model_family_name": body.model_family_name,
                "used_checkpoint_id": model_ckpt.id, "total_predictions": len(predictions),
                "returned_count": len(predictions), "suggestion_strategy": body.suggestion_strategy,
                "k": body.k, "rows": predictions,
            }

        strategy = body.suggestion_strategy.value if hasattr(body.suggestion_strategy, "value") else body.suggestion_strategy
        k = body.k or 20

        annotated_ids = ann_h.get_annotated_snippet_ids_for_snippet_set(self.db, model_ckpt.dataset_id, body.snippet_set_id)
        ranked = inf_h.rank_prediction_suggestions(self.db, model_ckpt.dataset_id, body.snippet_set_id, predictions, strategy, annotated_ids)

        return {
            "mode": "suggestions", "model_family_name": body.model_family_name,
            "used_checkpoint_id": model_ckpt.id, "total_predictions": len(predictions),
            "returned_count": min(k, len(ranked)), "suggestion_strategy": body.suggestion_strategy,
            "k": k, "rows": ranked[:k],
        }

    # ==================================================================
    # User Study-mode helpers (labeled pool + ground-truth label lookup)
    # ==================================================================

    def list_labeled_snippets(
        self,
        dataset_id: int,
        snippet_set_id: Optional[int] = None,
        scope: str = "any",
        user_id: Optional[int] = None,
    ) -> List[int]:
        """Snippet IDs that already have at least one annotation."""
        if scope == "user":
            if user_id is None:
                return []
            if snippet_set_id is None:
                return sorted(
                    ann_h.get_user_labeled_snippet_ids_for_dataset(self.db, dataset_id, user_id)
                )
            return sorted(
                ann_h.get_user_labeled_snippet_ids_for_snippet_set(
                    self.db, dataset_id, snippet_set_id, user_id
                )
            )
        if snippet_set_id is None:
            return sorted(ann_h.get_labeled_snippet_ids_for_dataset(self.db, dataset_id))
        return sorted(
            ann_h.get_labeled_snippet_ids_for_snippet_set(self.db, dataset_id, snippet_set_id)
        )

    def list_snippet_labels(
        self,
        dataset_id: int,
        snippet_set_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Per-snippet ground-truth / user labels — feeds the `actual_label` color filter."""
        labels_by_snippet = ann_h.get_labels_by_snippet(self.db, dataset_id, snippet_set_id)
        return [
            {"snippet_id": sid, "labels": labels}
            for sid, labels in sorted(labels_by_snippet.items())
        ]

    # ==================================================================
    # Manual retrain  (sync — kept for backward compat)
    # ==================================================================

    def manual_retrain(self, body) -> ALModelCheckpoint:
        parent_ckpt = ckpt_h.get_active_checkpoint_for_model_family(self.db, body.dataset_id, body.model_family_name)
        if parent_ckpt is None:
            raise ValueError(
                f"No active checkpoint found for dataset={body.dataset_id}, "
                f"model_family_name='{body.model_family_name}'."
            )
        hyper = parent_ckpt.hyperparameters or {}

        dataset_id = parent_ckpt.dataset_id
        snippet_set_id = hyper.get("resolved_snippet_set_id")
        embedding_model_id = hyper.get("embedding_model_id")
        label_order = hyper.get("label_order")

        if snippet_set_id is None:
            raise ValueError("Parent checkpoint missing resolved_snippet_set_id.")
        if embedding_model_id is None:
            raise ValueError("Parent checkpoint missing embedding_model_id.")
        if not label_order:
            raise ValueError("Parent checkpoint missing label_order.")

        epochs = body.epochs if body.epochs is not None else int(hyper.get("epochs", 20))
        lr = body.learning_rate if body.learning_rate is not None else float(hyper.get("learning_rate", 1e-3))
        bs = body.batch_size if body.batch_size is not None else int(hyper.get("batch_size", DEFAULT_BATCH_SIZE))
        is_mlp = parent_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL.value

        hd = int(hyper.get("hidden_dim")) if is_mlp and hyper.get("hidden_dim") is not None else None
        do = float(hyper.get("dropout")) if is_mlp and hyper.get("dropout") is not None else None
        dev = _resolve_device(body.device)

        new_version = f"{parent_ckpt.version}_manual_{int(datetime.now(timezone.utc).timestamp())}"

        new_ckpt = ALModelCheckpoint(
            dataset_id=dataset_id, model_family_name=parent_ckpt.model_family_name,
            version=new_version, checkpoint_path="", label_config_path=parent_ckpt.label_config_path,
            model_type=parent_ckpt.model_type,
            hyperparameters={**hyper, "training_mode": "manual_retrain", "parent_checkpoint_id": parent_ckpt.id,
                             "epochs": epochs, "learning_rate": lr, "batch_size": bs,
                             "hidden_dim": hd, "dropout": do, "device": dev},
            is_base=0, parent_checkpoint_id=parent_ckpt.id, status=ALModelStatus.LOADING,
        )
        self.db.add(new_ckpt)
        self.db.flush()

        recent_fb = fb_h.feedback_count_since_retrain(self.db, parent_ckpt.id)
        job = ALRetrainJob(
            model_checkpoint_id=new_ckpt.id, dataset_id=dataset_id, trigger="manual",
            feedback_count=recent_fb, status=ALRetrainStatus.PENDING,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(new_ckpt)
        self.db.refresh(job)

        try:
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            fb_h.sync_feedback_events_to_annotations(self.db, parent_ckpt.id)

            annotations_by_snippet = ann_h.get_trusted_annotations(self.db, dataset_id)
            if not annotations_by_snippet:
                raise ValueError("No trusted annotations available for retraining.")

            X, snippet_rows = data_h.load_embeddings(self.db, snippet_set_id, embedding_model_id)
            snippet_ids = [r["snippet_id"] for r in snippet_rows]

            keep = [i for i, sid in enumerate(snippet_ids) if sid in annotations_by_snippet]
            if not keep:
                raise ValueError("No embeddings found for snippets with trusted annotations.")

            X_train = X[keep]
            train_sids = [snippet_ids[i] for i in keep]
            y_train = ann_h.build_multihot_from_annotations(train_sids, label_order, annotations_by_snippet)

            keep_rows = y_train.sum(axis=1) > 0
            X_train, y_train = X_train[keep_rows], y_train[keep_rows]
            if X_train.shape[0] == 0:
                raise ValueError("No training rows remain after filtering empty rows.")

            is_mlp = parent_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL or parent_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL.value

            hd = int(hyper.get("hidden_dim")) if is_mlp and hyper.get("hidden_dim") is not None else None
            do = float(hyper.get("dropout")) if is_mlp and hyper.get("dropout") is not None else None
            model = ckpt_h.make_model(parent_ckpt.model_type)
            model.create_classifier(
                n_dim=X_train.shape[1],
                num_classes=y_train.shape[1],
                hidden_dim=hd,
                dropout=do,
            )
            model.to(dev)

            train_metrics = model.fit(X=X_train, y=y_train, epochs=epochs, learning_rate=lr, batch_size=bs, device=dev)

            cp = ckpt_h.make_checkpoint_path(dataset_id, new_ckpt.model_family_name, new_ckpt.version, new_ckpt.id)
            ckpt_h.save_classifier_checkpoint(model, cp, hd, do, label_order)

            new_ckpt.checkpoint_path = cp
            new_ckpt.status = ALModelStatus.AVAILABLE
            new_ckpt.hyperparameters = {
                **(new_ckpt.hyperparameters or {}),
                "n_dim": int(X_train.shape[1]), "num_classes": int(y_train.shape[1]),
                "train_samples": int(X_train.shape[0]), "label_order": label_order,
                "resolved_snippet_set_id": snippet_set_id, "embedding_model_id": embedding_model_id,
            }

            inference_metrics = None
            if body.run_inference:
                labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(self.db, dataset_id)
                inference_metrics = inf_h.run_and_store_inference(
                    self.db, dataset_id, new_ckpt, model, X, snippet_rows, label_order, labeled_ids,
                    body.threshold, body.density_k, body.composite_wu, body.composite_wd, body.composite_wr,
                )

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {
                "new_checkpoint_id": new_ckpt.id, "new_checkpoint_path": cp,
                "train_samples": int(X_train.shape[0]), "num_classes": int(y_train.shape[1]),
                "inference_metrics": inference_metrics, **train_metrics,
            }
            ckpt_h.set_active_family_checkpoint(self.db, dataset_id, parent_ckpt.model_family_name, new_ckpt.id)
            self.db.commit()
            self.db.refresh(new_ckpt)
            return new_ckpt

        except Exception as e:
            logger.exception("Manual retraining failed.")
            self._cleanup_failed_training_checkpoint(new_ckpt, job, e)
            raise

    # ==================================================================
    # Async setup / execute  (Celery tasks call execute_*)
    # ==================================================================

    def setup_train_from_scratch(self, body: ALTrainFromScratchRequest) -> tuple[ALModelCheckpoint, ALRetrainJob]:
        ds = ckpt_h.get_pam_dataset(self.db, body.dataset_id)
        snippet_set_id = body.snippet_set_id or ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError("No snippet_set_id provided and dataset has no default_snippet_set_id.")

        is_mlp = body.model_type == ALModelType.PAM_MLP_MULTILABEL
        device = _resolve_device(body.device)

        hidden_dim = body.hidden_dim if is_mlp else None
        dropout = body.dropout if is_mlp else None

        model_ckpt = ALModelCheckpoint(
            dataset_id=body.dataset_id, model_family_name=body.model_family_name,
            version=body.version, checkpoint_path="",
            label_config_path=body.label_config_path,
            model_type=body.model_type.value if hasattr(body.model_type, "value") else body.model_type,
            hyperparameters={
                "training_mode": "cold_start", "embedding_model_id": body.embedding_model_id,
                "snippet_set_id": snippet_set_id,
                "metadata_path": os.path.join(DATA_ROOT, body.metadata_path),
                "label_config_path": os.path.join(DATA_ROOT, body.label_config_path),
                "min_samples_per_class": body.min_samples_per_class,
                "max_samples_per_class": body.max_samples_per_class,
                "epochs": body.epochs, "learning_rate": body.learning_rate,
                "batch_size": body.batch_size, "hidden_dim": hidden_dim,
                "dropout": dropout, "device": device,
                "run_inference": body.run_inference, "threshold": body.threshold,
                "density_k": body.density_k, "composite_wu": body.composite_wu,
                "composite_wd": body.composite_wd, "composite_wr": body.composite_wr,
            },
            is_base=1, parent_checkpoint_id=None, status=ALModelStatus.LOADING,
        )
        self.db.add(model_ckpt)
        self.db.flush()

        job = ALRetrainJob(
            model_checkpoint_id=model_ckpt.id, dataset_id=body.dataset_id,
            trigger="cold_start", feedback_count=0, status=ALRetrainStatus.PENDING,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(model_ckpt)
        self.db.refresh(job)
        return model_ckpt, job

    def execute_train_from_scratch(self, checkpoint_id: int, job_id: int) -> ALModelCheckpoint:
        model_ckpt = ckpt_h.get_checkpoint(self.db, checkpoint_id)
        if model_ckpt is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        job = self.db.query(ALRetrainJob).filter(ALRetrainJob.id == job_id).first()
        if job is None:
            raise ValueError(f"Retrain job {job_id} not found.")

        hyper = model_ckpt.hyperparameters or {}
        ds = ckpt_h.get_pam_dataset(self.db, model_ckpt.dataset_id)
        snippet_set_id = hyper["snippet_set_id"]
        species_list = ckpt_h.load_species_from_label_config(hyper["label_config_path"])

        try:
            logger.info(
                "Starting cold-start execution checkpoint_id=%d job_id=%d dataset_id=%d snippet_set_id=%s",
                checkpoint_id,
                job_id,
                model_ckpt.dataset_id,
                snippet_set_id,
            )
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            X, snippet_rows = data_h.load_embeddings(self.db, snippet_set_id, hyper["embedding_model_id"])
            logger.info(
                "Loaded embeddings for cold-start checkpoint_id=%d rows=%d",
                checkpoint_id,
                len(snippet_rows),
            )
            gt_index = data_h.load_ground_truth_metadata(hyper["metadata_path"], species_list, ["train"])
            X_train, y_train, used_sids = data_h.align_embeddings_and_labels(X, snippet_rows, gt_index, species_list)
            logger.info(
                "Aligned training set for cold-start checkpoint_id=%d samples=%d classes=%d",
                checkpoint_id,
                int(y_train.shape[0]),
                int(y_train.shape[1]),
            )
            model = ckpt_h.make_model(model_ckpt.model_type)
            X_train, y_train, labeled_sids, used_sp, excl_sp, class_counts = model.filter_and_balance_classes(
                X=X_train, y=y_train, snippet_ids=used_sids, species_list=species_list,
                min_samples_per_class=hyper.get("min_samples_per_class", 1),
                max_samples_per_class=hyper.get("max_samples_per_class"),
            )

            if y_train.shape[0] == 0:
                raise ValueError("No training samples remain.")
            if y_train.shape[1] == 0:
                raise ValueError("No species remain.")

            n_dim, num_classes = X_train.shape[1], y_train.shape[1]
            is_mlp = model_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL or model_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL.value

            hd = int(hyper.get("hidden_dim")) if is_mlp and hyper.get("hidden_dim") is not None else None
            do = float(hyper.get("dropout")) if is_mlp and hyper.get("dropout") is not None else None
            dev = _resolve_device(hyper.get("device"))

            model.create_classifier(n_dim=n_dim, num_classes=num_classes, hidden_dim=hd, dropout=do)
            model.to(dev)
            train_metrics = model.fit(X=X_train, y=y_train, epochs=int(hyper.get("epochs", 20)),
                                      learning_rate=float(hyper.get("learning_rate", 1e-3)),
                                      batch_size=int(hyper.get("batch_size", DEFAULT_BATCH_SIZE)), device=dev)
            logger.info(
                "Finished cold-start model fit checkpoint_id=%d train_samples=%d num_classes=%d",
                checkpoint_id,
                int(X_train.shape[0]),
                int(num_classes),
            )

            cp = ckpt_h.make_checkpoint_path(ds.id, model_ckpt.model_family_name, model_ckpt.version, model_ckpt.id)
            lcp = ckpt_h.make_label_config_path(ds.id, model_ckpt.model_family_name, model_ckpt.version, model_ckpt.id)
            ckpt_h.save_label_config(lcp, used_sp)
            ckpt_h.save_classifier_checkpoint(model, cp, hd, do, used_sp)

            model_ckpt.checkpoint_path = cp
            model_ckpt.label_config_path = lcp
            model_ckpt.status = ALModelStatus.AVAILABLE
            model_ckpt.hyperparameters = {**(model_ckpt.hyperparameters or {}),
                "resolved_snippet_set_id": snippet_set_id, "n_dim": n_dim, "num_classes": num_classes,
                "train_samples": int(X_train.shape[0]), "label_order": used_sp,
                "used_species": used_sp, "excluded_species": excl_sp, "class_counts": class_counts}

            ann_h.store_snippet_annotations(self.db, model_ckpt.dataset_id, labeled_sids, y_train, used_sp, ALAnnotationSource.GROUND_TRUTH, model_ckpt.id)

            inference_metrics = None
            if hyper.get("run_inference"):
                labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(self.db, model_ckpt.dataset_id)
                inference_metrics = inf_h.run_and_store_inference(
                    self.db, model_ckpt.dataset_id, model_ckpt, model, X, snippet_rows, used_sp, labeled_ids,
                    hyper.get("threshold"), hyper.get("density_k"),
                    hyper.get("composite_wu"), hyper.get("composite_wd"), hyper.get("composite_wr"))
                logger.info("Completed initial inference for cold-start checkpoint_id=%d", checkpoint_id)

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {"new_checkpoint_id": model_ckpt.id, "new_checkpoint_path": cp,
                "aligned_snippet_count": len(used_sids), "train_samples": int(X_train.shape[0]),
                "num_classes": int(num_classes), "used_species": used_sp, "excluded_species": excl_sp,
                "class_counts": class_counts, "inference_metrics": inference_metrics, **train_metrics}

            ckpt_h.set_active_family_checkpoint(self.db, model_ckpt.dataset_id, model_ckpt.model_family_name, model_ckpt.id)
            self.db.commit()
            self.db.refresh(model_ckpt)
            logger.info(
                "Cold-start execution completed checkpoint_id=%d job_id=%d",
                checkpoint_id,
                job_id,
            )
            return model_ckpt

        except Exception as e:
            logger.exception("execute_train_from_scratch failed checkpoint_id=%d", checkpoint_id)
            self._cleanup_failed_training_checkpoint(model_ckpt, job, e)
            raise

    def setup_manual_retrain(self, body) -> tuple[ALModelCheckpoint, ALRetrainJob]:
        parent_ckpt = ckpt_h.get_active_checkpoint_for_model_family(self.db, body.dataset_id, body.model_family_name)
        if parent_ckpt is None:
            raise ValueError(
                f"No active checkpoint found for dataset={body.dataset_id}, "
                f"model_family_name='{body.model_family_name}'."
            )
        hyper = parent_ckpt.hyperparameters or {}

        snippet_set_id = hyper.get("resolved_snippet_set_id")
        if snippet_set_id is None:
            ds = ckpt_h.get_pam_dataset(self.db, parent_ckpt.dataset_id)
            snippet_set_id = ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError(
                f"Parent checkpoint {parent_ckpt.id} missing resolved_snippet_set_id and "
                "dataset has no default_snippet_set_id."
            )

        epochs = body.epochs if body.epochs is not None else int(hyper.get("epochs", 20))
        lr = body.learning_rate if body.learning_rate is not None else float(hyper.get("learning_rate", 1e-3))
        bs = body.batch_size if body.batch_size is not None else int(hyper.get("batch_size", DEFAULT_BATCH_SIZE))
        is_mlp = parent_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL or parent_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL.value
        hd = int(hyper.get("hidden_dim")) if is_mlp and hyper.get("hidden_dim") is not None else None
        do = float(hyper.get("dropout")) if is_mlp and hyper.get("dropout") is not None else None
        dev = _resolve_device(body.device)

        new_version = f"{parent_ckpt.version}_manual_{int(datetime.now(timezone.utc).timestamp())}"
        new_ckpt = ALModelCheckpoint(
            dataset_id=parent_ckpt.dataset_id, model_family_name=parent_ckpt.model_family_name,
            version=new_version, checkpoint_path="", label_config_path=parent_ckpt.label_config_path,
            model_type=parent_ckpt.model_type,
            hyperparameters={**hyper, "training_mode": "manual_retrain", "parent_checkpoint_id": parent_ckpt.id,
                "resolved_snippet_set_id": snippet_set_id,
                "epochs": epochs, "learning_rate": lr, "batch_size": bs, "hidden_dim": hd, "dropout": do, "device": dev,
                "run_inference": body.run_inference, "threshold": body.threshold, "density_k": body.density_k,
                "composite_wu": body.composite_wu, "composite_wd": body.composite_wd, "composite_wr": body.composite_wr},
            is_base=0, parent_checkpoint_id=parent_ckpt.id, status=ALModelStatus.LOADING,
        )
        self.db.add(new_ckpt)
        self.db.flush()

        job = ALRetrainJob(
            model_checkpoint_id=new_ckpt.id, dataset_id=parent_ckpt.dataset_id, trigger="manual",
            feedback_count=fb_h.feedback_count_since_retrain(self.db, parent_ckpt.id),
            status=ALRetrainStatus.PENDING, started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(new_ckpt)
        self.db.refresh(job)
        return new_ckpt, job

    def execute_manual_retrain(self, checkpoint_id: int, job_id: int) -> ALModelCheckpoint:
        return self._execute_retrain(checkpoint_id, job_id, sync_feedback=True)

    def setup_auto_retrain(self, parent_checkpoint_id: int) -> tuple[ALModelCheckpoint, ALRetrainJob]:
        parent_ckpt = ckpt_h.get_checkpoint(self.db, parent_checkpoint_id)
        if parent_ckpt is None:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} not found.")

        hyper = parent_ckpt.hyperparameters or {}
    # Auto-retrain checkpoints must carry all required hyperparameters so
    # the worker can execute without additional lookups.
        snippet_set_id = hyper.get("resolved_snippet_set_id")
        if snippet_set_id is None:
            # Fallback for older checkpoints that don't persist this field.
            ds = ckpt_h.get_pam_dataset(self.db, parent_ckpt.dataset_id)
            snippet_set_id = ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError(
                f"Parent checkpoint {parent_checkpoint_id} missing resolved_snippet_set_id and "
                "dataset has no default_snippet_set_id."
            )

        embedding_model_id = hyper.get("embedding_model_id")
        if embedding_model_id is None:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} missing embedding_model_id.")

        label_order = hyper.get("label_order")
        if not label_order:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} missing label_order.")

        new_version = f"{parent_ckpt.version}_r{int(datetime.now(timezone.utc).timestamp())}"

        new_ckpt = ALModelCheckpoint(
            dataset_id=parent_ckpt.dataset_id, model_family_name=parent_ckpt.model_family_name,
            version=new_version, checkpoint_path="", label_config_path=parent_ckpt.label_config_path,
            model_type=parent_ckpt.model_type,
            hyperparameters={
                **hyper,
                "training_mode": "feedback_retrain",
                "parent_checkpoint_id": parent_checkpoint_id,
                "resolved_snippet_set_id": snippet_set_id,
                "embedding_model_id": embedding_model_id,
                "label_order": label_order,
                "device": _resolve_device(),
                # Always run inference after auto-retrain so predictions exist for
                # the new checkpoint immediately.  The parent may have been created
                # with run_inference=False (e.g. cold-start default), but retrain
                # checkpoints must always generate predictions.
                "run_inference": True,
            },
            is_base=0, parent_checkpoint_id=parent_checkpoint_id, status=ALModelStatus.LOADING,
        )
        self.db.add(new_ckpt)
        self.db.flush()

        job = ALRetrainJob(
            model_checkpoint_id=new_ckpt.id, dataset_id=parent_ckpt.dataset_id, trigger="auto_feedback",
            feedback_count=fb_h.feedback_count_since_retrain(self.db, parent_checkpoint_id),
            status=ALRetrainStatus.PENDING, started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(new_ckpt)
        self.db.refresh(job)
        return new_ckpt, job

    def execute_auto_retrain(self, checkpoint_id: int, job_id: int) -> ALModelCheckpoint:
        return self._execute_retrain(checkpoint_id, job_id, sync_feedback=False)

    # ==================================================================
    # Shared retrain execution (used by both manual and auto)
    # ==================================================================

    def _execute_retrain(self, checkpoint_id: int, job_id: int, *, sync_feedback: bool) -> ALModelCheckpoint:
        new_ckpt = ckpt_h.get_checkpoint(self.db, checkpoint_id)
        if new_ckpt is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        job = self.db.query(ALRetrainJob).filter(ALRetrainJob.id == job_id).first()
        if job is None:
            raise ValueError(f"Retrain job {job_id} not found.")

        hyper = new_ckpt.hyperparameters or {}
        dataset_id = new_ckpt.dataset_id
        snippet_set_id = hyper.get("resolved_snippet_set_id")
        embedding_model_id = hyper.get("embedding_model_id")
        label_order = hyper.get("label_order")
        parent_checkpoint_id = hyper.get("parent_checkpoint_id")

        try:
            logger.info(
                "Starting retrain execution checkpoint_id=%d job_id=%d dataset_id=%d sync_feedback=%s",
                checkpoint_id,
                job_id,
                dataset_id,
                sync_feedback,
            )
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            # Validate required hyperparameters — inside the try block so
            # _cleanup_failed_training_checkpoint marks the checkpoint as ERROR
            # and the job as FAILED rather than leaving them in LOADING/RUNNING.
            if snippet_set_id is None:
                raise ValueError(
                    f"Checkpoint {checkpoint_id} missing resolved_snippet_set_id "
                    f"(hyperparameters keys: {list(hyper.keys())})."
                )
            if embedding_model_id is None:
                raise ValueError(
                    f"Checkpoint {checkpoint_id} missing embedding_model_id "
                    f"(hyperparameters keys: {list(hyper.keys())})."
                )
            if not label_order:
                raise ValueError(
                    f"Checkpoint {checkpoint_id} missing label_order "
                    f"(hyperparameters keys: {list(hyper.keys())})."
                )

            if sync_feedback and parent_checkpoint_id:
                fb_h.sync_feedback_events_to_annotations(self.db, parent_checkpoint_id)
                logger.info(
                    "Synchronized feedback annotations for retrain checkpoint_id=%d parent_checkpoint_id=%d",
                    checkpoint_id,
                    parent_checkpoint_id,
                )

            annotations_by_snippet = ann_h.get_trusted_annotations(self.db, dataset_id)
            logger.info(
                "Loaded trusted annotations for retrain checkpoint_id=%d annotated_snippets=%d",
                checkpoint_id,
                len(annotations_by_snippet),
            )
            if not annotations_by_snippet:
                raise ValueError(
                    f"No trusted annotations available for retraining "
                    f"(dataset_id={dataset_id}). "
                    "Ensure ground-truth or user labels exist before retraining."
                )

            X, snippet_rows = data_h.load_embeddings(self.db, snippet_set_id, embedding_model_id)
            snippet_ids = [r["snippet_id"] for r in snippet_rows]
            logger.info(
                "Loaded embeddings for retrain checkpoint_id=%d rows=%d",
                checkpoint_id,
                len(snippet_rows),
            )

            keep = [i for i, sid in enumerate(snippet_ids) if sid in annotations_by_snippet]
            if not keep:
                raise ValueError(
                    f"No embeddings found for snippets with trusted annotations "
                    f"(annotated snippet ids: {list(annotations_by_snippet.keys())[:10]}..., "
                    f"snippet_set snippet ids sample: {snippet_ids[:10]}...)."
                )

            X_train = X[keep]
            train_sids = [snippet_ids[i] for i in keep]
            y_train = ann_h.build_multihot_from_annotations(train_sids, label_order, annotations_by_snippet)

            keep_rows = y_train.sum(axis=1) > 0
            X_train, y_train = X_train[keep_rows], y_train[keep_rows]
            if X_train.shape[0] == 0:
                # Samples found but ALL have zero label vectors — most likely the
                # stored annotation labels don't match label_order.
                sample_labels = set()
                for sid in train_sids[:5]:
                    sample_labels.update(annotations_by_snippet.get(sid, set()))
                raise ValueError(
                    f"No training rows remain after filtering empty rows "
                    f"(train_sids={len(train_sids)}, label_order={label_order}, "
                    f"sample annotation labels={list(sample_labels)})."
                )
            logger.info(
                "Prepared retrain dataset checkpoint_id=%d train_samples=%d num_classes=%d",
                checkpoint_id,
                int(X_train.shape[0]),
                int(y_train.shape[1]),
            )

            is_mlp = new_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL or new_ckpt.model_type == ALModelType.PAM_MLP_MULTILABEL.value

            hd = int(hyper.get("hidden_dim")) if is_mlp and hyper.get("hidden_dim") is not None else None
            do = float(hyper.get("dropout")) if is_mlp and hyper.get("dropout") is not None else None
            dev = _resolve_device(hyper.get("device"))

            model = ckpt_h.make_model(new_ckpt.model_type)
            model.create_classifier(
                n_dim=X_train.shape[1],
                num_classes=y_train.shape[1],
                hidden_dim=hd,
                dropout=do,
            )
            model.to(dev)
            train_metrics = model.fit(X=X_train, y=y_train,
                epochs=int(hyper.get("epochs", 20)), learning_rate=float(hyper.get("learning_rate", 1e-3)),
                batch_size=int(hyper.get("batch_size", DEFAULT_BATCH_SIZE)), device=dev)
            logger.info("Finished retrain model fit checkpoint_id=%d", checkpoint_id)

            cp = ckpt_h.make_checkpoint_path(dataset_id, new_ckpt.model_family_name, new_ckpt.version, new_ckpt.id)
            ckpt_h.save_classifier_checkpoint(model, cp, hd, do, label_order)

            new_ckpt.checkpoint_path = cp
            new_ckpt.status = ALModelStatus.AVAILABLE
            new_ckpt.hyperparameters = {**(new_ckpt.hyperparameters or {}),
                "n_dim": int(X_train.shape[1]), "num_classes": int(y_train.shape[1]),
                "train_samples": int(X_train.shape[0]), "label_order": label_order,
                "resolved_snippet_set_id": snippet_set_id, "embedding_model_id": embedding_model_id}

            labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(self.db, dataset_id)
            inference_metrics = None
            if hyper.get("run_inference", True):
                inference_metrics = inf_h.run_and_store_inference(
                    self.db, dataset_id, new_ckpt, model, X, snippet_rows, label_order, labeled_ids,
                    hyper.get("threshold"), hyper.get("density_k"),
                    hyper.get("composite_wu"), hyper.get("composite_wd"), hyper.get("composite_wr"))
                logger.info("Completed post-retrain inference checkpoint_id=%d", checkpoint_id)

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {"new_checkpoint_id": new_ckpt.id, "new_checkpoint_path": cp,
                "train_samples": int(X_train.shape[0]), "num_classes": int(y_train.shape[1]),
                "inference_metrics": inference_metrics, **train_metrics}

            ckpt_h.set_active_family_checkpoint(self.db, dataset_id, new_ckpt.model_family_name, new_ckpt.id)
            self.db.commit()
            self.db.refresh(new_ckpt)
            logger.info("Retrain execution completed checkpoint_id=%d job_id=%d", checkpoint_id, job_id)
            return new_ckpt

        except Exception as e:
            logger.exception("Retrain failed checkpoint_id=%d", checkpoint_id)
            self._cleanup_failed_training_checkpoint(new_ckpt, job, e)
            raise

    # ==================================================================
    # Job polling
    # ==================================================================

    def get_retrain_job(self, job_id: int) -> Optional[ALRetrainJob]:
        return self.db.query(ALRetrainJob).filter(ALRetrainJob.id == job_id).first()

    def list_retrain_jobs(self, dataset_id: Optional[int] = None, limit: int = 50) -> list[ALRetrainJob]:
        q = self.db.query(ALRetrainJob)
        if dataset_id is not None:
            q = q.filter(ALRetrainJob.dataset_id == dataset_id)
        return q.order_by(ALRetrainJob.created_at.desc()).limit(limit).all()

    # ==================================================================
    # Bootstrap helpers (no checkpoint available)
    # ==================================================================

    def _build_random_snippet_suggestions(self, body) -> dict:
        k = body.k or 20

        snippets = (
            self.db.query(Snippet)
            .filter(Snippet.snippet_set_id == body.snippet_set_id)
            .all()
        )
        if not snippets:
            raise ValueError(f"No snippets found for snippet_set_id={body.snippet_set_id}.")

        sampled = random.sample(snippets, min(k, len(snippets)))

        rows = [
            ALPredictionResponse(
                snippet_id=snippet.id,
                predicted_labels=None,
                predicted_probabilities=None,
                uncertainty=None,
                diversity=None,
                density=None,
                composite_score=None,
            )
            for snippet in sampled
        ]

        return {
            "mode": "suggestions",
            "model_family_name": body.model_family_name,
            "used_checkpoint_id": None,
            "total_predictions": 0,
            "returned_count": len(rows),
            "suggestion_strategy": SamplingMode.RANDOM,
            "k": k,
            "rows": rows,
        }

    def _submit_bootstrap_feedback(self, body: ALFeedbackSubmit) -> dict:
        normalized_labels = fb_h.normalize_feedback_labels(body.labels)
        if not normalized_labels:
            raise ValueError(
                "Initial feedback must include explicit labels when no active checkpoint exists."
            )

        if body.embedding_model_id is None:
            raise ValueError(
                "embedding_model_id is required when submitting bootstrap feedback."
            )

        snippet = (
            self.db.query(Snippet)
            .filter(Snippet.id == body.snippet_id)
            .one_or_none()
        )
        if snippet is None:
            raise ValueError(f"Snippet {body.snippet_id} not found.")

        feedback = ALFeedbackEvent(
            dataset_id=body.dataset_id,
            model_checkpoint_id=None,
            snippet_id=body.snippet_id,
            user_id=body.user_id,
            action="MODIFY",
            final_labels=normalized_labels,
            notes=body.notes,
        )
        self.db.add(feedback)
        self.db.flush()

        ann_h.store_user_labels_for_snippet(
            db=self.db,
            dataset_id=body.dataset_id,
            snippet_id=body.snippet_id,
            labels=normalized_labels,
            model_checkpoint_id=None,
            user_id=body.user_id,
        )

        self.db.commit()
        self.db.refresh(feedback)

        feedback_count = fb_h.feedback_count_since_retrain(
            db=self.db,
            checkpoint_id=None,
            dataset_id=body.dataset_id,
        )

        retrain_triggered = False
        active_checkpoint_id = None

        if feedback_count >= RETRAIN_AFTER:
            retrain_triggered = True

            train_body = ALTrainFromScratchRequest(
                dataset_id=body.dataset_id,
                snippet_set_id=snippet.snippet_set_id,
                embedding_model_id=body.embedding_model_id,
                metadata_path=None,
                label_config_path=None,
                model_family_name=body.model_family_name,
                version="v0",
                model_type=body.model_type,
                epochs=DEFAULT_EPOCHS,
                learning_rate=DEFAULT_LEARNING_RATE,
                batch_size=DEFAULT_BATCH_SIZE,
                hidden_dim=DEFAULT_HIDDEN_DIM,
                dropout=DEFAULT_DROPOUT,
                device=_resolve_device(),
                run_inference=True,
                threshold=DEFAULT_INFERENCE_THRESHOLD,
                density_k=DEFAULT_DENSITY_K,
                composite_wu=DEFAULT_COMPOSITE_WU,
                composite_wd=DEFAULT_COMPOSITE_WD,
                composite_wr=DEFAULT_COMPOSITE_WR,
            )

            new_ckpt = self.train_from_scratch(train_body)
            active_checkpoint_id = new_ckpt.id

        return {
            "id": feedback.id,
            "model_family_name": body.model_family_name,
            "model_checkpoint_id": None,
            "active_checkpoint_id": active_checkpoint_id,
            "snippet_id": feedback.snippet_id,
            "action": feedback.action,
            "final_labels": feedback.final_labels,
            "notes": feedback.notes,
            "created_at": feedback.created_at,
            "feedback_count_since_retrain": feedback_count,
            "retrain_triggered": retrain_triggered,
        }

