from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.embedding import EmbeddingVector
from app.models.pam_active_learning import (
    ALModelCheckpoint, ALPrediction, ALSnippetAnnotation, ALFeedbackEvent, ALRetrainJob, ALModelStatus, ALRetrainStatus, ALAnnotationSource
)
from app.schemas.pam_active_learning import (
    ALTrainFromScratchRequest,
    ALInferenceRow,
    ALFeedbackSubmit,

)
from active_learning.al_classifier import MultiLabelMLPClassifier

from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
    RETRAIN_AFTER,
)

logger = logging.getLogger(__name__)

DATA_ROOT = settings.DATA_ROOT or "/data"

class PAMActiveLearningService:
    def __init__(self, db: Session):
        self.db = db

    def register_checkpoint(
        self,
        dataset_id: int,
        name: str,
        version: str = "v0",
        checkpoint_path: Optional[str] = None,
        label_config_path: Optional[str] = None,
        model_type: str = "pam_multilabel_classifier",
        hyperparameters: Optional[Dict[str, Any]] = None,
        is_base: bool = False,
        parent_checkpoint_id: Optional[int] = None,
    ) -> ALModelCheckpoint:
        self.get_pam_dataset(dataset_id)

        existing = (
            self.db.query(ALModelCheckpoint)
            .filter(
                and_(
                    ALModelCheckpoint.dataset_id == dataset_id,
                    ALModelCheckpoint.name == name,
                    ALModelCheckpoint.version == version,
                )
            )
            .first()
        )

        if existing:
            existing.checkpoint_path = checkpoint_path or ""
            existing.label_config_path = label_config_path or ""
            existing.model_type = model_type
            existing.hyperparameters = hyperparameters
            existing.is_base = int(is_base)
            existing.parent_checkpoint_id = parent_checkpoint_id
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            logger.info("Updated PAM checkpoint id=%d", existing.id)
            return existing

        ckpt = ALModelCheckpoint(
            dataset_id=dataset_id,
            name=name,
            version=version,
            checkpoint_path=checkpoint_path or "",
            label_config_path=label_config_path or "",
            model_type=model_type,
            hyperparameters=hyperparameters,
            is_base=int(is_base),
            parent_checkpoint_id=parent_checkpoint_id,
            status=ALModelStatus.AVAILABLE,
        )
        self.db.add(ckpt)
        self.db.commit()
        self.db.refresh(ckpt)
        logger.info("Registered PAM checkpoint id=%d name=%s is_base=%s", ckpt.id, name, is_base)
        return ckpt




    def _save_classifier_checkpoint(
        self,
        model,
        checkpoint_path: str,
        hidden_dim: int,
        dropout: float,
        label_order:list[str]
    ) -> None:
        if model.model is None:
            raise ValueError("Cannot save checkpoint: classifier architecture has not been created.")

        checkpoint = {
            "model_type": "pam_multilabel_classifier",
            "n_dim": model.n_dim,
            "num_classes": model.num_classes,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "state_dict": model.state_dict(),
            "label_order": label_order,
        }
        torch.save(checkpoint, checkpoint_path)



    # Train from scratch (COLD START)

    def train_from_scratch(self, body: ALTrainFromScratchRequest) -> ALModelCheckpoint:
        logger.info("Getting dataset")
        ds = self._get_pam_dataset(body.dataset_id)

        snippet_set_id = body.snippet_set_id or ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError(
                "No snippet_set_id provided and dataset has no default_snippet_set_id."
            )
        metadata_path = os.path.join(DATA_ROOT, body.metadata_path)
        label_config_path = os.path.join(DATA_ROOT, body.label_config_path)
        species_list = self._load_species_from_label_config(label_config_path)
        model_type = body.model_type.lower()

        if model_type != "pam_multilabel_classifier":
            raise ValueError(
                f"Unsupported model_type '{body.model_type}'. "
                "Only 'pam_multilabel_classifier' is currently supported."
            )


        logger.info("Creating an entry for model checkpoint")

        model_ckpt = ALModelCheckpoint(
            dataset_id=body.dataset_id,
            name=body.checkpoint_name,
            version=body.version,
            checkpoint_path="",
            label_config_path=body.label_config_path,
            model_type=body.model_type,
            hyperparameters={
                "training_mode": "cold_start",
                "embedding_model_id": body.embedding_model_id,
                "metadata_path": metadata_path,
                "label_config_path": label_config_path,
                "min_samples_per_class": body.min_samples_per_class,
                "max_samples_per_class": body.max_samples_per_class,
                "epochs": body.epochs,
                "learning_rate": body.learning_rate,
                "batch_size": body.batch_size,
                "hidden_dim": body.hidden_dim,
                "dropout": body.dropout,
                "device": body.device,
            },
            is_base=1,
            parent_checkpoint_id=None,
            status=ALModelStatus.LOADING,
        )
        self.db.add(model_ckpt)
        self.db.flush()

        logger.info("Creating an entry for training job in ALRetrainJob")
        job = ALRetrainJob(
            model_checkpoint_id=model_ckpt.id,
            dataset_id=body.dataset_id,
            trigger="cold_start",
            feedback_count=0,
            status=ALRetrainStatus.PENDING,
            result_metrics=None,
            error_message=None,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(model_ckpt)
        self.db.refresh(job)

        try:
            logger.info("Marking retrain status as RUNNING")
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            logger.info("Loading embeddings")
            X, snippet_rows = self._load_embeddings(
                snippet_set_id=snippet_set_id,
                embedding_model_id=body.embedding_model_id,
            )
            logger.info("Loading ground truth labels")
            gt_index = self._load_ground_truth_metadata(
                metadata_path=metadata_path,
                species_list=species_list,
                allowed_subsets=["train"]
            )

            X_train, y_train, used_snippet_ids = self._align_embeddings_and_labels(
                X=X,
                snippet_rows=snippet_rows,
                gt_index=gt_index,
                species_list=species_list,
            )

            model = MultiLabelMLPClassifier()

            X_train, y_train, labeled_snippet_ids, used_species, excluded_species, class_counts = (
                model.filter_and_balance_classes(
                    X=X_train,
                    y=y_train,
                    snippet_ids=used_snippet_ids,
                    species_list=species_list,
                    min_samples_per_class=body.min_samples_per_class,
                    max_samples_per_class=body.max_samples_per_class,
                )
            )

            if y_train.shape[0] == 0:
                raise ValueError(
                    "No training samples remain after aligning embeddings and labels."
                )

            if y_train.shape[1] == 0:
                raise ValueError(
                    "No species remain after applying min_samples_per_class filtering."
                )

            n_dim = X_train.shape[1]
            num_classes = y_train.shape[1]

            model.create_classifier(
                n_dim=n_dim,
                num_classes=num_classes,
                hidden_dim=body.hidden_dim,
                dropout=body.dropout,
            )
            model.to(body.device)

            train_metrics = model.fit(
                X=X_train,
                y=y_train,
                epochs=body.epochs,
                learning_rate=body.learning_rate,
                batch_size=body.batch_size,
                device=body.device,
            )

            checkpoint_dir = self._ensure_dir(
                os.path.join(
                    settings.PAM_CHECKPOINTS_DIR,
                    "pam_active_learning",
                    str(ds.id),
                )
            )
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{body.checkpoint_name}_{body.version}_ckpt_{model_ckpt.id}.pt",
            )

            resolved_label_config_path = os.path.join(
                checkpoint_dir,
                f"{body.checkpoint_name}_{body.version}_labels_{model_ckpt.id}.json",
            )

            self._save_label_config(
                label_config_path=resolved_label_config_path,
                species_list=used_species,
            )

            self._save_classifier_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                hidden_dim=body.hidden_dim,
                dropout=body.dropout,
                label_order=used_species
            )

            model_ckpt.checkpoint_path = checkpoint_path
            model_ckpt.label_config_path = resolved_label_config_path
            model_ckpt.status = ALModelStatus.AVAILABLE
            model_ckpt.hyperparameters = {
                **(model_ckpt.hyperparameters or {}),
                "resolved_snippet_set_id": snippet_set_id,
                "n_dim": n_dim,
                "num_classes": num_classes,
                "train_samples": int(X_train.shape[0]),
                "label_order": used_species,
                "used_species": used_species,
                "excluded_species": excluded_species,
                "class_counts": class_counts,
            }
            self._store_snippet_annotations(
                dataset_id=body.dataset_id,
                snippet_ids=labeled_snippet_ids,
                y=y_train,
                label_order=used_species,
                # Hardcoding source as groundtruth as this service would only be called for cold start
                source=ALAnnotationSource.GROUND_TRUTH,
                model_checkpoint_id=model_ckpt.id,
                user_id=None,
            )

            inference_metrics = None
            if body.run_inference:
                inference_metrics = self._run_and_store_inference(
                    dataset_id=body.dataset_id,
                    model_ckpt=model_ckpt,
                    model=model,
                    X=X,
                    snippet_rows=snippet_rows,
                    label_order=used_species,
                    threshold=body.threshold,
                    density_k=body.density_k,
                    wu=body.composite_wu,
                    wd=body.composite_wd,
                    wr=body.composite_wr,
                )
            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {
                "new_checkpoint_id": model_ckpt.id,
                "new_checkpoint_path": checkpoint_path,
                "label_config_path": body.label_config_path,
                "aligned_snippet_count": len(used_snippet_ids),
                "train_samples": int(X_train.shape[0]),
                "num_classes": int(num_classes),
                "used_species": used_species,
                "excluded_species": excluded_species,
                "class_counts": class_counts,
                **train_metrics,
            }
            self.db.commit()
            self.db.refresh(model_ckpt)
            self.db.refresh(job)
            return model_ckpt

        except Exception as e:
            logger.exception("Cold-start training failed.")
            model_ckpt.status = ALModelStatus.ERROR
            job.status = ALRetrainStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self.db.commit()
            self.db.refresh(model_ckpt)
            self.db.refresh(job)
            raise

    def submit_feedback(self, body: ALFeedbackSubmit) -> dict:
        """
        Save snippet-level user feedback.

        Flow
        ----
        1. Resolve prediction row for (model_checkpoint_id, snippet_id)
        2. Store ALFeedbackEvent
        3. If ACCEPT or MODIFY, store trusted labels into ALSnippetAnnotation
        4. Count feedback since last retrain
        5. If threshold reached and no retrain is active, trigger auto retrain
        """
        model_ckpt = self._get_checkpoint(body.model_checkpoint_id)
        if model_ckpt is None:
            raise ValueError(f"Model checkpoint {body.model_checkpoint_id} not found.")

        if model_ckpt.dataset_id != body.dataset_id:
            raise ValueError(
                f"Checkpoint {body.model_checkpoint_id} does not belong to dataset {body.dataset_id}."
            )

        prediction = (
            self.db.query(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == body.model_checkpoint_id,
                ALPrediction.snippet_id == body.snippet_id,
            )
            .one_or_none()
        )
        if prediction is None:
            raise ValueError(
                f"No prediction found for checkpoint={body.model_checkpoint_id}, snippet={body.snippet_id}."
            )

        action_value = body.action.value if hasattr(body.action, "value") else body.action

        if action_value == "MODIFY" and not body.labels:
            raise ValueError("labels are required when action=MODIFY")

        feedback = ALFeedbackEvent(
            prediction_id=prediction.id,
            user_id=body.user_id,
            action=action_value,
            modified_labels=body.labels if action_value == "MODIFY" else None,
            notes=body.notes,
        )
        self.db.add(feedback)
        self.db.flush()

        labels_to_store = self._resolve_feedback_labels(
            action=action_value,
            prediction=prediction,
            labels=body.labels,
        )

        if labels_to_store:
            self._store_user_labels_for_snippet(
                dataset_id=body.dataset_id,
                snippet_id=body.snippet_id,
                labels=labels_to_store,
                model_checkpoint_id=body.model_checkpoint_id,
                user_id=body.user_id,
            )

        self.db.commit()
        self.db.refresh(feedback)

        feedback_count = self._feedback_count_since_retrain(body.model_checkpoint_id)
        retrain_triggered = False

        if (
                feedback_count >= RETRAIN_AFTER
                and not self._has_active_retrain_job(body.model_checkpoint_id)
        ):
            retrain_triggered = True
            self._auto_retrain_from_feedback(body.model_checkpoint_id)
            feedback_count = self._feedback_count_since_retrain(body.model_checkpoint_id)

        return {
            "id": feedback.id,
            "prediction_id": feedback.prediction_id,
            "action": feedback.action,
            "modified_labels": feedback.modified_labels,
            "notes": feedback.notes,
            "created_at": feedback.created_at,
            "feedback_count_since_retrain": feedback_count,
            "retrain_triggered": retrain_triggered,
        }

    def get_or_create_predictions(
            self,
            body,
    ) -> list[ALPrediction]:
        """
        Return predictions for a checkpoint and snippet set.

        Flow
        ----
        1. Check whether predictions already exist for this checkpoint on this snippet set
        2. If yes, return them
        3. Otherwise load model + embeddings, run inference, save predictions, return them
        """
        model_ckpt = self._get_checkpoint(body.model_checkpoint_id)
        if model_ckpt is None:
            raise ValueError(f"Model checkpoint {body.model_checkpoint_id} not found.")

        hyper = model_ckpt.hyperparameters or {}
        embedding_model_id = hyper.get("embedding_model_id")
        if embedding_model_id is None:
            raise ValueError(
                f"Checkpoint {model_ckpt.id} is missing embedding_model_id in hyperparameters."
            )

        threshold, density_k, wu, wd, wr = self._resolve_inference_params(
            threshold=body.threshold,
            density_k=body.density_k,
            wu=body.composite_wu,
            wd=body.composite_wd,
            wr=body.composite_wr,
        )

        existing = self._get_predictions_for_checkpoint_and_snippet_set(
            model_checkpoint_id=model_ckpt.id,
            snippet_set_id=body.snippet_set_id,
        )

        if existing and not body.force_refresh:
            return existing

        X, snippet_rows = self._load_embeddings(
            snippet_set_id=body.snippet_set_id,
            embedding_model_id=embedding_model_id,
        )

        model = MultiLabelMLPClassifier.load_from_checkpoint(
            checkpoint_path=model_ckpt.checkpoint_path,
            device=body.device,
        )

        label_order = getattr(model, "label_order", None)
        if not label_order:
            label_order = hyper.get("label_order")

        if not label_order:
            raise ValueError(
                f"No label_order found in checkpoint file or checkpoint hyperparameters for checkpoint {model_ckpt.id}."
            )

        self._run_and_store_inference(
            dataset_id=model_ckpt.dataset_id,
            model_ckpt=model_ckpt,
            model=model,
            X=X,
            snippet_rows=snippet_rows,
            label_order=label_order,
            threshold=threshold,
            density_k=density_k,
            wu=wu,
            wd=wd,
            wr=wr,
        )

        self.db.commit()

        return self._get_predictions_for_checkpoint_and_snippet_set(
            model_checkpoint_id=model_ckpt.id,
            snippet_set_id=body.snippet_set_id,
        )

    def manual_retrain(self, body: ALRetrainRequest) -> ALModelCheckpoint:
        """
        Manually trigger retraining regardless of RETRAIN_AFTER threshold.

        Flow
        ----
        1. Load parent checkpoint
        2. Sync recent feedback since last retrain into ALSnippetAnnotation
        3. Load all trusted annotations (GROUND_TRUTH + USER)
        4. Rebuild full cumulative training set
        5. Train child checkpoint
        6. Optionally run inference
        """
        parent_ckpt = self._get_checkpoint(body.model_checkpoint_id)
        if parent_ckpt is None:
            raise ValueError(f"Model checkpoint {body.model_checkpoint_id} not found.")

        hyper = parent_ckpt.hyperparameters or {}

        dataset_id = parent_ckpt.dataset_id
        snippet_set_id = hyper.get("resolved_snippet_set_id")
        embedding_model_id = hyper.get("embedding_model_id")
        label_order = hyper.get("label_order")

        if snippet_set_id is None:
            raise ValueError("Parent checkpoint missing resolved_snippet_set_id in hyperparameters.")
        if embedding_model_id is None:
            raise ValueError("Parent checkpoint missing embedding_model_id in hyperparameters.")
        if not label_order:
            raise ValueError("Parent checkpoint missing label_order in hyperparameters.")

        epochs = body.epochs if body.epochs is not None else int(hyper.get("epochs", 20))
        learning_rate = (
            body.learning_rate if body.learning_rate is not None else float(hyper.get("learning_rate", 1e-3))
        )
        batch_size = body.batch_size if body.batch_size is not None else int(hyper.get("batch_size", 32))
        hidden_dim = body.hidden_dim if body.hidden_dim is not None else int(hyper.get("hidden_dim", 128))
        dropout = body.dropout if body.dropout is not None else float(hyper.get("dropout", 0.5))
        device = body.device if body.device is not None else str(hyper.get("device", "cpu"))

        new_version = f"{parent_ckpt.version}_manual_{int(datetime.now(timezone.utc).timestamp())}"

        new_ckpt = ALModelCheckpoint(
            dataset_id=dataset_id,
            name=parent_ckpt.name,
            version=new_version,
            checkpoint_path="",
            label_config_path=parent_ckpt.label_config_path,
            model_type=parent_ckpt.model_type,
            hyperparameters={
                **hyper,
                "training_mode": "manual_retrain",
                "parent_checkpoint_id": parent_ckpt.id,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "device": device,
            },
            is_base=0,
            parent_checkpoint_id=parent_ckpt.id,
            status=ALModelStatus.LOADING,
        )
        self.db.add(new_ckpt)
        self.db.flush()

        recent_feedback_count = self._feedback_count_since_retrain(parent_ckpt.id)

        job = ALRetrainJob(
            model_checkpoint_id=new_ckpt.id,
            dataset_id=dataset_id,
            trigger="manual",
            feedback_count=recent_feedback_count,
            status=ALRetrainStatus.PENDING,
            result_metrics=None,
            error_message=None,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(new_ckpt)
        self.db.refresh(job)

        try:
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            processed_feedback_events = self._sync_feedback_events_to_annotations(parent_ckpt.id)

            annotations_by_snippet = self._get_trusted_annotations(dataset_id=dataset_id)
            if not annotations_by_snippet:
                raise ValueError("No trusted annotations available for retraining.")

            X, snippet_rows = self._load_embeddings(
                snippet_set_id=snippet_set_id,
                embedding_model_id=embedding_model_id,
            )

            snippet_ids = [row["snippet_id"] for row in snippet_rows]

            keep_indices = [i for i, sid in enumerate(snippet_ids) if sid in annotations_by_snippet]
            if not keep_indices:
                raise ValueError("No embeddings found for snippets with trusted annotations.")

            X_train = X[keep_indices]
            train_snippet_ids = [snippet_ids[i] for i in keep_indices]

            y_train = self._build_multihot_from_annotations(
                snippet_ids=train_snippet_ids,
                label_order=label_order,
                annotations_by_snippet=annotations_by_snippet,
            )

            keep_rows = y_train.sum(axis=1) > 0
            X_train = X_train[keep_rows]
            y_train = y_train[keep_rows]
            train_snippet_ids = [sid for sid, keep in zip(train_snippet_ids, keep_rows) if keep]

            if X_train.shape[0] == 0:
                raise ValueError("No training rows remain after filtering empty annotation rows.")

            model = MultiLabelMLPClassifier()
            model.create_classifier(
                n_dim=X_train.shape[1],
                num_classes=y_train.shape[1],
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            model.to(device)

            train_metrics = model.fit(
                X=X_train,
                y=y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=device,
            )

            checkpoint_dir = self._ensure_dir(
                os.path.join(
                    settings.MODEL_ARTIFACTS_DIR,
                    "pam_active_learning",
                    str(dataset_id),
                )
            )

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{new_ckpt.name}_{new_ckpt.version}_ckpt_{new_ckpt.id}.pt",
            )

            self._save_classifier_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                hidden_dim=hidden_dim,
                dropout=dropout,
                label_order=label_order,
            )

            new_ckpt.checkpoint_path = checkpoint_path
            new_ckpt.status = ALModelStatus.AVAILABLE
            new_ckpt.hyperparameters = {
                **(new_ckpt.hyperparameters or {}),
                "n_dim": int(X_train.shape[1]),
                "num_classes": int(y_train.shape[1]),
                "train_samples": int(X_train.shape[0]),
                "label_order": label_order,
                "resolved_snippet_set_id": snippet_set_id,
                "embedding_model_id": embedding_model_id,
            }

            inference_metrics = None
            if body.run_inference:
                inference_metrics = self._run_and_store_inference(
                    dataset_id=dataset_id,
                    model_ckpt=new_ckpt,
                    model=model,
                    X=X,
                    snippet_rows=snippet_rows,
                    label_order=label_order,
                    threshold=body.threshold,
                    density_k=body.density_k,
                    wu=body.composite_wu,
                    wd=body.composite_wd,
                    wr=body.composite_wr,
                )

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {
                "new_checkpoint_id": new_ckpt.id,
                "new_checkpoint_path": checkpoint_path,
                "feedback_events_since_last_retrain": recent_feedback_count,
                "processed_feedback_events": processed_feedback_events,
                "train_samples": int(X_train.shape[0]),
                "num_classes": int(y_train.shape[1]),
                "run_inference": body.run_inference,
                "inference_metrics": inference_metrics,
                **train_metrics,
            }

            self.db.commit()
            self.db.refresh(new_ckpt)
            self.db.refresh(job)
            return new_ckpt

        except Exception as e:
            logger.exception("Manual retraining failed.")
            new_ckpt.status = ALModelStatus.ERROR
            job.status = ALRetrainStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self.db.commit()
            self.db.refresh(new_ckpt)
            self.db.refresh(job)
            raise

    def _auto_retrain_from_feedback(self, parent_checkpoint_id: int) -> ALModelCheckpoint:
        parent_ckpt = self._get_checkpoint(parent_checkpoint_id)
        if parent_ckpt is None:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} not found.")

        hyper = parent_ckpt.hyperparameters or {}

        dataset_id = parent_ckpt.dataset_id
        snippet_set_id = hyper.get("resolved_snippet_set_id")
        embedding_model_id = hyper.get("embedding_model_id")
        label_order = hyper.get("label_order")

        if snippet_set_id is None:
            raise ValueError("Parent checkpoint missing resolved_snippet_set_id in hyperparameters.")
        if embedding_model_id is None:
            raise ValueError("Parent checkpoint missing embedding_model_id in hyperparameters.")
        if not label_order:
            raise ValueError("Parent checkpoint missing label_order in hyperparameters.")

        new_version = f"{parent_ckpt.version}_r{int(datetime.now(timezone.utc).timestamp())}"

        new_ckpt = ALModelCheckpoint(
            dataset_id=dataset_id,
            name=parent_ckpt.name,
            version=new_version,
            checkpoint_path="",
            label_config_path=parent_ckpt.label_config_path,
            model_type=parent_ckpt.model_type,
            hyperparameters={
                **hyper,
                "training_mode": "feedback_retrain",
                "parent_checkpoint_id": parent_checkpoint_id,
            },
            is_base=0,
            parent_checkpoint_id=parent_checkpoint_id,
            status=ALModelStatus.LOADING,
        )
        self.db.add(new_ckpt)
        self.db.flush()

        job = ALRetrainJob(
            model_checkpoint_id=new_ckpt.id,
            dataset_id=dataset_id,
            trigger="auto_feedback",
            feedback_count=self._feedback_count_since_retrain(parent_checkpoint_id),
            status=ALRetrainStatus.PENDING,
            result_metrics=None,
            error_message=None,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(new_ckpt)
        self.db.refresh(job)

        try:
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            X, snippet_rows = self._load_embeddings(
                snippet_set_id=snippet_set_id,
                embedding_model_id=embedding_model_id,
            )

            snippet_ids = [row["snippet_id"] for row in snippet_rows]

            annotations_by_snippet = self._get_trusted_annotations(dataset_id=dataset_id)

            keep_indices = [i for i, sid in enumerate(snippet_ids) if sid in annotations_by_snippet]
            if not keep_indices:
                raise ValueError("No trusted annotations available for retraining.")

            X_train = X[keep_indices]
            train_snippet_ids = [snippet_ids[i] for i in keep_indices]
            y_train = self._build_multihot_from_annotations(
                snippet_ids=train_snippet_ids,
                label_order=label_order,
                annotations_by_snippet=annotations_by_snippet,
            )

            keep_rows = y_train.sum(axis=1) > 0
            X_train = X_train[keep_rows]
            y_train = y_train[keep_rows]
            train_snippet_ids = [sid for sid, keep in zip(train_snippet_ids, keep_rows) if keep]

            if X_train.shape[0] == 0:
                raise ValueError("No training rows remain after filtering empty annotation rows.")

            model = MultiLabelMLPClassifier()
            model.create_classifier(
                n_dim=X_train.shape[1],
                num_classes=y_train.shape[1],
                hidden_dim=int(hyper.get("hidden_dim", 128)),
                dropout=float(hyper.get("dropout", 0.5)),
            )
            model.to(hyper.get("device", "cpu"))

            train_metrics = model.fit(
                X=X_train,
                y=y_train,
                epochs=int(hyper.get("epochs", 20)),
                learning_rate=float(hyper.get("learning_rate", 1e-3)),
                batch_size=int(hyper.get("batch_size", 32)),
                device=hyper.get("device", "cpu"),
            )

            checkpoint_dir = self._ensure_dir(
                os.path.join(
                    settings.MODEL_ARTIFACTS_DIR,
                    "pam_active_learning",
                    str(dataset_id),
                )
            )

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{new_ckpt.name}_{new_ckpt.version}_ckpt_{new_ckpt.id}.pt",
            )

            self._save_classifier_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                hidden_dim=int(hyper.get("hidden_dim", 128)),
                dropout=float(hyper.get("dropout", 0.5)),
                label_order=label_order,
            )

            new_ckpt.checkpoint_path = checkpoint_path
            new_ckpt.status = ALModelStatus.AVAILABLE
            new_ckpt.hyperparameters = {
                **(new_ckpt.hyperparameters or {}),
                "n_dim": int(X_train.shape[1]),
                "num_classes": int(y_train.shape[1]),
                "train_samples": int(X_train.shape[0]),
            }

            inference_metrics = self._run_and_store_inference(
                dataset_id=dataset_id,
                model_ckpt=new_ckpt,
                model=model,
                X=X,
                snippet_rows=snippet_rows,
                label_order=label_order,
                threshold=hyper.get("threshold"),
                density_k=hyper.get("density_k"),
                wu=hyper.get("composite_wu"),
                wd=hyper.get("composite_wd"),
                wr=hyper.get("composite_wr"),
            )

            job.status = ALRetrainStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result_metrics = {
                "new_checkpoint_id": new_ckpt.id,
                "new_checkpoint_path": checkpoint_path,
                "train_samples": int(X_train.shape[0]),
                "num_classes": int(y_train.shape[1]),
                "inference_metrics": inference_metrics,
                **train_metrics,
            }

            self.db.commit()
            self.db.refresh(new_ckpt)
            self.db.refresh(job)
            return new_ckpt

        except Exception as e:
            logger.exception("Auto retraining from feedback failed.")
            new_ckpt.status = ALModelStatus.ERROR
            job.status = ALRetrainStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self.db.commit()
            self.db.refresh(new_ckpt)
            self.db.refresh(job)
            raise

    def build_inference_rows(
            self,
            probs: torch.Tensor,
            preds: torch.Tensor,
            embeddings: torch.Tensor,
            snippet_ids: Sequence[int],
            labeled_snippet_ids: set[int],
            label_order: List[str],
            density_k: int,
            wu: float,
            wd: float,
            wr: float,
    ) -> list[ALInferenceRow]:
        """
        Compute prediction rows for all snippets and attach acquisition scores
        for unlabeled snippets.
        """
        uncertainty_scores = uncertainty(probs)

        unlabeled_indices = [i for i, sid in enumerate(snippet_ids) if sid not in labeled_snippet_ids]
        labeled_indices = [i for i, sid in enumerate(snippet_ids) if sid in labeled_snippet_ids]

        z_u = embeddings[unlabeled_indices] if unlabeled_indices else torch.empty((0, embeddings.shape[1]),
                                                                                  device=embeddings.device)
        z_l = embeddings[labeled_indices] if labeled_indices else torch.empty((0, embeddings.shape[1]),
                                                                              device=embeddings.device)

        diversity_scores_u = diversity(z_u, z_l)
        density_scores_u = density(z_u, k=density_k)
        composite_scores_u = composite(
            uncertainty_scores=uncertainty_scores[unlabeled_indices] if unlabeled_indices else torch.empty(0,
                                                                                                           device=embeddings.device),
            diversity_scores=diversity_scores_u,
            density_scores=density_scores_u,
            wu=wu,
            wd=wd,
            wr=wr,
        )

        diversity_full = [None] * len(snippet_ids)
        density_full = [None] * len(snippet_ids)
        composite_full = [None] * len(snippet_ids)

        for pos, idx in enumerate(unlabeled_indices):
            diversity_full[idx] = float(diversity_scores_u[pos].item())
            density_full[idx] = float(density_scores_u[pos].item())
            composite_full[idx] = float(composite_scores_u[pos].item())

        rows: list[ALInferenceRow] = []

        for i, snippet_id in enumerate(snippet_ids):
            pred_indices = torch.where(preds[i] > 0)[0].tolist()
            pred_labels = [label_order[j] for j in pred_indices]
            prob_dict = {
                label_order[j]: float(probs[i, j].item())
                for j in range(len(label_order))
            }

            rows.append(
                ALInferenceRow(
                    snippet_id=snippet_id,
                    predicted_labels=pred_labels,
                    predicted_probabilities=prob_dict,
                    uncertainty=float(uncertainty_scores[i].item()),
                    diversity=diversity_full[i],
                    density=density_full[i],
                    composite_score=composite_full[i],
                )
            )

        return rows





    def list_checkpoints(
        self, dataset_id: Optional[int] = None
    ) -> List[ALModelCheckpoint]:
        q = self.db.query(ALModelCheckpoint)
        if dataset_id is not None:
            q = q.filter(ALModelCheckpoint.dataset_id == dataset_id)
        return q.order_by(ALModelCheckpoint.created_at.desc()).all()


    def _get_pam_dataset(self, dataset_id: int) -> Dataset:
        ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        if ds.dataset_type != DatasetType.PAM:
            raise ValueError(
                f"Dataset {dataset_id} is of type '{ds.dataset_type.value}', expected 'PAM'"
            )
        return ds

    def _feedback_count_since_retrain(self, checkpoint_id: int) -> int:
        """
        Count feedback events created after the most recent completed retrain
        for the given checkpoint.
        """
        last_retrain = (
            self.db.query(ALRetrainJob.completed_at)
            .filter(
                ALRetrainJob.model_checkpoint_id == checkpoint_id,
                ALRetrainJob.status == ALRetrainStatus.COMPLETED,
            )
            .order_by(ALRetrainJob.completed_at.desc())
            .first()
        )

        cutoff = last_retrain[0] if last_retrain and last_retrain[0] is not None else datetime.min.replace(
            tzinfo=timezone.utc)

        count = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction, ALPrediction.id == ALFeedbackEvent.prediction_id)
            .filter(
                ALPrediction.model_checkpoint_id == checkpoint_id,
                ALFeedbackEvent.created_at > cutoff,
            )
            .scalar()
        )
        return int(count or 0)

    def _get_last_completed_retrain_cutoff(self, checkpoint_id: int) -> datetime:
        last_retrain = (
            self.db.query(ALRetrainJob.completed_at)
            .filter(
                ALRetrainJob.model_checkpoint_id == checkpoint_id,
                ALRetrainJob.status == ALRetrainStatus.COMPLETED,
            )
            .order_by(ALRetrainJob.completed_at.desc())
            .first()
        )
        if last_retrain and last_retrain[0] is not None:
            return last_retrain[0]
        return datetime.min.replace(tzinfo=timezone.utc)

    def _sync_feedback_events_to_annotations(
            self,
            checkpoint_id: int,
    ) -> int:
        """
        Sync recent ACCEPT / MODIFY feedback since the last retrain into
        ALSnippetAnnotation.

        Returns the number of feedback events processed.
        """
        events = self._get_feedback_events_since_last_retrain(checkpoint_id)
        if not events:
            return 0

        for event in events:
            prediction = event.prediction
            if prediction is None:
                continue

            model_ckpt = prediction.model_checkpoint
            if model_ckpt is None:
                continue

            dataset_id = model_ckpt.dataset_id
            snippet_id = prediction.snippet_id

            labels_to_store: list[str] = []

            if event.action == ALFeedbackAction.ACCEPT:
                labels_to_store = prediction.predicted_labels or []

            elif event.action == ALFeedbackAction.MODIFY:
                labels_to_store = event.modified_labels or []

            elif event.action == ALFeedbackAction.REJECT:
                labels_to_store = []

            for label in labels_to_store:
                exists = (
                    self.db.query(ALSnippetAnnotation)
                    .filter(
                        ALSnippetAnnotation.dataset_id == dataset_id,
                        ALSnippetAnnotation.snippet_id == snippet_id,
                        ALSnippetAnnotation.label == label,
                        ALSnippetAnnotation.source == ALAnnotationSource.USER,
                        ALSnippetAnnotation.user_id == event.user_id,
                        ALSnippetAnnotation.model_checkpoint_id == checkpoint_id,
                    )
                    .one_or_none()
                )

                if exists is None:
                    self.db.add(
                        ALSnippetAnnotation(
                            dataset_id=dataset_id,
                            snippet_id=snippet_id,
                            label=label,
                            source=ALAnnotationSource.USER,
                            user_id=event.user_id,
                            model_checkpoint_id=checkpoint_id,
                        )
                    )

        self.db.flush()
        return len(events)

    def _get_feedback_events_since_last_retrain(
            self,
            checkpoint_id: int,
    ) -> list[ALFeedbackEvent]:
        cutoff = self._get_last_completed_retrain_cutoff(checkpoint_id)

        return (
            self.db.query(ALFeedbackEvent)
            .join(ALPrediction, ALPrediction.id == ALFeedbackEvent.prediction_id)
            .filter(
                ALPrediction.model_checkpoint_id == checkpoint_id,
                ALFeedbackEvent.created_at > cutoff,
            )
            .order_by(ALFeedbackEvent.created_at.asc())
            .all()
        )

    def _get_trusted_annotations(
            self,
            dataset_id: int,
    ) -> dict[int, set[str]]:
        rows = (
            self.db.query(ALSnippetAnnotation.snippet_id, ALSnippetAnnotation.label)
            .filter(
                ALSnippetAnnotation.dataset_id == dataset_id,
                ALSnippetAnnotation.source.in_([
                    ALAnnotationSource.GROUND_TRUTH,
                    ALAnnotationSource.USER,
                ]),
            )
            .all()
        )

        out: dict[int, set[str]] = {}
        for snippet_id, label in rows:
            out.setdefault(snippet_id, set()).add(label)
        return out

    def _align_embeddings_and_labels(
            self,
            X: np.ndarray,
            snippet_rows: List[Dict[str, Any]],
            gt_index: Dict[str, List[Dict[str, Any]]],
            species_list: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Align snippet embeddings with recording-level or segment-level ground truth.

        Matching strategy
        -----------------
        1. Match snippet recording by file_name
        2. If not found, try file_path
        3. For matching metadata rows:
           - if row has no time interval: applies to whole recording
           - else include labels if metadata interval overlaps snippet interval
        """
        keep_indices: List[int] = []
        y_rows: List[np.ndarray] = []
        used_snippet_ids: List[int] = []

        for i, snippet in enumerate(snippet_rows):
            snippet_start = float(snippet["start_time"])
            snippet_end = float(snippet["end_time"])

            events = gt_index.get(snippet["file_name"])
            if events is None:
                events = gt_index.get(snippet["file_path"], [])

            y = np.zeros(len(species_list), dtype=np.float32)

            for event in events:
                event_labels = event["labels"]
                event_start = event["start_time"]
                event_end = event["end_time"]

                # recording-level label
                if event_start is None or event_end is None:
                    y = np.maximum(y, event_labels)
                    continue

                # interval overlap
                overlaps = (event_start < snippet_end) and (event_end > snippet_start)
                if overlaps:
                    y = np.maximum(y, event_labels)

            if y.sum() > 0:
                keep_indices.append(i)
                y_rows.append(y)
                used_snippet_ids.append(snippet["snippet_id"])

        if not keep_indices:
            raise ValueError(
                "No overlap found between snippet embeddings and ground-truth metadata."
            )

        X_aligned = X[keep_indices]
        y_aligned = np.stack(y_rows, axis=0).astype(np.float32)
        return X_aligned, y_aligned, used_snippet_ids

    # TODO: This function is suitable for AnuraSet and will need adaptation in future

    def _load_ground_truth_metadata(
            self,
            metadata_path: str,
            species_list: List[str],
            allowed_subsets: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.isfile(metadata_path):
            raise ValueError(f"Metadata file not found: {metadata_path}")

        species_to_idx = {species: i for i, species in enumerate(species_list)}
        gt_index: Dict[str, List[Dict[str, Any]]] = {}

        with open(metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            subset_col = "subset" if "subset" in fieldnames else None

            has_fname_clip_key = all(col in fieldnames for col in ["fname", "min_t", "max_t"])

            id_col = None
            if not has_fname_clip_key:
                for candidate in [
                    "file_name",
                    "recording_file",
                    "recording_name",
                    "file_path",
                    "fname",
                    "sample_name",
                ]:
                    if candidate in fieldnames:
                        id_col = candidate
                        break

            if not has_fname_clip_key and id_col is None:
                raise ValueError(
                    "Metadata must contain either:\n"
                    "- fname + min_t + max_t for snippet-level matching, or\n"
                    "- one of sample_name, fname, file_name, recording_file, recording_name, file_path."
                )

            start_col = None
            end_col = None
            if "min_t" in fieldnames and "max_t" in fieldnames:
                start_col = "min_t"
                end_col = "max_t"
            elif "start_time" in fieldnames and "end_time" in fieldnames:
                start_col = "start_time"
                end_col = "end_time"
            elif "onset" in fieldnames and "offset" in fieldnames:
                start_col = "onset"
                end_col = "offset"

            has_species_columns = all(sp in fieldnames for sp in species_list)
            species_col = None
            for candidate in ["species", "label"]:
                if candidate in fieldnames:
                    species_col = candidate
                    break

            if not has_species_columns and species_col is None:
                raise ValueError(
                    "Metadata must contain either:\n"
                    "- one binary column per species in species_list, or\n"
                    "- a 'species' / 'label' column."
                )

            for row in reader:
                if subset_col and allowed_subsets is not None:
                    subset_value = str(row.get(subset_col, "")).strip().lower()
                    if subset_value not in allowed_subsets:
                        continue

                start_time = None
                end_time = None
                if start_col is not None and end_col is not None:
                    raw_start = row.get(start_col)
                    raw_end = row.get(end_col)
                    if raw_start not in (None, "") and raw_end not in (None, ""):
                        start_time = float(raw_start)
                        end_time = float(raw_end)

                if has_fname_clip_key:
                    fname = str(row["fname"]).strip()
                    if not fname:
                        continue
                    if start_time is None or end_time is None:
                        raise ValueError("fname-based snippet metadata requires min_t/max_t or equivalent times.")

                    def _fmt_time(t: float) -> str:
                        return str(int(t)) if float(t).is_integer() else str(t)

                    recording_key = f"{fname}_{_fmt_time(start_time)}_{_fmt_time(end_time)}.wav"
                else:
                    recording_key = str(row[id_col]).strip()
                    if not recording_key:
                        continue

                y = np.zeros(len(species_list), dtype=np.float32)

                if has_species_columns:
                    for sp in species_list:
                        value = str(row.get(sp, "0")).strip().lower()
                        y[species_to_idx[sp]] = 1.0 if value in {"1", "true", "yes"} else 0.0
                else:
                    species_value = str(row[species_col]).strip()
                    if species_value in species_to_idx:
                        y[species_to_idx[species_value]] = 1.0

                if y.sum() == 0:
                    continue

                gt_index.setdefault(recording_key, []).append(
                    {
                        "labels": y,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )

        if not gt_index:
            raise ValueError(f"No usable ground-truth rows found in metadata file: {metadata_path}")

        return gt_index

    def _get_checkpoint(self, checkpoint_id: int) -> Optional[ALModelCheckpoint]:
        return (
            self.db.query(ALModelCheckpoint)
            .filter(ALModelCheckpoint.id == checkpoint_id)
            .first()
        )


    def _ensure_dir(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _load_embeddings(
            self,
            snippet_set_id: int,
            embedding_model_id: int,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Load embeddings for a snippet set using embedding_model_id.

        Returns
        -------
        X : np.ndarray
            Embedding matrix [N, D]
        snippet_rows : List[Dict[str, Any]]
            Per-snippet metadata needed for alignment with ground truth
        """
        rows = (
            self.db.query(
                Snippet.id,
                Snippet.recording_id,
                Snippet.start_time,
                Snippet.end_time,
                Recording.file_name,
                Recording.file_path,
                EmbeddingVector.vector,
                EmbeddingVector.dim,
            )
            .join(Recording, Snippet.recording_id == Recording.id)
            .join(EmbeddingVector, Snippet.id == EmbeddingVector.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
            .filter(EmbeddingVector.embedding_model_id == embedding_model_id)
            .order_by(Snippet.id)
            .all()
        )

        if not rows:
            raise ValueError(
                f"No embeddings found for snippet_set_id={snippet_set_id}, "
                f"embedding_model_id={embedding_model_id}"
            )

        dims = {row[7] for row in rows}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent embedding dimensions found: {dims}")

        X = np.asarray([row[6] for row in rows], dtype=np.float32)

        snippet_rows = [
            {
                "snippet_id": row[0],
                "recording_id": row[1],
                "start_time": float(row[2]),
                "end_time": float(row[3]),
                "file_name": row[4],
                "file_path": row[5],
            }
            for row in rows
        ]

        return X, snippet_rows

    # To update the species list if some species have to be eliminated during min max check
    def _save_label_config(self, label_config_path: str, species_list: List[str]) -> None:
        payload = {"species_list": species_list}
        with open(label_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_species_from_label_config(self, label_config_path: str) -> List[str]:

        if not label_config_path:
            raise ValueError("label_config_path is required.")
        if not os.path.isfile(label_config_path):
            raise ValueError(f"Label config file not found: {label_config_path}")

        with open(label_config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        species_list = payload.get("species_list")
        if not isinstance(species_list, list) or len(species_list) == 0:
            raise ValueError("Label config must contain a non-empty 'species_list' field.")

        return [str(s) for s in species_list]

    def _store_snippet_annotations(
            self,
            dataset_id: int,
            snippet_ids: list[int],
            y: np.ndarray,
            label_order: list[str],
            source: ALAnnotationSource,
            model_checkpoint_id: int | None = None,
            user_id: int | None = None,
    ) -> None:
        """
        Store one annotation row per positive label per snippet.
        """
        if len(snippet_ids) != y.shape[0]:
            raise ValueError(
                f"Mismatch: {len(snippet_ids)=} but y has {y.shape[0]} rows."
            )

        if len(label_order) != y.shape[1]:
            raise ValueError(
                f"Mismatch: {len(label_order)=} but y has {y.shape[1]} columns."
            )

        for row_idx, snippet_id in enumerate(snippet_ids):
            positive_indices = np.where(y[row_idx] > 0)[0]

            for class_idx in positive_indices:
                label = label_order[class_idx]

                exists = (
                    self.db.query(ALSnippetAnnotation)
                    .filter(
                        ALSnippetAnnotation.snippet_id == snippet_id,
                        ALSnippetAnnotation.label == label,
                        ALSnippetAnnotation.source == source,
                        ALSnippetAnnotation.user_id == user_id,
                        ALSnippetAnnotation.model_checkpoint_id == model_checkpoint_id,
                    )
                    .first()
                )

                if exists is None:
                    self.db.add(
                        ALSnippetAnnotation(
                            dataset_id=dataset_id,
                            snippet_id=snippet_id,
                            label=label,
                            source=source,
                            user_id=user_id,
                            model_checkpoint_id=model_checkpoint_id,
                        )
                    )

    def _resolve_inference_params(
            self,
            threshold: float | None,
            density_k: int | None,
            wu: float | None,
            wd: float | None,
            wr: float | None,
    ) -> tuple[float, int, float, float, float]:
        return (
            threshold if threshold is not None else DEFAULT_INFERENCE_THRESHOLD,
            density_k if density_k is not None else DEFAULT_DENSITY_K,
            wu if wu is not None else DEFAULT_COMPOSITE_WU,
            wd if wd is not None else DEFAULT_COMPOSITE_WD,
            wr if wr is not None else DEFAULT_COMPOSITE_WR,
        )

    def _get_labeled_snippet_ids_for_dataset(self, dataset_id: int) -> set[int]:
        rows = (
            self.db.query(ALSnippetAnnotation.snippet_id)
            .filter(ALSnippetAnnotation.dataset_id == dataset_id)
            .distinct()
            .all()
        )
        return {row[0] for row in rows}

    def _save_prediction_rows(
            self,
            model_checkpoint_id: int,
            rows,
    ) -> None:
        for row in rows:
            existing = (
                self.db.query(ALPrediction)
                .filter(
                    ALPrediction.model_checkpoint_id == model_checkpoint_id,
                    ALPrediction.snippet_id == row.snippet_id,
                )
                .one_or_none()
            )

            if existing is None:
                existing = ALPrediction(
                    model_checkpoint_id=model_checkpoint_id,
                    snippet_id=row.snippet_id,
                )
                self.db.add(existing)

            existing.predicted_labels = row.predicted_labels
            existing.predicted_probabilities = row.predicted_probabilities
            existing.uncertainty = row.uncertainty
            existing.diversity = row.diversity
            existing.density = row.density
            existing.composite_score = row.composite_score

    def _run_and_store_inference(
            self,
            dataset_id: int,
            model_ckpt,
            model,
            X,
            snippet_rows,
            label_order: list[str],
            threshold: float | None = None,
            density_k: int | None = None,
            wu: float | None = None,
            wd: float | None = None,
            wr: float | None = None,
    ) -> dict:
        threshold, density_k, wu, wd, wr = self._resolve_inference_params(
            threshold=threshold,
            density_k=density_k,
            wu=wu,
            wd=wd,
            wr=wr,
        )

        device = next(model.parameters()).device
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        probs, preds = model.predict(x_tensor, threshold=threshold)
        snippet_ids = [row["snippet_id"] for row in snippet_rows]
        #snippet_ids = [row.snippet_id for row in snippet_rows]
        labeled_snippet_ids = self._get_labeled_snippet_ids_for_dataset(dataset_id)

        rows = self.build_inference_rows(
            probs=probs,
            preds=preds,
            embeddings=x_tensor,
            snippet_ids=snippet_ids,
            labeled_snippet_ids=labeled_snippet_ids,
            label_order=label_order,
            density_k=density_k,
            wu=wu,
            wd=wd,
            wr=wr,
        )

        self._save_prediction_rows(
            model_checkpoint_id=model_ckpt.id,
            rows=rows,
        )

        return {
            "num_predictions": len(rows),
            "num_labeled_snippets": len(labeled_snippet_ids),
            "threshold": threshold,
            "density_k": density_k,
            "composite_wu": wu,
            "composite_wd": wd,
            "composite_wr": wr,
        }

    def _get_predictions_for_checkpoint_and_snippet_set(
            self,
            model_checkpoint_id: int,
            snippet_set_id: int,
    ) -> list[ALPrediction]:
        return (
            self.db.query(ALPrediction)
            .join(Snippet, Snippet.id == ALPrediction.snippet_id)
            .filter(
                ALPrediction.model_checkpoint_id == model_checkpoint_id,
                Snippet.snippet_set_id == snippet_set_id,
            )
            .order_by(ALPrediction.composite_score.desc().nullslast(), ALPrediction.id.asc())
            .all()
        )

    def _store_user_labels_for_snippet(
            self,
            dataset_id: int,
            snippet_id: int,
            labels: list[str],
            model_checkpoint_id: int,
            user_id: int | None = None,
    ) -> None:
        for label in labels:
            exists = (
                self.db.query(ALSnippetAnnotation)
                .filter(
                    ALSnippetAnnotation.dataset_id == dataset_id,
                    ALSnippetAnnotation.snippet_id == snippet_id,
                    ALSnippetAnnotation.label == label,
                    ALSnippetAnnotation.source == ALAnnotationSource.USER,
                    ALSnippetAnnotation.user_id == user_id,
                    ALSnippetAnnotation.model_checkpoint_id == model_checkpoint_id,
                )
                .one_or_none()
            )

            if exists is None:
                self.db.add(
                    ALSnippetAnnotation(
                        dataset_id=dataset_id,
                        snippet_id=snippet_id,
                        label=label,
                        source=ALAnnotationSource.USER,
                        user_id=user_id,
                        model_checkpoint_id=model_checkpoint_id,
                    )
                )



    def _build_multihot_from_annotations(
            self,
            snippet_ids: list[int],
            label_order: list[str],
            annotations_by_snippet: dict[int, set[str]],
    ) -> np.ndarray:
        label_to_idx = {label: i for i, label in enumerate(label_order)}
        y = np.zeros((len(snippet_ids), len(label_order)), dtype=np.float32)

        for row_idx, snippet_id in enumerate(snippet_ids):
            for label in annotations_by_snippet.get(snippet_id, set()):
                if label in label_to_idx:
                    y[row_idx, label_to_idx[label]] = 1.0

        return y

    def _resolve_feedback_labels(
            self,
            action: str,
            prediction: ALPrediction,
            labels: list[str] | None,
    ) -> list[str]:
        if action == "ACCEPT":
            return labels if labels else (prediction.predicted_labels or [])
        if action == "MODIFY":
            return labels or []
        return []

    def _has_active_retrain_job(self, checkpoint_id: int) -> bool:
        return (
                self.db.query(ALRetrainJob)
                .filter(
                    ALRetrainJob.model_checkpoint_id == checkpoint_id,
                    ALRetrainJob.status.in_([ALRetrainStatus.PENDING, ALRetrainStatus.RUNNING]),
                )
                .first()
                is not None
        )

    # def _checkout(self, ckpt: ALModelCheckpoint) -> PAMModelHandle:
    #     return checkout_model(
    #         checkpoint_id=ckpt.id,
    #         dataset_id=ckpt.dataset_id,
    #         name=ckpt.name,
    #         version=ckpt.version,
    #         checkpoint_path=ckpt.checkpoint_path,
    #         model_type=ckpt.model_type,
    #         hyperparameters=ckpt.hyperparameters or {},
    #         is_base=bool(ckpt.is_base),
    #         parent_checkpoint_id=ckpt.parent_checkpoint_id,
    #         base_model_path_setting=settings.PAM_BASE_MODEL_PATH,
    #     )