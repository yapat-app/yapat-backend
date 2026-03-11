from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.embedding import EmbeddingVector
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALRetrainJob,
    ALModelStatus,
    ALRetrainStatus,
)
from app.schemas.pam_active_learning import ALTrainFromScratchRequest
from app.services.pam_classifier import MultiLabelMLPClassifier

logger = logging.getLogger(__name__)


class PAMActiveLearningService:
    def __init__(self, db: Session):
        self.db = db

    def get_pam_dataset(self, dataset_id: int) -> Dataset:
        ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        if ds.dataset_type != DatasetType.PAM:
            raise ValueError(
                f"Dataset {dataset_id} is of type '{ds.dataset_type.value}', expected 'PAM'"
            )
        return ds

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

    def get_checkpoint(self, checkpoint_id: int) -> Optional[ALModelCheckpoint]:
        return (
            self.db.query(ALModelCheckpoint)
            .filter(ALModelCheckpoint.id == checkpoint_id)
            .first()
        )

    def _checkout(self, ckpt: ALModelCheckpoint) -> PAMModelHandle:
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

# TODO: This function is suitable for AnuraSet and will need adaptation in future

    def _load_ground_truth_metadata(
            self,
            metadata_path: str,
            species_list: List[str],
            allowed_subsets: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load recording-level ground truth metadata.

        Returns
        -------
        gt_index : Dict[str, List[Dict[str, Any]]]
            Indexed by recording identifier (sample_name / fname / file_name / file_path),
            each value is a list of annotation events:
                {
                    "labels": np.ndarray,          # multi-hot label vector
                    "start_time": Optional[float],
                    "end_time": Optional[float],
                }

        Supported formats
        -----------------
        Format A: event-style
            Required:
              - recording identifier: file_name | recording_file | recording_name | file_path | sample_name | fname
              - species column: species | label
            Optional:
              - start_time / end_time
              - onset / offset
              - min_t / max_t

        Format B: wide multi-label
            Required:
              - recording identifier: sample_name | fname | file_name | file_path
              - one column per species in species_list
            Optional:
              - min_t / max_t
              - start_time / end_time
              - onset / offset
        """
        if not os.path.isfile(metadata_path):
            raise ValueError(f"Metadata file not found: {metadata_path}")

        species_to_idx = {species: i for i, species in enumerate(species_list)}
        gt_index: Dict[str, List[Dict[str, Any]]] = {}

        with open(metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # recording identifier column
            id_col = None
            for candidate in [
                "sample_name",
                "fname",
                "file_name",
                "recording_file",
                "recording_name",
                "file_path",
            ]:
                if candidate in fieldnames:
                    id_col = candidate
                    break

            if id_col is None:
                raise ValueError(
                    "Metadata must contain one of: "
                    "'sample_name', 'fname', 'file_name', 'recording_file', "
                    "'recording_name', or 'file_path'."
                )

            # time columns
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

            # format detection
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

            subset_col = "subset" if "subset" in fieldnames else None

            for row in reader:
                if subset_col and allowed_subsets is not None:
                    subset_value = str(row.get(subset_col, "")).strip().lower()
                    if subset_value not in allowed_subsets:
                        continue
                recording_key = str(row[id_col]).strip()
                if not recording_key:
                    continue

                start_time = None
                end_time = None
                if start_col is not None and end_col is not None:
                    raw_start = row.get(start_col)
                    raw_end = row.get(end_col)
                    if raw_start not in (None, "") and raw_end not in (None, ""):
                        start_time = float(raw_start)
                        end_time = float(raw_end)

                y = np.zeros(len(species_list), dtype=np.float32)

                # Format B: wide multi-label row
                if has_species_columns:
                    for sp in species_list:
                        value = str(row.get(sp, "0")).strip().lower()
                        y[species_to_idx[sp]] = 1.0 if value in {"1", "true", "yes"} else 0.0

                # Format A: one species per row
                else:
                    species_value = str(row[species_col]).strip()
                    if species_value in species_to_idx:
                        y[species_to_idx[species_value]] = 1.0

                # skip empty rows
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

    # Align indices of embeddings, labels and snippet ids

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

    def _save_classifier_checkpoint(
        self,
        model: MultiLabelMLPClassifier,
        checkpoint_path: str,
        hidden_dim: int,
        dropout: float,
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
        }
        torch.save(checkpoint, checkpoint_path)

# To update the species list if some species have to be eliminated during min max check
    def _save_label_config(self, label_config_path: str, species_list: List[str]) -> None:
        payload = {"species_list": species_list}
        with open(label_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # Train from scratch (COLD START)

    def train_from_scratch(self, body: ALTrainFromScratchRequest) -> ALRetrainJob:
        ds = self.get_pam_dataset(body.dataset_id)

        snippet_set_id = body.snippet_set_id or ds.default_snippet_set_id
        if snippet_set_id is None:
            raise ValueError(
                "No snippet_set_id provided and dataset has no default_snippet_set_id."
            )

        species_list = self._load_species_from_label_config(body.label_config_path)
        model_type = body.model_type.lower()

        if model_type != "pam_multilabel_classifier":
            raise ValueError(
                f"Unsupported model_type '{body.model_type}'. "
                "Only 'pam_multilabel_classifier' is currently supported."
            )

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
                "metadata_path": body.metadata_path,
                "label_config_path": body.label_config_path,
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

        job = ALRetrainJob(
            model_checkpoint_id=model_ckpt.id,
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
            job.status = ALRetrainStatus.RUNNING
            self.db.commit()

            X, snippet_rows = self._load_embeddings(
                snippet_set_id=snippet_set_id,
                embedding_model_id=body.embedding_model_id,
            )

            gt_index = self._load_ground_truth_metadata(
                metadata_path=body.metadata_path,
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

            X_train, y_train, used_species, excluded_species, class_counts = (
                model.filter_and_balance_classes(
                    X=X_train,
                    y=y_train,
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
                    settings.MODEL_ARTIFACTS_DIR,
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
                "used_species": used_species,
                "excluded_species": excluded_species,
                "class_counts": class_counts,
            }

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
            return job

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

    def manual_retrain(
        self,
        model_checkpoint_id: int,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> ALRetrainJob:
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
            self.db.query(ALRetrainJob)
            .filter(ALRetrainJob.model_checkpoint_id == model_checkpoint_id)
            .order_by(ALRetrainJob.created_at.desc())
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
            self.db.query(ALModelCheckpoint.version)
            .filter(
                ALModelCheckpoint.dataset_id == ckpt.dataset_id,
                ALModelCheckpoint.name == ckpt.name,
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

        job = ALRetrainJob(
            model_checkpoint_id=checkpoint_id,
            trigger=trigger,
            feedback_count=feedback_count,
            status=ALRetrainStatus.RUNNING,
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
            job.status = ALRetrainStatus.COMPLETED
            job.result_metrics = metrics
            job.completed_at = datetime.utcnow()

            # ── Create a new checkpoint record for the retrained version ──
            new_checkpoint_path = metrics.get("new_checkpoint_path")
            new_ckpt = ALModelCheckpoint(
                dataset_id=ckpt.dataset_id,
                name=ckpt.name,
                version=new_version,
                checkpoint_path=new_checkpoint_path,
                model_type=ckpt.model_type,
                hyperparameters=ckpt.hyperparameters,
                is_base=0,
                parent_checkpoint_id=checkpoint_id,
                status=ALModelStatus.AVAILABLE,
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
            job.status = ALRetrainStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            logger.error("PAM retrain job %d failed: %s", job.id, exc)

        self.db.commit()
        self.db.refresh(job)
        return job.status == ALRetrainStatus.COMPLETED

    # ================================================================
    # 6. Statistics
    # ================================================================

    def get_stats(self, model_checkpoint_id: int) -> Dict[str, Any]:
        """Aggregate statistics for a checkpoint."""
        total_preds = (
            self.db.query(func.count(ALPrediction.id))
            .filter(ALPrediction.model_checkpoint_id == model_checkpoint_id)
            .scalar()
        ) or 0

        total_fb = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction)
            .filter(ALPrediction.model_checkpoint_id == model_checkpoint_id)
            .scalar()
        ) or 0

        accepted = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == model_checkpoint_id,
                ALFeedbackEvent.action == ALFeedbackAction.ACCEPT,
            )
            .scalar()
        ) or 0

        rejected = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == model_checkpoint_id,
                ALFeedbackEvent.action == ALFeedbackAction.REJECT,
            )
            .scalar()
        ) or 0

        modified = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == model_checkpoint_id,
                ALFeedbackEvent.action == ALFeedbackAction.MODIFY,
            )
            .scalar()
        ) or 0

        fb_since = self._feedback_count_since_retrain(model_checkpoint_id)

        retrain_jobs = (
            self.db.query(func.count(ALRetrainJob.id))
            .filter(ALRetrainJob.model_checkpoint_id == model_checkpoint_id)
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

    # ================================================================
    # 3. Inference + scoring
    # ================================================================

    def run_inference(
        self,
        model_checkpoint_id: int,
        snippet_set_id: int,
        k: int = 20,
        device: str = "cpu",
    ) -> ALInferenceResult:
        """
        Run the full inference → scoring → ranking pipeline.

        Steps:
          1. Load embeddings for the snippet set.
          2. Check out the model and load the classifier.
          3. Run classifier inference (labels + sampling scores).
          4. Select top-k and persist predictions.

        Returns:
            dict with ``predictions``, ``total_scored``, ``model_info``.
        """
        # Load embeddings
        X_pool, snippet_ids = self._load_embeddings(snippet_set_id)
        n_dim = X_pool.shape[1]
        ckpt = self.get_checkpoint(model_checkpoint_id)
        checkpoint_path = None
        model_type = "pam_multilabel_classifier"
        if ckpt is not None:
            handle = self._checkout(ckpt)
            checkpoint_path = handle.effective_path
            model_type = handle.model_type

        # Load classifier
        classifier = load_pam_classifier(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
            n_dim=X_pool.shape[1],
            num_classes=42 #TODO: shouldn't be hardcoded
        )

        # Inference
        labels, confidences = classifier.predict(X_pool)

        # Combined scoring
        scores = combined_score(confidences)

        # Already-labeled mask (predictions with feedback)
        labeled_snippet_ids = set(
            r[0]
            for r in self.db.query(ALFeedbackEvent.prediction_id)
            .join(ALPrediction)
            .filter(ALPrediction.model_checkpoint_id == model_checkpoint_id)
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

    def _upsert_prediction(
        self,
        model_checkpoint_id: int,
        snippet_id: int,
        predicted_label: str,
        confidence: float,
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
            return existing

        pred = PAMPrediction(
            model_checkpoint_id=model_checkpoint_id,
            snippet_id=snippet_id,
            predicted_label=predicted_label,
            confidence=confidence,
        )
        self.db.add(pred)
        return pred


    def _feedback_count_since_retrain(self, checkpoint_id: int) -> int:
        """Count feedback events after the most recent completed retrain."""
        last_retrain = (
            self.db.query(ALRetrainJob.completed_at)
            .filter(
                ALRetrainJob.model_checkpoint_id == checkpoint_id,
                ALRetrainJob.status == ALRetrainStatus.COMPLETED,
            )
            .order_by(ALRetrainJob.completed_at.desc())
            .first()
        )
        cutoff = last_retrain[0] if last_retrain else datetime.min

        count = (
            self.db.query(func.count(ALFeedbackEvent.id))
            .join(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == checkpoint_id,
                ALFeedbackEvent.created_at > cutoff,
            )
            .scalar()
        )
        return count or 0


    def list_checkpoints(
        self, dataset_id: Optional[int] = None
    ) -> List[ALModelCheckpoint]:
        q = self.db.query(ALModelCheckpoint)
        if dataset_id is not None:
            q = q.filter(ALModelCheckpoint.dataset_id == dataset_id)
        return q.order_by(ALModelCheckpoint.created_at.desc()).all()