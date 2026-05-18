"""
WSSED Service

Handles communication with the WSSED GPU server and manages training job
state in the local database.
"""

from __future__ import annotations

import logging
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, SnippetSet
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALModelFamilyState,
    ALModelStatus,
    ALRetrainJob,
    ALRetrainStatus,
)
from app.models.snippet import Snippet
from app.models.wssed import (
    FeedbackType,
    TrainingStatus,
    WSSEDSnippetLabel,
    WSSEDSpeciesModel,
    WSSEDTrainingJob,
)

logger = logging.getLogger(__name__)


class WSSEDService:
    def __init__(self, db: Session):
        self.db = db
        self.gpu_url = settings.WSSED_GPU_SERVER_URL.rstrip("/")
        self.timeout = settings.WSSED_TIMEOUT

    # ------------------------------------------------------------------ #
    # GPU server communication                                             #
    # ------------------------------------------------------------------ #

    async def trigger_remote_training(self, job_id: int) -> str:
        """
        Send a training request to the WSSED GPU server and update job
        status to TRAINING.  Returns the job_id as a string task id.
        """
        job = self._get_training_job(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if dataset is None:
            raise ValueError(f"Dataset {job.dataset_id} not found")

        payload = {
            "job_id": job_id,
            "dataset_id": job.dataset_id,
            "dataset_path": dataset.source_uri,
            "hyperparameters": job.hyperparameters or {},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.gpu_url}/wssed/train", json=payload)
            response.raise_for_status()

        job.status = TrainingStatus.TRAINING
        self.db.commit()

        return str(job_id)

    async def trigger_detection(self, job_id: int, threshold: float = 0.5) -> str:
        """
        Send a detection request to the WSSED GPU server.
        """
        job = self._get_training_job(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")
        if not job.model_path:
            raise ValueError(f"Training job {job_id} has no model path yet")

        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if dataset is None:
            raise ValueError(f"Dataset {job.dataset_id} not found")

        payload = {
            "job_id": job_id,
            "model_path": job.model_path,
            "dataset_path": dataset.source_uri,
            "threshold": threshold,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.gpu_url}/wssed/detect", json=payload)
            response.raise_for_status()

        return str(job_id)

    async def update_training_status(self, job_id: int) -> WSSEDTrainingJob:
        """
        Poll the GPU server for the latest training status and persist it.
        """
        job = self._get_training_job(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.gpu_url}/wssed/train/{job_id}/status"
            )
            response.raise_for_status()
            data = response.json()

        gpu_status = data.get("status", "PENDING")
        status_map = {
            "PENDING": TrainingStatus.PENDING,
            "TRAINING": TrainingStatus.TRAINING,
            "COMPLETED": TrainingStatus.COMPLETED,
            "FAILED": TrainingStatus.FAILED,
        }
        job.status = status_map.get(gpu_status, TrainingStatus.TRAINING)

        if data.get("model_path"):
            job.model_path = data["model_path"]
        if data.get("model_paths"):
            job.model_paths = data["model_paths"]
        if data.get("metrics"):
            job.training_metrics = data["metrics"]
        if data.get("progress") is not None:
            job.progress = data["progress"]
        if data.get("error"):
            job.error_message = data["error"]

        now = datetime.now(timezone.utc)
        job.updated_at = now
        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
            job.completed_at = now

        self.db.commit()
        self.db.refresh(job)

        if job.status == TrainingStatus.COMPLETED:
            try:
                checkpoint = self._ensure_training_job_registered_for_al(job)
                if checkpoint is not None:
                    self._ensure_training_job_inference_enqueued(job, checkpoint)
            except Exception as e:
                logger.exception(
                    "Failed to register WSSED job %s for active learning: %s",
                    job.id,
                    e,
                )
                metrics = dict(job.training_metrics or {})
                metrics["al_registration_error"] = str(e)
                job.training_metrics = metrics
                self.db.commit()

        return job

    # ------------------------------------------------------------------ #
    # Prediction storage                                                   #
    # ------------------------------------------------------------------ #

    def store_predictions(self, job_id: int, predictions: list) -> int:
        """Persist predictions received from the GPU server."""
        from app.models.wssed import WSSEDPrediction

        stored = 0
        for pred in predictions:
            p = WSSEDPrediction(
                training_job_id=job_id,
                recording_id=pred["recording_id"],
                species_name=pred["species_name"],
                start_time=pred["start_time"],
                end_time=pred["end_time"],
                confidence=pred["confidence"],
                frame_probabilities=pred.get("frame_probabilities"),
            )
            self.db.add(p)
            stored += 1

        self.db.commit()
        return stored

    # ------------------------------------------------------------------ #
    # Active learning – suggestions                                        #
    # ------------------------------------------------------------------ #

    def get_suggestions(
        self,
        dataset_id: int,
        snippet_set_id: int,
        species_name: str,
        threshold: float = 0.0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Return unlabeled snippet suggestions for the given species model,
        filtered to the requested snippet set and confidence threshold.
        """
        model = self._get_or_none_species_model(dataset_id, species_name)
        if model is None:
            return {"model_info": {"species_model_id": -1}, "suggestions": []}

        rows = (
            self.db.query(WSSEDSnippetLabel)
            .join(Snippet, Snippet.id == WSSEDSnippetLabel.snippet_id)
            .filter(
                WSSEDSnippetLabel.species_model_id == model.id,
                Snippet.snippet_set_id == snippet_set_id,
                WSSEDSnippetLabel.predicted_label >= threshold,
                WSSEDSnippetLabel.user_label == None,  # noqa: E711 – unlabeled only
            )
            .order_by(WSSEDSnippetLabel.predicted_label.desc())
            .limit(limit)
            .all()
        )

        suggestions = [
            {"snippet_id": r.snippet_id, "confidence": r.predicted_label}
            for r in rows
        ]

        return {
            "model_info": {"species_model_id": model.id},
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------ #
    # Active learning – labeling                                           #
    # ------------------------------------------------------------------ #

    def submit_label(
        self,
        snippet_set_id: int,
        dataset_id: int,
        species_name: str,
        snippet_id: int,
        label: int,
    ) -> None:
        """Accept (1) or reject (0) a snippet for the given species."""
        model = self._get_or_none_species_model(dataset_id, species_name)
        if model is None:
            raise ValueError(
                f"No species model found for dataset={dataset_id}, species={species_name}"
            )

        row = (
            self.db.query(WSSEDSnippetLabel)
            .filter(
                WSSEDSnippetLabel.species_model_id == model.id,
                WSSEDSnippetLabel.snippet_id == snippet_id,
            )
            .first()
        )
        if row is None:
            raise ValueError(
                f"Snippet label record not found for snippet={snippet_id}"
            )

        row.user_label = FeedbackType.ACCEPTED if label == 1 else FeedbackType.REJECTED
        row.labeled_at = datetime.now(timezone.utc)
        self.db.commit()

    # ------------------------------------------------------------------ #
    # Active learning – retrain                                            #
    # ------------------------------------------------------------------ #

    def retrain(
        self,
        snippet_set_id: int,
        dataset_id: int,
        species_name: str,
        device: str = "cpu",
        epochs: int = 10,
        lr: float = 0.001,
    ) -> WSSEDTrainingJob:
        """
        Create a training job for species-level re-training and dispatch it
        to the Celery worker (which will call the GPU server).
        """
        hyperparameters = {
            "species_name": species_name,
            "snippet_set_id": snippet_set_id,
            "device": device,
            "epochs": epochs,
            "learning_rate": lr,
        }

        job = self.create_training_job(
            dataset_id=dataset_id,
            model_name="CDur",
            hyperparameters=hyperparameters,
        )

        try:
            self.enqueue_wssed_training_dispatch(job.id)
        except Exception as e:
            self.fail_training_job(
                job.id,
                f"Failed to queue training task: {e}",
            )
            raise

        return job

    # ------------------------------------------------------------------ #
    # Histogram                                                            #
    # ------------------------------------------------------------------ #

    def get_histogram(
        self, model_id: int, snippet_set_id: int, bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute a histogram of predicted_label values for the given model
        restricted to snippets in the given snippet set.
        """
        rows = (
            self.db.query(WSSEDSnippetLabel.predicted_label)
            .join(Snippet, Snippet.id == WSSEDSnippetLabel.snippet_id)
            .filter(
                WSSEDSnippetLabel.species_model_id == model_id,
                Snippet.snippet_set_id == snippet_set_id,
            )
            .all()
        )

        values = [r[0] for r in rows if r[0] is not None]
        if not values:
            return {"bin_edges": [], "counts": []}

        counts, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
        return {
            "bin_edges": [round(float(e), 4) for e in bin_edges],
            "counts": [int(c) for c in counts],
        }

    # ------------------------------------------------------------------ #
    # Training job management                                              #
    # ------------------------------------------------------------------ #

    def create_training_job(
        self,
        dataset_id: int,
        model_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> WSSEDTrainingJob:
        """Create a new WSSED training job record."""
        job = WSSEDTrainingJob(
            dataset_id=dataset_id,
            model_name=model_name,
            hyperparameters=hyperparameters or {},
            status=TrainingStatus.PENDING,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def enqueue_wssed_training_dispatch(self, job_id: int) -> str:
        """
        Publish ``trigger_wssed_training`` to the broker on the ``default`` queue.

        Returns the Celery async result task id. Raises if the message cannot be
        published (broker down, wrong routing, etc.).
        """
        if self._get_training_job(job_id) is None:
            raise ValueError(f"Training job {job_id} not found")

        from app.tasks.wssed_tasks import trigger_wssed_training

        async_result = trigger_wssed_training.apply_async(
            args=[job_id],
            queue="default",
        )
        logger.info(
            "Queued trigger_wssed_training job_id=%s celery_task_id=%s queue=default",
            job_id,
            async_result.id,
        )
        return str(async_result.id)

    def fail_training_job(self, job_id: int, message: str) -> None:
        """Mark a job FAILED (e.g. Celery enqueue failed)."""
        job = self._get_training_job(job_id)
        if job is None:
            return
        job.status = TrainingStatus.FAILED
        job.error_message = message
        self.db.commit()

    def _get_training_job(self, job_id: int) -> Optional[WSSEDTrainingJob]:
        return (
            self.db.query(WSSEDTrainingJob)
            .filter(WSSEDTrainingJob.id == job_id)
            .first()
        )

    def get_latest_training_job(self, dataset_id: int) -> Optional[WSSEDTrainingJob]:
        """Most recent WSSED training job for a dataset (by id)."""
        return (
            self.db.query(WSSEDTrainingJob)
            .filter(WSSEDTrainingJob.dataset_id == dataset_id)
            .order_by(WSSEDTrainingJob.id.desc())
            .first()
        )

    def get_training_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Return the current status dict for a training job (DB only)."""
        job = self._get_training_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job.id,
            "status": job.status.value,
            "model_path": job.model_path,
            "model_paths": job.model_paths,
            "metrics": job.training_metrics,
            "error": job.error_message,
            "progress": job.progress or self._default_training_progress(job),
            "_updated_at": job.updated_at,  # internal; used by API for probe throttle
        }

    def _default_training_progress(self, job: WSSEDTrainingJob) -> Dict[str, Any]:
        """Build useful progress metadata before the first GPU sync arrives."""
        hyperparameters = job.hyperparameters or {}
        return {
            "phase": job.status.value.lower(),
            "current_epoch": None,
            "total_epochs": hyperparameters.get("epochs"),
            "model_name": hyperparameters.get("model_name") or job.model_name,
            "bag_seconds": hyperparameters.get("bag_seconds"),
            "hop_seconds": hyperparameters.get("hop_seconds"),
            "learning_rate": hyperparameters.get("learning_rate"),
            "threshold": hyperparameters.get("threshold"),
        }

    def is_status_stale(self, job_id: int, ttl_seconds: int) -> bool:
        """
        Return True if the DB status for this job has not been probed from the
        GPU server within the last ``ttl_seconds`` seconds.
        """
        job = self._get_training_job(job_id)
        if job is None:
            return False
        if job.updated_at is None:
            return True
        age = (datetime.now(timezone.utc) - job.updated_at).total_seconds()
        return age > ttl_seconds

    # ------------------------------------------------------------------ #
    # Active learning checkpoint registration                              #
    # ------------------------------------------------------------------ #

    def _ensure_training_job_registered_for_al(
        self,
        job: WSSEDTrainingJob,
    ) -> Optional[ALModelCheckpoint]:
        """Register the completed WSSED segment checkpoint as an AL model family."""
        checkpoint_path = self._select_preferred_checkpoint_path(job)
        if not checkpoint_path:
            logger.warning("WSSED job %s completed without a checkpoint path", job.id)
            return None

        input_dim, num_classes = self._infer_linear_checkpoint_shape(checkpoint_path)
        label_order = self._resolve_label_order(job, num_classes)
        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if dataset is None:
            raise ValueError(f"Dataset {job.dataset_id} not found")

        snippet_set_id, embedding_model_id = self._resolve_default_embedding_scope(dataset)
        family_name = self._model_family_name(job)
        version = f"job_{job.id}"
        label_config_path = self._write_label_config(
            dataset_id=job.dataset_id,
            family_name=family_name,
            version=version,
            job_id=job.id,
            label_order=label_order,
        )

        from app.schemas.pam_active_learning import ALModelType

        hyperparameters = {
            **(job.hyperparameters or {}),
            "training_mode": "wssed_remote_training",
            "wssed_training_job_id": job.id,
            "model_paths": job.model_paths or {},
            "preferred_checkpoint_key": self._selected_checkpoint_key(job),
            "label_order": label_order,
            "used_species": label_order,
            "n_dim": input_dim,
            "num_classes": num_classes,
            "embedding_model_id": embedding_model_id,
            "resolved_snippet_set_id": snippet_set_id,
            "threshold": (job.hyperparameters or {}).get("threshold", 0.5),
        }

        checkpoint = (
            self.db.query(ALModelCheckpoint)
            .filter(
                ALModelCheckpoint.dataset_id == job.dataset_id,
                ALModelCheckpoint.model_family_name == family_name,
                ALModelCheckpoint.version == version,
            )
            .one_or_none()
        )

        if checkpoint is None:
            checkpoint = ALModelCheckpoint(
                dataset_id=job.dataset_id,
                model_family_name=family_name,
                version=version,
                checkpoint_path=checkpoint_path,
                label_config_path=label_config_path,
                model_type=ALModelType.WSSED_BIRDNET_SEGMENT.value,
                hyperparameters=hyperparameters,
                is_base=1,
                parent_checkpoint_id=None,
                status=ALModelStatus.AVAILABLE,
            )
            self.db.add(checkpoint)
            self.db.flush()
        else:
            checkpoint.checkpoint_path = checkpoint_path
            checkpoint.label_config_path = label_config_path
            checkpoint.model_type = ALModelType.WSSED_BIRDNET_SEGMENT.value
            checkpoint.hyperparameters = hyperparameters
            checkpoint.status = ALModelStatus.AVAILABLE
            checkpoint.updated_at = datetime.now(timezone.utc)

        self._set_active_family_checkpoint(job.dataset_id, family_name, checkpoint.id)

        metrics = dict(job.training_metrics or {})
        metrics["al_checkpoint_id"] = checkpoint.id
        metrics["al_model_family_name"] = family_name
        metrics["al_label_order"] = label_order
        job.training_metrics = metrics

        self.db.commit()
        self.db.refresh(checkpoint)
        logger.info(
            "Registered WSSED training job %s as AL checkpoint %s (%s)",
            job.id,
            checkpoint.id,
            family_name,
        )
        return checkpoint

    def _ensure_training_job_inference_enqueued(
        self,
        job: WSSEDTrainingJob,
        checkpoint: ALModelCheckpoint,
    ) -> None:
        hyperparameters = checkpoint.hyperparameters or {}
        snippet_set_id = hyperparameters.get("resolved_snippet_set_id")
        if not snippet_set_id:
            logger.warning(
                "Skipping WSSED inference enqueue for job %s: no snippet set available",
                job.id,
            )
            return

        existing_job_id = (job.training_metrics or {}).get("al_inference_job_id")
        if existing_job_id:
            return

        existing = (
            self.db.query(ALRetrainJob)
            .filter(
                ALRetrainJob.model_checkpoint_id == checkpoint.id,
                ALRetrainJob.dataset_id == job.dataset_id,
                ALRetrainJob.trigger == "wssed_inference",
            )
            .order_by(ALRetrainJob.created_at.desc())
            .first()
        )
        if existing is not None and existing.status in {
            ALRetrainStatus.PENDING,
            ALRetrainStatus.RUNNING,
            ALRetrainStatus.COMPLETED,
        }:
            metrics = dict(job.training_metrics or {})
            metrics["al_inference_job_id"] = existing.id
            job.training_metrics = metrics
            self.db.commit()
            return

        inference_job = ALRetrainJob(
            model_checkpoint_id=checkpoint.id,
            dataset_id=job.dataset_id,
            trigger="wssed_inference",
            feedback_count=0,
            status=ALRetrainStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(inference_job)
        self.db.commit()
        self.db.refresh(inference_job)

        body = {
            "model_family_name": checkpoint.model_family_name,
            "dataset_id": job.dataset_id,
            "snippet_set_id": int(snippet_set_id),
            "device": "cpu",
            "threshold": hyperparameters.get("threshold", 0.5),
            "force_refresh": True,
            "sample_suggestion": False,
        }

        try:
            from app.tasks.pam_al_tasks import pam_al_create_predictions

            pam_al_create_predictions.delay(
                job_id=inference_job.id,
                inference_body=body,
            )
            metrics = dict(job.training_metrics or {})
            metrics["al_inference_job_id"] = inference_job.id
            job.training_metrics = metrics
            self.db.commit()
            logger.info(
                "Queued WSSED inference job %s for checkpoint %s",
                inference_job.id,
                checkpoint.id,
            )
        except Exception as e:
            inference_job.status = ALRetrainStatus.FAILED
            inference_job.error_message = str(e)
            inference_job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            logger.warning("Could not queue WSSED inference job %s: %s", inference_job.id, e)

    def _select_preferred_checkpoint_path(self, job: WSSEDTrainingJob) -> Optional[str]:
        paths = job.model_paths or {}
        return (
            paths.get("best_micro_model_segment")
            or paths.get("best_micro_model")
        )

    def _selected_checkpoint_key(self, job: WSSEDTrainingJob) -> Optional[str]:
        selected = self._select_preferred_checkpoint_path(job)
        for key, path in (job.model_paths or {}).items():
            if path == selected:
                return key
        return None

    def _infer_linear_checkpoint_shape(self, checkpoint_path: str) -> tuple[int, int]:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid WSSED checkpoint: {checkpoint_path}")

        weight = state_dict.get("linear.weight")
        if weight is None:
            raise ValueError(
                f"WSSED AL expects a BirdNET segment checkpoint with linear.weight: {checkpoint_path}"
            )
        return int(weight.shape[1]), int(weight.shape[0])

    def _resolve_label_order(self, job: WSSEDTrainingJob, num_classes: int) -> List[str]:
        metrics = job.training_metrics or {}
        hyperparameters = job.hyperparameters or {}

        for candidate in (
            metrics.get("label_order"),
            metrics.get("used_species"),
            hyperparameters.get("target_species"),
            hyperparameters.get("species_list"),
        ):
            labels = self._normalize_labels(candidate)
            if len(labels) == num_classes:
                return labels

        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if dataset is not None:
            labels = self._labels_from_dataset_metadata(dataset)
            if len(labels) == num_classes:
                return labels

        logger.warning(
            "Could not resolve %s WSSED labels for job %s; using generic class names",
            num_classes,
            job.id,
        )
        return [f"class_{idx}" for idx in range(num_classes)]

    def _normalize_labels(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return []

    def _labels_from_dataset_metadata(self, dataset: Dataset) -> List[str]:
        dataset_path = self._resolve_dataset_path(dataset.source_uri)
        metadata_path = dataset_path / "metadata_filtered_filled.csv"
        if not metadata_path.is_file():
            return []

        labels: set[str] = set()
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Code" not in (reader.fieldnames or []):
                return []
            for row in reader:
                code = str(row.get("Code") or "").strip()
                if code and code != "IGNORE":
                    labels.add(code)
        return sorted(labels)

    def _resolve_dataset_path(self, source_uri: str) -> Path:
        path = Path(source_uri)
        if path.is_absolute():
            return path
        if settings.DATA_ROOT:
            return Path(settings.DATA_ROOT) / path
        return path

    def _resolve_default_embedding_scope(self, dataset: Dataset) -> tuple[Optional[int], Optional[int]]:
        if dataset.default_snippet_set_id:
            snippet_set = (
                self.db.query(SnippetSet)
                .filter(SnippetSet.id == dataset.default_snippet_set_id)
                .one_or_none()
            )
            if snippet_set is not None:
                return snippet_set.id, snippet_set.embedding_model_id

        job = (
            self.db.query(EmbeddingJob)
            .filter(
                EmbeddingJob.dataset_id == dataset.id,
                EmbeddingJob.status == EmbeddingJobStatus.COMPLETED,
            )
            .order_by(EmbeddingJob.completed_at.desc().nullslast(), EmbeddingJob.created_at.desc())
            .first()
        )
        if job is None:
            return None, None
        return job.snippet_set_id, job.embedding_model_id

    def _write_label_config(
        self,
        dataset_id: int,
        family_name: str,
        version: str,
        job_id: int,
        label_order: List[str],
    ) -> str:
        checkpoint_dir = Path(settings.PAM_CHECKPOINTS_DIR) / "wssed_active_learning" / str(dataset_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{family_name}_{version}_labels_{job_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"species_list": label_order}, f, indent=2)
        return str(path)

    def _model_family_name(self, job: WSSEDTrainingJob) -> str:
        model_name = (job.model_name or "model").strip().lower().replace(" ", "_")
        return f"wssed_{model_name}_segment"

    def _set_active_family_checkpoint(
        self,
        dataset_id: int,
        family_name: str,
        checkpoint_id: int,
    ) -> None:
        row = (
            self.db.query(ALModelFamilyState)
            .filter(
                ALModelFamilyState.dataset_id == dataset_id,
                ALModelFamilyState.model_family_name == family_name,
            )
            .one_or_none()
        )
        if row is None:
            row = ALModelFamilyState(
                dataset_id=dataset_id,
                model_family_name=family_name,
                active_model_checkpoint_id=checkpoint_id,
            )
            self.db.add(row)
        else:
            row.active_model_checkpoint_id = checkpoint_id

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_or_none_species_model(
        self, dataset_id: int, species_name: str
    ) -> Optional[WSSEDSpeciesModel]:
        return (
            self.db.query(WSSEDSpeciesModel)
            .filter(
                WSSEDSpeciesModel.dataset_id == dataset_id,
                WSSEDSpeciesModel.species_name == species_name,
            )
            .first()
        )
