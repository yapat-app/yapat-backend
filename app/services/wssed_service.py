"""
WSSED Service

Handles communication with the WSSED GPU server and manages training job
state in the local database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import Dataset
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
        if data.get("metrics"):
            job.training_metrics = data["metrics"]
        if data.get("error"):
            job.error_message = data["error"]
        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
            job.completed_at = datetime.now(timezone.utc)

        self.db.commit()
        self.db.refresh(job)
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

        from app.tasks.wssed_tasks import trigger_wssed_training
        trigger_wssed_training.delay(job.id)

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

    def _get_training_job(self, job_id: int) -> Optional[WSSEDTrainingJob]:
        return (
            self.db.query(WSSEDTrainingJob)
            .filter(WSSEDTrainingJob.id == job_id)
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
            "metrics": job.training_metrics,
            "error": job.error_message,
            "progress": None,
        }

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
