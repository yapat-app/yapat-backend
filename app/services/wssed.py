"""
WSSED-related services.

Currently used by dataset processing to auto-register species models for
FOCAL_RECORDINGS datasets.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.wssed import WSSEDSpeciesModel, TrainingStatus


class ActiveLearningService:
    """
    Minimal service for managing WSSED species model metadata.
    """

    def __init__(self, db: Session):
        self.db = db

    def register_species_model(
        self,
        species_name: str,
        dataset_id: int,
        base_model_directory: str,
        metric_type: str = "macro",
        prediction_level: str = "segment",
        model_version: str | None = None,
        hyperparameters: dict | None = None,
    ) -> WSSEDSpeciesModel:
        """
        Create (or return existing) `WSSEDSpeciesModel` row for a dataset+species.
        """
        existing = (
            self.db.query(WSSEDSpeciesModel)
            .filter(
                WSSEDSpeciesModel.dataset_id == dataset_id,
                WSSEDSpeciesModel.species_name == species_name,
            )
            .first()
        )
        if existing is not None:
            return existing

        model = WSSEDSpeciesModel(
            species_name=species_name,
            dataset_id=dataset_id,
            model_directory=base_model_directory,
            metric_type=metric_type,
            prediction_level=prediction_level,
            model_version=model_version,
            hyperparameters=hyperparameters,
            status=TrainingStatus.COMPLETED,
        )
        self.db.add(model)
        # Ensure ID is available to caller even if commit happens outside.
        self.db.flush()
        return model

