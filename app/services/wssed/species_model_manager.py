"""
Species Model Manager

Handles CRUD operations for species-specific models.
"""

from typing import Optional, List, Dict, Any
import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.wssed import WSSEDSpeciesModel, TrainingStatus

logger = logging.getLogger(__name__)


class SpeciesModelManager:
    """Manages species-specific model records in the database."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def get_safe_directory_name(species_name: str) -> str:
        """
        Convert species name to safe directory name.
        
        Args:
            species_name: Species name (e.g., "FNJV Species" or "Bird/Owl")
            
        Returns:
            Safe directory name (e.g., "fnjv_species" or "bird_owl")
        """
        return species_name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")

    def register_model(
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
        
        Creates a species-specific subdirectory within the base directory
        to ensure each species has its own checkpoint storage.

        Args:
            species_name: Name of the species
            dataset_id: Dataset ID
            base_model_directory: Base directory for all species models
            metric_type: "macro" or "micro"
            prediction_level: "segment" or "clip"
            model_version: Optional version identifier
            hyperparameters: Optional model hyperparameters

        Returns:
            WSSEDSpeciesModel instance
        """
        # Species-specific subdirectory: safe name from species (e.g. Dendropsophus_nanus -> dendropsophus_nanus)
        dir_name = self.get_safe_directory_name(species_name)
        species_model_directory = os.path.join(base_model_directory, dir_name)
        
        # Create directory if it doesn't exist
        os.makedirs(species_model_directory, exist_ok=True)
        logger.info(f"Species model directory: {species_model_directory}")
        
        # Check if model already exists
        existing_model = self.db.query(WSSEDSpeciesModel).filter(
            and_(
                WSSEDSpeciesModel.species_name == species_name,
                WSSEDSpeciesModel.dataset_id == dataset_id
            )
        ).first()

        if existing_model:
            # Update existing model
            existing_model.model_directory = species_model_directory
            existing_model.metric_type = metric_type
            existing_model.prediction_level = prediction_level
            existing_model.model_version = model_version
            existing_model.hyperparameters = hyperparameters
            existing_model.updated_at = datetime.utcnow()
            logger.info(f"Updated existing species model: {species_name} (ID: {existing_model.id})")
            model = existing_model
        else:
            # Create new model
            model = WSSEDSpeciesModel(
                species_name=species_name,
                dataset_id=dataset_id,
                model_directory=species_model_directory,
                metric_type=metric_type,
                prediction_level=prediction_level,
                model_version=model_version,
                hyperparameters=hyperparameters,
                status=TrainingStatus.COMPLETED
            )
            self.db.add(model)
            logger.info(f"Registered new species model: {species_name}")

        self.db.commit()
        self.db.refresh(model)
        return model

    def get_by_id(self, species_model_id: int) -> Optional[WSSEDSpeciesModel]:
        """Get a species model by ID."""
        return self.db.query(WSSEDSpeciesModel).filter(
            WSSEDSpeciesModel.id == species_model_id
        ).first()

    def get_by_name(
        self, species_name: str, dataset_id: int
    ) -> Optional[WSSEDSpeciesModel]:
        """Get a species model by name and dataset (exact match on species_name)."""
        return self.db.query(WSSEDSpeciesModel).filter(
            and_(
                WSSEDSpeciesModel.species_name == species_name,
                WSSEDSpeciesModel.dataset_id == dataset_id
            )
        ).first()

    def list_models(self, dataset_id: Optional[int] = None) -> List[WSSEDSpeciesModel]:
        """List all species models, optionally filtered by dataset."""
        query = self.db.query(WSSEDSpeciesModel)
        if dataset_id is not None:
            query = query.filter(WSSEDSpeciesModel.dataset_id == dataset_id)
        return query.all()

    def update_version(self, species_model: WSSEDSpeciesModel, version: str):
        """Update model version and timestamp."""
        species_model.model_version = version
        species_model.updated_at = datetime.utcnow()
        self.db.commit()
