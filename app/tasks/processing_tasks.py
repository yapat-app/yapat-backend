"""
Celery tasks for dataset scanning and orchestration.
"""

from contextlib import contextmanager
from celery import shared_task
import os
import logging

from app.database import SessionLocal
from app.services.dataset_service import DatasetService
from app.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Context manager for DB session handling
# --------------------------------------------------------------------

@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------
# Task: Scan dataset for recordings
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.processing_tasks.scan_dataset")
def scan_dataset(self, dataset_id: int):
    """
    Discover recordings in dataset.source_uri.
    Uses DatasetService.scan_recordings().
    """
    import logging
    logger = logging.getLogger(__name__)
    
    with session_scope() as db:
        svc = DatasetService(db)

        dataset = svc.get_dataset(dataset_id)
        if dataset is None:
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        self.update_state(state="SCANNING", meta={"dataset_id": dataset_id})
        logger.info(f"Starting scan for dataset {dataset_id} (source_uri: {dataset.source_uri})")

        try:
            new_recs = svc.scan_recordings(dataset)
            logger.info(f"Scan completed for dataset {dataset_id}: {len(new_recs)} recordings created")
        except Exception as e:
            error_msg = f"Scan failed for dataset {dataset_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "recordings_created": len(new_recs),
        }


# --------------------------------------------------------------------
# Orchestration: dataset processing = scan only
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.processing_tasks.process_dataset")
def process_dataset(self, dataset_id: int):
    """
    Trigger the dataset scanning pipeline.

    New architecture:
    - Only scanning runs at dataset creation
    - Snippet generation is triggered by embedding jobs, not datasets
    - For WEAKLY_LABELED datasets, auto-register species models after scanning
    """
    # Start scanning
    scan_result = scan_dataset.delay(dataset_id)
    
    # Chain auto-registration after scanning completes
    # This will only register models if:
    # - AUTO_REGISTER_SPECIES_MODELS is True
    # - ACTIVE_LEARNING_MODELS_DIR is configured
    # - Dataset type is WEAKLY_LABELED
    auto_register_result = auto_register_species_models.apply_async(
        args=[dataset_id],
        link_error=None  # Don't fail the whole pipeline if auto-registration fails
    )

    return {
        "status": "submitted",
        "dataset_id": dataset_id,
        "scan_task_id": scan_result.id,
        "auto_register_task_id": auto_register_result.id,
    }

@shared_task(bind=True)
def generate_snippets(self, dataset_id: int, snippet_set_id: int):
    """
    Placeholder for snippet segmentation logic.

    In the new SnippetSet architecture, snippet generation
    happens inside embedding jobs (run_embedding).
    """
    return {"status": "not_implemented"}


# --------------------------------------------------------------------
# Task: Auto-register species models for WEAKLY_LABELED datasets
# --------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.processing_tasks.auto_register_species_models")
def auto_register_species_models(self, dataset_id: int):
    """
    Automatically register species models for a WEAKLY_LABELED dataset.
    
    This task:
    1. Checks if dataset type is WEAKLY_LABELED
    2. Extracts species from filenames (FNJV format)
    3. Registers a species model for each unique species
    
    Args:
        dataset_id: ID of the dataset
    
    Returns:
        dict with status and registered species
    """
    logger.info(f"Starting auto-registration of species models for dataset {dataset_id}")
    
    # Check if auto-registration is enabled
    if not settings.AUTO_REGISTER_SPECIES_MODELS:
        logger.info("Auto-registration disabled in settings")
        return {"status": "skipped", "reason": "disabled_in_settings"}
    
    # Check if models directory is configured
    if not settings.ACTIVE_LEARNING_MODELS_DIR:
        logger.warning("ACTIVE_LEARNING_MODELS_DIR not configured, skipping auto-registration")
        return {"status": "skipped", "reason": "models_dir_not_configured"}
    
    # Check if models directory exists
    if not os.path.isdir(settings.ACTIVE_LEARNING_MODELS_DIR):
        logger.warning(f"Models directory not found: {settings.ACTIVE_LEARNING_MODELS_DIR}")
        return {"status": "skipped", "reason": "models_dir_not_found"}
    
    with session_scope() as db:
        from app.services.dataset_service import DatasetService
        from app.services.wssed_species_extractor import get_dataset_species_list
        from app.services.wssed import ActiveLearningService
        from app.models.dataset import DatasetType
        
        # Get dataset
        dataset_svc = DatasetService(db)
        dataset = dataset_svc.get_dataset(dataset_id)
        
        if not dataset:
            error_msg = f"Dataset {dataset_id} not found"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        # Check if dataset is WEAKLY_LABELED
        if dataset.dataset_type != DatasetType.WEAKLY_LABELED:
            logger.info(f"Dataset {dataset_id} is not WEAKLY_LABELED (type: {dataset.dataset_type}), skipping")
            return {
                "status": "skipped",
                "reason": "not_weakly_labeled",
                "dataset_type": dataset.dataset_type.value
            }
        
        # Get species list from dataset
        try:
            species_list = get_dataset_species_list(dataset_id, db)
            logger.info(f"Found {len(species_list)} species in dataset {dataset_id}: {species_list}")
        except Exception as e:
            error_msg = f"Failed to extract species from dataset {dataset_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
        
        if not species_list:
            logger.warning(f"No species found in dataset {dataset_id}")
            return {
                "status": "ok",
                "registered_count": 0,
                "species": [],
                "message": "No species found"
            }
        
        # Register species models
        al_service = ActiveLearningService(db)
        registered_species = []
        failed_species = []
        
        for species_name in species_list:
            try:
                model = al_service.register_species_model(
                    species_name=species_name,
                    dataset_id=dataset_id,
                    base_model_directory=settings.ACTIVE_LEARNING_MODELS_DIR,
                    metric_type="macro",
                    prediction_level="segment",
                    model_version="auto_v1.0"
                )
                registered_species.append({
                    "species_name": species_name,
                    "model_id": model.id
                })
                logger.info(f"Registered species model for {species_name} (ID: {model.id})")
            except Exception as e:
                failed_species.append({
                    "species_name": species_name,
                    "error": str(e)
                })
                logger.error(f"Failed to register model for {species_name}: {str(e)}", exc_info=True)
        
        result = {
            "status": "ok",
            "dataset_id": dataset_id,
            "registered_count": len(registered_species),
            "failed_count": len(failed_species),
            "registered_species": registered_species,
            "failed_species": failed_species
        }
        
        logger.info(
            f"Auto-registration completed for dataset {dataset_id}: "
            f"{len(registered_species)} successful, {len(failed_species)} failed"
        )
        
        return result
