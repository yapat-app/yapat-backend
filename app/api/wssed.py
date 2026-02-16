"""
WSSED API endpoints

Provides REST API for WSSED training, detection, predictions, and feedback.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.wssed import (
    TrainingJobCreate,
    TrainingJob,
    TrainingStatus,
    Prediction,
    RecordingTimeline,
    FeedbackSubmit,
    FeedbackResponse,
    FeedbackStats,
    SpeciesList,
    DetectionResponse,
    FeedbackType,
    # Active Learning schemas
    SpeciesModelCreate,
    SpeciesModel,
    ActiveLearningSuggestionsRequest,
    ActiveLearningSuggestionsResponse,
    ActiveLearningSuggestion,
    ActiveLearningLabelSubmit,
    ActiveLearningLabelResponse,
    ActiveLearningStats,
    SnippetLabelResponse,
    PredictionHistogramResponse,
)
from app.services.wssed_service import WSSEDService
from app.services.wssed import ActiveLearningService
from app.tasks.wssed_tasks import trigger_wssed_training, trigger_wssed_detection

router = APIRouter()


# ============ TRAINING ENDPOINTS ============

@router.post("/training-jobs", response_model=TrainingJob, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job_in: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create and start a new WSSED training job.
    
    The training will run on a remote GPU server. This endpoint returns immediately
    with the job details. Use GET /training-jobs/{job_id} to poll for status updates.
    """
    service = WSSEDService(db)
    
    try:
        # Create training job
        job = service.create_training_job(
            dataset_id=job_in.dataset_id,
            hyperparameters=job_in.hyperparameters
        )
        
        # Trigger training in background
        background_tasks.add_task(trigger_wssed_training, job.id)
        
        return job
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/training-jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get training job status and metrics.
    
    If the job is still training, this will poll the GPU server for the latest status.
    """
    service = WSSEDService(db)
    job = service.get_training_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Poll GPU server for latest status if still training
    if job.status == TrainingStatus.TRAINING:
        try:
            job = await service.update_training_status(job_id)
        except Exception:
            # If polling fails, just return current status
            pass
    
    return job


@router.get("/training-jobs", response_model=List[TrainingJob])
def list_training_jobs(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    status: Optional[TrainingStatus] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all training jobs with optional filtering.
    """
    service = WSSEDService(db)
    return service.list_training_jobs(dataset_id, status, skip, limit)


# ============ DETECTION ENDPOINTS ============

@router.post("/training-jobs/{job_id}/detect", response_model=DetectionResponse)
async def trigger_detection(
    job_id: int,
    threshold: float = Query(0.5, ge=0, le=1, description="Detection confidence threshold"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Run detection using a trained model.
    
    This will apply the model to all recordings in the dataset and store predictions.
    The operation runs asynchronously on the GPU server.
    """
    service = WSSEDService(db)
    
    job = service.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Training job must be completed. Current status: {job.status.value}"
        )
    
    try:
        # Trigger detection in background
        background_tasks.add_task(trigger_wssed_detection, job_id, threshold)
        
        return DetectionResponse(
            message="Detection started",
            task_id=f"detection_{job_id}",
            training_job_id=job_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger detection: {str(e)}")


# ============ PREDICTION ENDPOINTS ============

@router.get("/predictions", response_model=List[Prediction])
def list_predictions(
    recording_id: Optional[int] = Query(None, description="Filter by recording ID"),
    training_job_id: Optional[int] = Query(None, description="Filter by training job ID"),
    species_name: Optional[str] = Query(None, description="Filter by species name"),
    threshold: float = Query(0.0, ge=0, le=1, description="Minimum confidence threshold (ignored if uncertain_range is provided)"),
    uncertain_range: Optional[float] = Query(None, ge=0, le=0.5, description="Filter for uncertain predictions close to 0.5. For example, 0.1 will show predictions where 0.4 <= confidence <= 0.6"),
    feedback_filter: Optional[FeedbackType] = Query(None, description="Filter by feedback status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List predictions with filtering options.
    
    Use this to browse all predictions or filter by specific criteria.
    
    To show uncertain predictions (close to 0.5), use the uncertain_range parameter.
    For example, uncertain_range=0.1 will show predictions where confidence is between 0.4 and 0.6.
    """
    service = WSSEDService(db)
    return service.list_predictions(
        recording_id=recording_id,
        training_job_id=training_job_id,
        species_name=species_name,
        threshold=threshold,
        uncertain_range=uncertain_range,
        feedback_filter=feedback_filter,
        skip=skip,
        limit=limit
    )


@router.get("/recordings/{recording_id}/predictions-timeline", response_model=RecordingTimeline)
def get_recording_timeline(
    recording_id: int,
    training_job_id: Optional[int] = Query(None, description="Filter by specific training job"),
    threshold: float = Query(0.5, ge=0, le=1, description="Minimum confidence threshold (ignored if uncertain_range is provided)"),
    uncertain_range: Optional[float] = Query(None, ge=0, le=0.5, description="Filter for uncertain predictions close to 0.5. For example, 0.1 will show predictions where 0.4 <= confidence <= 0.6"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all predictions for a recording formatted for timeline visualization.
    
    This endpoint is optimized for displaying predictions on an audio timeline player.
    It returns predictions sorted by time with their confidence scores and feedback status.
    
    To show uncertain predictions (close to 0.5), use the uncertain_range parameter.
    For example, uncertain_range=0.1 will show predictions where confidence is between 0.4 and 0.6.
    """
    service = WSSEDService(db)
    
    try:
        return service.get_recording_timeline(recording_id, training_job_id, threshold, uncertain_range)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============ FEEDBACK ENDPOINTS ============

@router.post("/predictions/{prediction_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    prediction_id: int,
    feedback_in: FeedbackSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Submit feedback on a prediction (accept or reject).
    
    - **ACCEPTED**: Creates a strong_positive label (species present) for retraining
    - **REJECTED**: Creates a strong_negative label (species absent) for retraining
    
    If 5 or more feedbacks have been submitted since the last training,
    this will automatically trigger a retraining job.
    """
    service = WSSEDService(db)
    
    try:
        result = service.submit_feedback(prediction_id, feedback_in.feedback)
        
        message = f"Feedback submitted successfully"
        if result['retraining_triggered']:
            message += ". Retraining automatically triggered."
        
        return FeedbackResponse(
            success=True,
            retraining_triggered=result['retraining_triggered'],
            feedback_count=result['feedback_count'],
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/training-jobs/{job_id}/feedback-stats", response_model=FeedbackStats)
def get_feedback_stats(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get feedback statistics for a training job.
    
    Returns counts of accepted, rejected, and pending predictions,
    plus the number of new feedbacks since the job was created.
    """
    service = WSSEDService(db)
    
    job = service.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return service.get_feedback_stats(job_id)


# ============ RETRAINING ENDPOINTS ============

@router.post("/training-jobs/{job_id}/retrain", response_model=TrainingJob, status_code=status.HTTP_201_CREATED)
def manual_retrain(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually trigger retraining with accumulated feedback.
    
    This creates a new training job that includes:
    - Original weak labels from the dataset
    - Strong labels from accepted predictions
    - Negative examples from rejected predictions
    """
    service = WSSEDService(db)
    
    try:
        new_job = service.trigger_retraining(job_id)
        
        # Trigger training in background
        background_tasks.add_task(trigger_wssed_training, new_job.id)
        
        return new_job
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


# ============ SPECIES ENDPOINTS ============

@router.get("/datasets/{dataset_id}/species", response_model=SpeciesList)
def get_dataset_species(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get list of species detected in a dataset.
    
    Species are extracted from the FNJV filename format.
    """
    service = WSSEDService(db)
    
    try:
        return service.get_dataset_species(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get species list: {str(e)}")


# ============ ACTIVE LEARNING ENDPOINTS ============

@router.post("/species-models", response_model=SpeciesModel, status_code=status.HTTP_201_CREATED)
def register_species_model(
    model_in: SpeciesModelCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Register a species-specific model for active learning.
    
    This creates or updates a species model entry. A species-specific subdirectory
    will be created within the base model directory to store checkpoints for this species.
    
    Example: base_directory="/models" + species="FNJV Species" -> "/models/fnjv_species/"
    """
    service = ActiveLearningService(db)
    
    try:
        model = service.register_species_model(
            species_name=model_in.species_name,
            dataset_id=model_in.dataset_id,
            base_model_directory=model_in.model_directory,
            metric_type=model_in.metric_type,
            prediction_level=model_in.prediction_level,
            model_version=model_in.model_version,
            hyperparameters=model_in.hyperparameters
        )
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register species model: {str(e)}")


@router.get("/species-models", response_model=List[SpeciesModel])
def list_species_models(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all registered species models, optionally filtered by dataset.
    """
    service = ActiveLearningService(db)
    
    try:
        models = service.list_species_models(dataset_id=dataset_id)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list species models: {str(e)}")


@router.get("/species-models/{model_id}", response_model=SpeciesModel)
def get_species_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a specific species model by ID.
    """
    service = ActiveLearningService(db)
    
    model = service.get_species_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Species model not found")
    
    return model


@router.post("/active-learning/suggestions", response_model=ActiveLearningSuggestionsResponse)
def get_active_learning_suggestions(
    request: ActiveLearningSuggestionsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get active learning suggestions for a species and snippet set.
    
    This endpoint:
    1. Loads the species-specific model
    2. Loads embeddings for the snippet set
    3. Applies the active learning query strategy
    4. Returns the most informative snippets for labeling
    
    Query strategies:
    - "uncertainty": Samples close to decision boundary (prob ~ 0.5)
    - "diversity": Diverse samples using embedding space (requires Z_pool)
    - "density": Samples in high-density regions (requires Z_pool)
    - "random": Random sampling
    """
    service = ActiveLearningService(db)
    
    try:
        result = service.get_suggestions(
            snippet_set_id=request.snippet_set_id,
            species_name=request.species_name,
            dataset_id=request.dataset_id,
            strategy=request.strategy,
            k=request.k,
            device=request.device,
            seed=request.seed
        )
        
        # Format response with suggestions (probs only)
        suggestions = [
            ActiveLearningSuggestion(
                snippet_id=sid,
                predicted_probability=prob
            )
            for sid, prob in zip(result["snippet_ids"], result["probs"])
        ]
        
        return ActiveLearningSuggestionsResponse(
            snippet_ids=result["snippet_ids"],
            probs=result["probs"],
            n_labeled=result["n_labeled"],
            model_info=result["model_info"],
            suggestions=suggestions
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.post("/active-learning/labels", response_model=ActiveLearningLabelResponse)
def submit_active_learning_labels(
    label_data: ActiveLearningLabelSubmit,
    device: str = Query("cpu", description="Device for training"),
    epochs: int = Query(5, ge=1, le=100, description="Training epochs"),
    lr: float = Query(1e-3, gt=0, description="Learning rate"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Submit a single user label for active learning and optionally trigger retraining.
    
    Label values:
    - 0: Reject (species not present)
    - 1: Accept (species present)
    
    The model will automatically retrain after every 5 labels.
    """
    service = ActiveLearningService(db)
    
    try:
        # Convert single label to dict format for service
        snippet_id_to_label = {label_data.snippet_id: label_data.label}
        
        result = service.submit_labels_and_maybe_retrain(
            snippet_set_id=label_data.snippet_set_id,
            species_name=label_data.species_name,
            dataset_id=label_data.dataset_id,
            snippet_id_to_label=snippet_id_to_label,
            retrain_every=5,
            device=device,
            epochs=epochs,
            lr=lr
        )
        
        return ActiveLearningLabelResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit labels: {str(e)}")


@router.post("/active-learning/retrain", response_model=ActiveLearningLabelResponse)
def manual_retrain_model(
    snippet_set_id: int = Query(..., description="Snippet set ID"),
    species_name: str = Query(..., description="Species name"),
    dataset_id: int = Query(..., description="Dataset ID"),
    device: str = Query("cpu", description="Device for training"),
    epochs: int = Query(5, ge=1, le=100, description="Training epochs"),
    lr: float = Query(1e-3, gt=0, description="Learning rate"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually trigger retraining for a species model.
    
    This endpoint allows you to retrain the model at any time,
    regardless of the number of labels or automatic retraining schedule.
    
    The model will be retrained using all currently available labels
    for the specified species in the given snippet set.
    """
    service = ActiveLearningService(db)
    
    try:
        result = service.manual_retrain(
            snippet_set_id=snippet_set_id,
            species_name=species_name,
            dataset_id=dataset_id,
            device=device,
            epochs=epochs,
            lr=lr
        )
        
        return ActiveLearningLabelResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")


@router.get("/active-learning/stats", response_model=ActiveLearningStats)
def get_active_learning_stats(
    species_model_id: int = Query(..., description="Species model ID"),
    snippet_set_id: Optional[int] = Query(None, description="Optional snippet set filter"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get statistics for active learning progress.
    
    Returns counts of total predictions, labeled, unlabeled, accepted, and rejected.
    """
    service = ActiveLearningService(db)
    
    try:
        stats = service.get_statistics(
            species_model_id=species_model_id,
            snippet_set_id=snippet_set_id
        )
        return ActiveLearningStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/species-models/{model_id}/labels", response_model=List[SnippetLabelResponse])
def get_species_model_labels(
    model_id: int,
    snippet_set_id: Optional[int] = Query(None, description="Filter by snippet set"),
    labeled_only: bool = Query(False, description="Return only labeled snippets"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all snippet labels for a species model.

    Useful for reviewing labeled data and monitoring active learning progress.
    """
    from app.models.wssed import WSSEDSnippetLabel
    from app.models.snippet import Snippet
    from sqlalchemy import and_

    query = db.query(WSSEDSnippetLabel).filter(
        WSSEDSnippetLabel.species_model_id == model_id
    )

    if snippet_set_id:
        query = query.join(Snippet).filter(Snippet.snippet_set_id == snippet_set_id)

    if labeled_only:
        query = query.filter(WSSEDSnippetLabel.user_label.isnot(None))

    labels = query.offset(skip).limit(limit).all()

    return labels


@router.get("/species-models/{model_id}/histogram", response_model=PredictionHistogramResponse)
def get_species_prediction_histogram(
    model_id: int,
    snippet_set_id: Optional[int] = Query(None, description="Restrict to this snippet set (e.g. weekly labeled set)"),
    num_bins: int = Query(10, ge=1, le=100, description="Number of bins in [0, 1] for the histogram"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get histogram of model predictions for a species' snippets.

    When a species (e.g. BOARAN) is selected from the file explorer, returns the distribution
    of model outputs: X axis = prediction value in [0, 1], Y axis = count of snippets in each bin.
    Uses all snippets for this species model, optionally restricted to one snippet_set_id (weekly).
    """
    service = ActiveLearningService(db)
    try:
        result = service.get_prediction_histogram(
            species_model_id=model_id,
            snippet_set_id=snippet_set_id,
            num_bins=num_bins,
        )
        return PredictionHistogramResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
