"""
WSSED Service Layer

Handles all WSSED-related business logic including training, detection, predictions, and feedback.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from datetime import datetime
import httpx
import logging

from app.models.wssed import WSSEDTrainingJob, WSSEDPrediction, WSSEDStrongLabel, TrainingStatus, FeedbackType, LabelType
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.schemas.wssed import (
    WSSEDHyperparameters,
    TrainingJob,
    Prediction,
    TimelinePrediction,
    RecordingTimeline,
    FeedbackStats,
    SpeciesList
)
from app.services.wssed_species_extractor import get_dataset_species_list
from app.config import settings

logger = logging.getLogger(__name__)


class WSSEDService:
    """Service for managing WSSED training, detection, and predictions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.gpu_server_url = settings.WSSED_GPU_SERVER_URL
    
    # ============ TRAINING JOB MANAGEMENT ============
    
    def create_training_job(
        self,
        dataset_id: int,
        hyperparameters: WSSEDHyperparameters
    ) -> WSSEDTrainingJob:
        """
        Create a new training job.
        
        Args:
            dataset_id: ID of the dataset to train on
            hyperparameters: Training hyperparameters
        
        Returns:
            Created training job
        
        Raises:
            ValueError: If dataset not found
        """
        # Validate dataset exists
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Create training job
        job = WSSEDTrainingJob(
            dataset_id=dataset_id,
            model_name=hyperparameters.model_name,
            hyperparameters=hyperparameters.dict(),
            status=TrainingStatus.PENDING
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        logger.info(f"Created training job {job.id} for dataset {dataset_id}")
        return job
    
    def get_training_job(self, job_id: int) -> Optional[WSSEDTrainingJob]:
        """Get training job by ID"""
        return self.db.query(WSSEDTrainingJob).filter(
            WSSEDTrainingJob.id == job_id
        ).first()
    
    def list_training_jobs(
        self,
        dataset_id: Optional[int] = None,
        status: Optional[TrainingStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WSSEDTrainingJob]:
        """List training jobs with optional filtering"""
        query = self.db.query(WSSEDTrainingJob)
        
        if dataset_id is not None:
            query = query.filter(WSSEDTrainingJob.dataset_id == dataset_id)
        
        if status is not None:
            query = query.filter(WSSEDTrainingJob.status == status)
        
        return query.order_by(
            WSSEDTrainingJob.created_at.desc()
        ).offset(skip).limit(limit).all()
    
    def update_training_job_status(
        self,
        job_id: int,
        status: TrainingStatus,
        model_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> WSSEDTrainingJob:
        """Update training job status and results"""
        job = self.get_training_job(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        job.status = status
        
        if model_path:
            job.model_path = model_path
        
        if metrics:
            job.training_metrics = metrics
        
        if error_message:
            job.error_message = error_message
        
        if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            job.completed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(job)
        
        logger.info(f"Updated training job {job_id} status to {status}")
        return job
    
    async def trigger_remote_training(self, job_id: int) -> str:
        """
        Trigger training on remote GPU server.
        
        Args:
            job_id: ID of the training job
        
        Returns:
            Task ID from GPU server
        """
        job = self.get_training_job(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {job.dataset_id} not found")
        
        # Prepare request payload
        payload = {
            "job_id": job.id,
            "dataset_id": job.dataset_id,
            "dataset_path": dataset.source_uri,
            "hyperparameters": job.hyperparameters,
            "feedback_labels": None  # No feedback for initial training
        }
        
        # Send request to GPU server
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.gpu_server_url}/wssed/train",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            # Update job status to TRAINING
            self.update_training_job_status(job_id, TrainingStatus.TRAINING)
            
            logger.info(f"Triggered training for job {job_id} on GPU server")
            return result.get("task_id", str(job_id))
            
        except Exception as e:
            logger.error(f"Failed to trigger training for job {job_id}: {e}")
            self.update_training_job_status(
                job_id,
                TrainingStatus.FAILED,
                error_message=str(e)
            )
            raise
    
    async def update_training_status(self, job_id: int) -> WSSEDTrainingJob:
        """Poll GPU server for training status update"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.gpu_server_url}/wssed/train/{job_id}/status"
                )
                response.raise_for_status()
                status_data = response.json()
            
            # Update job with latest status
            status_str = status_data.get("status", "PENDING")
            status_enum = TrainingStatus(status_str)
            
            return self.update_training_job_status(
                job_id,
                status_enum,
                model_path=status_data.get("model_path"),
                metrics=status_data.get("metrics"),
                error_message=status_data.get("error")
            )
            
        except Exception as e:
            logger.error(f"Failed to update training status for job {job_id}: {e}")
            return self.get_training_job(job_id)
    
    # ============ DETECTION ============
    
    async def trigger_detection(self, job_id: int, threshold: float = 0.5) -> str:
        """
        Trigger detection on GPU server using trained model.
        
        Args:
            job_id: ID of the completed training job
            threshold: Detection threshold
        
        Returns:
            Task ID
        """
        job = self.get_training_job(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        if job.status != TrainingStatus.COMPLETED:
            raise ValueError(f"Training job {job_id} is not completed")
        
        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {job.dataset_id} not found")
        
        payload = {
            "job_id": job.id,
            "model_path": job.model_path,
            "dataset_path": dataset.source_uri,
            "threshold": threshold
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.gpu_server_url}/wssed/detect",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            logger.info(f"Triggered detection for job {job_id}")
            return result.get("task_id", str(job_id))
            
        except Exception as e:
            logger.error(f"Failed to trigger detection for job {job_id}: {e}")
            raise
    
    def store_predictions(
        self,
        job_id: int,
        predictions: List[Dict[str, Any]]
    ) -> int:
        """
        Store predictions from detection in database.
        
        Args:
            job_id: Training job ID
            predictions: List of prediction dicts with keys:
                - recording_id
                - species_name
                - start_time
                - end_time
                - confidence
                - frame_probabilities (optional)
        
        Returns:
            Number of predictions stored
        """
        stored_count = 0
        
        for pred_data in predictions:
            prediction = WSSEDPrediction(
                training_job_id=job_id,
                recording_id=pred_data["recording_id"],
                species_name=pred_data["species_name"],
                start_time=pred_data["start_time"],
                end_time=pred_data["end_time"],
                confidence=pred_data["confidence"],
                frame_probabilities=pred_data.get("frame_probabilities")
            )
            
            self.db.add(prediction)
            stored_count += 1
        
        self.db.commit()
        logger.info(f"Stored {stored_count} predictions for job {job_id}")
        
        return stored_count
    
    # ============ PREDICTIONS ============
    
    def list_predictions(
        self,
        recording_id: Optional[int] = None,
        training_job_id: Optional[int] = None,
        species_name: Optional[str] = None,
        threshold: float = 0.0,
        uncertain_range: Optional[float] = None,
        feedback_filter: Optional[FeedbackType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WSSEDPrediction]:
        """
        List predictions with filtering.
        
        Args:
            uncertain_range: If provided, filters predictions where confidence is within
                           this range of 0.5. For example, uncertain_range=0.1 will show
                           predictions where 0.4 <= confidence <= 0.6 (uncertain predictions).
        """
        query = self.db.query(WSSEDPrediction)
        
        if recording_id is not None:
            query = query.filter(WSSEDPrediction.recording_id == recording_id)
        
        if training_job_id is not None:
            query = query.filter(WSSEDPrediction.training_job_id == training_job_id)
        
        if species_name:
            query = query.filter(WSSEDPrediction.species_name == species_name)
        
        # Filter for uncertain predictions (close to 0.5)
        if uncertain_range is not None and uncertain_range > 0:
            lower_bound = 0.5 - uncertain_range
            upper_bound = 0.5 + uncertain_range
            query = query.filter(
                and_(
                    WSSEDPrediction.confidence >= lower_bound,
                    WSSEDPrediction.confidence <= upper_bound
                )
            )
        elif threshold > 0:
            # Only apply threshold filter if uncertain_range is not specified
            query = query.filter(WSSEDPrediction.confidence >= threshold)
        
        if feedback_filter is not None:
            query = query.filter(WSSEDPrediction.user_feedback == feedback_filter)
        
        return query.order_by(
            WSSEDPrediction.recording_id,
            WSSEDPrediction.start_time
        ).offset(skip).limit(limit).all()
    
    def get_recording_timeline(
        self,
        recording_id: int,
        training_job_id: Optional[int] = None,
        threshold: float = 0.5,
        uncertain_range: Optional[float] = None
    ) -> RecordingTimeline:
        """
        Get all predictions for a recording formatted for timeline visualization.
        
        Args:
            recording_id: Recording ID
            training_job_id: Optional filter by training job
            threshold: Minimum confidence threshold (ignored if uncertain_range is provided)
            uncertain_range: If provided, filters predictions where confidence is within
                           this range of 0.5. For example, uncertain_range=0.1 will show
                           predictions where 0.4 <= confidence <= 0.6 (uncertain predictions).
        
        Returns:
            RecordingTimeline object
        """
        recording = self.db.query(Recording).filter(
            Recording.id == recording_id
        ).first()
        
        if not recording:
            raise ValueError(f"Recording {recording_id} not found")
        
        query = self.db.query(WSSEDPrediction).filter(
            WSSEDPrediction.recording_id == recording_id
        )
        
        # Filter for uncertain predictions (close to 0.5)
        if uncertain_range is not None and uncertain_range > 0:
            lower_bound = 0.5 - uncertain_range
            upper_bound = 0.5 + uncertain_range
            query = query.filter(
                and_(
                    WSSEDPrediction.confidence >= lower_bound,
                    WSSEDPrediction.confidence <= upper_bound
                )
            )
        else:
            # Only apply threshold filter if uncertain_range is not specified
            query = query.filter(WSSEDPrediction.confidence >= threshold)
        
        if training_job_id:
            query = query.filter(WSSEDPrediction.training_job_id == training_job_id)
        
        predictions = query.order_by(WSSEDPrediction.start_time).all()
        
        timeline_predictions = [
            TimelinePrediction(
                prediction_id=pred.id,
                species=pred.species_name,
                start=pred.start_time,
                end=pred.end_time,
                confidence=pred.confidence,
                feedback=pred.user_feedback.value if pred.user_feedback else None
            )
            for pred in predictions
        ]
        
        return RecordingTimeline(
            recording_id=recording.id,
            file_name=recording.file_name,
            duration=recording.duration or 0.0,
            predictions=timeline_predictions
        )
    
    # ============ FEEDBACK ============
    
    def submit_feedback(
        self,
        prediction_id: int,
        feedback: FeedbackType
    ) -> Dict[str, Any]:
        """
        Submit feedback on a prediction.
        
        Args:
            prediction_id: Prediction ID
            feedback: ACCEPTED (creates strong_positive label) or REJECTED (creates strong_negative label)
        
        Returns:
            Dict with feedback_count and retraining_triggered flag
        """
        prediction = self.db.query(WSSEDPrediction).filter(
            WSSEDPrediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        # Update prediction feedback
        prediction.user_feedback = feedback
        prediction.feedback_at = datetime.utcnow()
        
        # Create strong label for both accepted (present) and rejected (absent)
        if feedback == FeedbackType.ACCEPTED:
            self._create_strong_label(prediction, LabelType.STRONG_POSITIVE)
        elif feedback == FeedbackType.REJECTED:
            self._create_strong_label(prediction, LabelType.STRONG_NEGATIVE)
        
        self.db.commit()
        
        # Count feedback since last training
        feedback_count = self._count_feedback_since_training(prediction.training_job_id)
        
        # Check if should trigger retraining (threshold: 5 feedbacks)
        retraining_triggered = False
        if feedback_count >= 5:
            try:
                self.trigger_retraining(prediction.training_job_id)
                retraining_triggered = True
                logger.info(f"Auto-triggered retraining after {feedback_count} feedbacks")
            except Exception as e:
                logger.error(f"Failed to auto-trigger retraining: {e}")
        
        return {
            "feedback_count": feedback_count,
            "retraining_triggered": retraining_triggered
        }
    
    def _create_strong_label(self, prediction: WSSEDPrediction, label_type: LabelType) -> WSSEDStrongLabel:
        """
        Create strong label from prediction feedback.
        
        Args:
            prediction: The prediction to create a label from
            label_type: STRONG_POSITIVE for accepted (present) or STRONG_NEGATIVE for rejected (absent)
        """
        # Check if already exists
        existing = self.db.query(WSSEDStrongLabel).filter(
            WSSEDStrongLabel.prediction_id == prediction.id
        ).first()
        
        if existing:
            # Update existing label type if it changed
            if existing.label_type != label_type:
                existing.label_type = label_type
                self.db.commit()
            return existing
        
        strong_label = WSSEDStrongLabel(
            prediction_id=prediction.id,
            recording_id=prediction.recording_id,
            species_name=prediction.species_name,
            start_time=prediction.start_time,
            end_time=prediction.end_time,
            confidence=prediction.confidence,
            label_type=label_type
        )
        
        self.db.add(strong_label)
        self.db.commit()
        
        logger.info(f"Created {label_type.value} strong label from prediction {prediction.id}")
        return strong_label
    
    def _count_feedback_since_training(self, job_id: int) -> int:
        """Count feedbacks submitted since this training job was created"""
        job = self.get_training_job(job_id)
        if not job:
            return 0
        
        count = self.db.query(func.count(WSSEDPrediction.id)).filter(
            WSSEDPrediction.training_job_id == job_id,
            WSSEDPrediction.user_feedback.isnot(None),
            WSSEDPrediction.feedback_at >= job.created_at
        ).scalar()
        
        return count or 0
    
    def get_feedback_stats(self, job_id: int) -> FeedbackStats:
        """Get feedback statistics for a training job"""
        total = self.db.query(func.count(WSSEDPrediction.id)).filter(
            WSSEDPrediction.training_job_id == job_id
        ).scalar() or 0
        
        accepted = self.db.query(func.count(WSSEDPrediction.id)).filter(
            WSSEDPrediction.training_job_id == job_id,
            WSSEDPrediction.user_feedback == FeedbackType.ACCEPTED
        ).scalar() or 0
        
        rejected = self.db.query(func.count(WSSEDPrediction.id)).filter(
            WSSEDPrediction.training_job_id == job_id,
            WSSEDPrediction.user_feedback == FeedbackType.REJECTED
        ).scalar() or 0
        
        pending = total - accepted - rejected
        
        feedback_since = self._count_feedback_since_training(job_id)
        
        return FeedbackStats(
            training_job_id=job_id,
            total_predictions=total,
            accepted_count=accepted,
            rejected_count=rejected,
            pending_count=pending,
            feedback_since_last_training=feedback_since
        )
    
    # ============ RETRAINING ============
    
    def trigger_retraining(self, original_job_id: int) -> WSSEDTrainingJob:
        """
        Trigger retraining with accumulated feedback.
        
        Args:
            original_job_id: ID of the original training job
        
        Returns:
            New training job
        """
        original_job = self.get_training_job(original_job_id)
        if not original_job:
            raise ValueError(f"Training job {original_job_id} not found")
        
        # Get feedback labels
        feedback_labels = self._get_feedback_labels(original_job_id)
        
        # Create new training job with same hyperparameters
        new_job = WSSEDTrainingJob(
            dataset_id=original_job.dataset_id,
            model_name=original_job.model_name,
            hyperparameters=original_job.hyperparameters,
            status=TrainingStatus.PENDING
        )
        
        self.db.add(new_job)
        self.db.commit()
        self.db.refresh(new_job)
        
        logger.info(
            f"Created retraining job {new_job.id} from original job {original_job_id} "
            f"with {len(feedback_labels)} feedback labels"
        )
        
        # TODO: Trigger training with feedback labels on GPU server
        
        return new_job
    
    def _get_feedback_labels(self, job_id: int) -> List[Dict[str, Any]]:
        """Get all feedback labels for retraining (both positive and negative)"""
        strong_labels = self.db.query(WSSEDStrongLabel).join(
            WSSEDPrediction
        ).filter(
            WSSEDPrediction.training_job_id == job_id,
            WSSEDPrediction.user_feedback.in_([FeedbackType.ACCEPTED, FeedbackType.REJECTED])
        ).all()
        
        return [
            {
                "recording_id": label.recording_id,
                "species_name": label.species_name,
                "start_time": label.start_time,
                "end_time": label.end_time,
                "label_type": label.label_type.value
            }
            for label in strong_labels
        ]
    
    # ============ SPECIES ============
    
    def get_dataset_species(self, dataset_id: int) -> SpeciesList:
        """Get list of species in dataset"""
        species_list = get_dataset_species_list(dataset_id, self.db)
        
        return SpeciesList(
            dataset_id=dataset_id,
            species=species_list,
            count=len(species_list)
        )
