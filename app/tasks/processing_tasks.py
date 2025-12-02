"""
Celery tasks for recording and snippet processing
"""

import os
from typing import List, Dict, Any
from celery import group

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet, SnippetConfig


@celery_app.task(bind=True, name="app.tasks.processing_tasks.process_recording")
def process_recording(self, recording_id: int):
    """
    Process a single recording: extract metadata and generate snippets
    
    Args:
        recording_id: ID of the recording to process
        
    Returns:
        dict: Processing results
    """
    db = SessionLocal()
    try:
        self.update_state(
            state='PROCESSING',
            meta={'recording_id': recording_id, 'stage': 'loading'}
        )
        
        # Get recording
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            return {
                "status": "error",
                "message": f"Recording {recording_id} not found"
            }
        
        # TODO: In actual implementation, this would:
        # 1. Load audio file
        # 2. Extract metadata (duration, sample_rate, channels, or whatever)
        # 3. Validate audio quality
        # 4. Update recording metadata
        
        self.update_state(
            state='PROCESSING',
            meta={'recording_id': recording_id, 'stage': 'metadata_extracted'}
        )
        
        # Trigger snippet generation
        snippet_task = generate_snippets_for_recording.delay(recording_id)
        
        return {
            "status": "success",
            "recording_id": recording_id,
            "snippet_task_id": snippet_task.id
        }
    except Exception as e:
        return {
            "status": "error",
            "recording_id": recording_id,
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.processing_tasks.generate_snippets_for_recording")
def generate_snippets_for_recording(
    self,
    recording_id: int,
    window_duration_sec: float = 3.0,
    hop_duration_sec: float = 1.5
):
    """
    Generate snippets from a recording using sliding window
    
    Args:
        recording_id: ID of the recording
        window_duration_sec: Duration of each snippet in seconds
        hop_duration_sec: Hop size between snippets in seconds
        
    Returns:
        dict: Results with snippet count
    """
    db = SessionLocal()
    try:
        self.update_state(
            state='PROCESSING',
            meta={'recording_id': recording_id, 'stage': 'generating_snippets'}
        )
        
        # Get recording
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            return {
                "status": "error",
                "message": f"Recording {recording_id} not found"
            }
        
        # TODO: In actual implementation, this would:
        # 1. Load audio file
        # 2. Apply sliding window segmentation
        # 3. Extract each snippet to file or keep reference
        # 4. Create Snippet database entries
        # 5. Optionally trigger embedding generation
        
        # Placeholder: Create dummy snippets based on duration
        snippets_created = []
        if recording.duration_sec:
            start_time = 0.0
            snippet_count = 0
            
            while start_time + window_duration_sec <= recording.duration_sec:
                snippet = Snippet(
                    recording_id=recording_id,
                    start_time=start_time,
                    end_time=start_time + window_duration_sec,
                    duration=window_duration_sec,
                    file_path=None,  # Will be set when actually extracted
                    is_annotated=False
                )
                db.add(snippet)
                snippets_created.append(snippet)
                
                start_time += hop_duration_sec
                snippet_count += 1
                
                # Update progress periodically
                if snippet_count % 10 == 0:
                    self.update_state(
                        state='PROCESSING',
                        meta={
                            'recording_id': recording_id,
                            'snippets_created': snippet_count
                        }
                    )
            
            db.commit()
            
            # Refresh to get IDs
            for snippet in snippets_created:
                db.refresh(snippet)
        
        return {
            "status": "success",
            "recording_id": recording_id,
            "snippets_created": len(snippets_created),
            "snippet_ids": [s.id for s in snippets_created]
        }
    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "recording_id": recording_id,
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.processing_tasks.scan_and_process_dataset")
def scan_and_process_dataset(self, dataset_id: int):
    """
    Scan a dataset directory and process all audio files
    
    Args:
        dataset_id: ID of the dataset to scan
        
    Returns:
        dict: Summary of scanning and processing
    """
    db = SessionLocal()
    try:
        self.update_state(
            state='PROCESSING',
            meta={'dataset_id': dataset_id, 'stage': 'scanning'}
        )
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {
                "status": "error",
                "message": f"Dataset {dataset_id} not found"
            }
        
        # TODO: In actual implementation, this would:
        # 1. Scan dataset.source_uri directory
        # 2. Find all audio files (wav, flac, mp3, etc.)
        # 3. Parse filenames for metadata (timestamps, station names)
        # 4. Create Recording entries for new files
        # 5. Trigger processing for each recording
        
        # Placeholder: Check if source_uri exists
        recordings_found = []
        if dataset.source_uri and os.path.exists(dataset.source_uri):
            # This would actually scan the directory
            # For now, just return success
            pass
        
        # Trigger processing for all recordings in dataset
        recordings = db.query(Recording).filter(
            Recording.dataset_id == dataset_id
        ).all()
        
        if recordings:
            # Create parallel processing tasks
            processing_tasks = group(
                process_recording.s(rec.id) for rec in recordings
            )
            result = processing_tasks.apply_async()
            
            return {
                "status": "started",
                "dataset_id": dataset_id,
                "recordings_found": len(recordings),
                "processing_task_ids": [str(r.id) for r in result.results]
            }
        else:
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "recordings_found": 0,
                "message": "No recordings found in dataset"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "message": str(e)
        }
    finally:
        db.close()
