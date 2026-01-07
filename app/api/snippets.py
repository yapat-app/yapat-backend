"""
Snippet endpoints (updated for SnippetSet-based architecture)
"""

from typing import List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.snippet import Snippet as SnippetModel
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.models.recording import Recording
from app.models.user import User
from app.schemas.snippet import Snippet
from app.services.audio_service import audio_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=List[Snippet])
def read_snippets(
    dataset_id: int,
    snippet_set_id: int,
    recording_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List snippets belonging to a dataset and a snippet_set.
    Optionally filter further by recording.
    """

    # Validate snippet_set belongs to dataset and is READY
    ss = (
        db.query(SnippetSet)
        .filter(
            SnippetSet.id == snippet_set_id,
            SnippetSet.dataset_id == dataset_id,
        )
        .first()
    )
    if not ss:
        raise HTTPException(404, detail="SnippetSet not found for this dataset")
    if ss.status != SnippetSetStatus.READY:
        raise HTTPException(
            400,
            detail=f"SnippetSet is not READY (status: {ss.status.value}). Only READY SnippetSets can be queried."
        )

    query = (
        db.query(SnippetModel)
        .join(SnippetModel.recording)
        .filter(SnippetModel.snippet_set_id == snippet_set_id)
    )

    if recording_id is not None:
        query = query.filter(SnippetModel.recording_id == recording_id)

    return (
        query.order_by(SnippetModel.start_time)
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{snippet_id}", response_model=Snippet)
def read_snippet(
    snippet_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single snippet by ID."""
    snippet = (
        db.query(SnippetModel)
        .filter(SnippetModel.id == snippet_id)
        .first()
    )
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    return snippet


@router.get("/{snippet_id}/audio")
def get_snippet_audio(
    snippet_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Extract and serve audio for a snippet.
    
    Implements caching:
    - First request: extracts from recording and caches
    - Subsequent requests: serves cached file (fast)
    
    Returns WAV file for browser playback.
    """
    # Fetch snippet with recording info
    snippet = (
        db.query(SnippetModel)
        .join(Recording)
        .filter(SnippetModel.id == snippet_id)
        .first()
    )
    
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    
    recording = snippet.recording
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found for snippet")
    
    try:
        # Extract or get cached audio
        audio_path, sample_rate, channels = audio_service.extract_snippet(
            recording_file_path=recording.file_path,
            start_time=snippet.start_time,
            end_time=snippet.end_time,
            snippet_id=snippet_id,
            use_cache=True,
        )
        
        # Serve the audio file for inline playback
        return FileResponse(
            path=str(audio_path),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="snippet_{snippet_id}.wav"',
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
                "Accept-Ranges": "bytes",  # Enable seeking in audio player
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": str(channels),
            }
        )
    
    except FileNotFoundError as e:
        logger.error(f"Recording file not found for snippet {snippet_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Recording file not found: {recording.file_path}"
        )
    
    except ValueError as e:
        logger.error(f"Invalid snippet parameters for {snippet_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error extracting audio for snippet {snippet_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to extract audio snippet"
        )


@router.get("/{snippet_id}/spectrogram")
def get_snippet_spectrogram(
    snippet_id: int,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: float = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Generate and serve mel spectrogram visualization for a snippet.
    
    Query parameters:
    - n_mels: Number of mel frequency bands (default: 128)
    - fmin: Minimum frequency in Hz (default: 0)
    - fmax: Maximum frequency in Hz (default: sample_rate/2)
    
    Returns PNG image of the spectrogram.
    Implements smart caching - subsequent requests serve cached image.
    """
    # Fetch snippet with recording info
    snippet = (
        db.query(SnippetModel)
        .join(Recording)
        .filter(SnippetModel.id == snippet_id)
        .first()
    )
    
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    
    recording = snippet.recording
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found for snippet")
    
    try:
        # First, extract or get cached audio
        audio_path, sample_rate, channels = audio_service.extract_snippet(
            recording_file_path=recording.file_path,
            start_time=snippet.start_time,
            end_time=snippet.end_time,
            snippet_id=snippet_id,
            use_cache=True,
        )
        
        # Read audio data for spectrogram generation
        import soundfile as sf
        audio_data, _ = sf.read(str(audio_path))
        
        # Handle stereo by converting to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Generate or get cached spectrogram
        spectrogram_path = audio_service.generate_spectrogram(
            audio_data=audio_data,
            sample_rate=sample_rate,
            snippet_id=snippet_id,
            use_cache=True,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        
        # Serve the spectrogram image
        return FileResponse(
            path=str(spectrogram_path),
            media_type="image/png",
            filename=f"spectrogram_{snippet_id}.png",
            headers={
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
                "X-Snippet-ID": str(snippet_id),
                "X-Sample-Rate": str(sample_rate),
            }
        )
    
    except ImportError as e:
        logger.error(f"Spectrogram dependencies not installed: {e}")
        raise HTTPException(
            status_code=501,
            detail="Spectrogram generation not available. librosa/matplotlib not installed."
        )
    
    except FileNotFoundError as e:
        logger.error(f"Recording file not found for snippet {snippet_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Recording file not found: {recording.file_path}"
        )
    
    except ValueError as e:
        logger.error(f"Invalid snippet parameters for {snippet_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error generating spectrogram for snippet {snippet_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate spectrogram"
        )
