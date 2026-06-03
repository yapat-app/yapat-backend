"""
Feed endpoints for snippet retrieval

Supports feed generation methods:
- default: Prioritizes unannotated snippets
- random: Random sampling
- similarity: Similarity search
- similarity with uploaded audio: Find similar snippets using uploaded audio file
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, Query, HTTPException, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.user_feed import UserFeed
from app.schemas.snippet import Snippet, UserFeedSnapshot
from app.services.snippet_service import SnippetService

router = APIRouter()


def _save_feed_snapshot(
    db: Session,
    user_id: int,
    method: str,
    snippets: List[Snippet],
    request_params: Dict[str, Any],
) -> None:
    """
    Save a feed snapshot and enforce last 5 feeds per user+method.
    Silently fails if persistence fails to not block feed generation.
    """
    try:
        snippet_payload = jsonable_encoder(
            [Snippet.from_orm(s) for s in snippets]
        )
        
        feed_snapshot = UserFeed(
            user_id=user_id,
            method=method,
            request_params=request_params,
            response=snippet_payload,
        )
        db.add(feed_snapshot)
        db.flush()
        
        # Enforce last 5 feeds per user+method
        subquery = (
            db.query(UserFeed.id)
            .filter(UserFeed.user_id == user_id, UserFeed.method == method)
            .order_by(UserFeed.created_at.desc(), UserFeed.id.desc())
            .offset(5)
            .all()
        )
        ids_to_delete = [row[0] for row in subquery]
        if ids_to_delete:
            db.query(UserFeed).filter(UserFeed.id.in_(ids_to_delete)).delete(
                synchronize_session=False
            )
        
        db.commit()
    except Exception:
        db.rollback()


@router.get("/", response_model=List[Snippet])
def get_feed(
        method: Optional[str] = Query(
            default=None,
            description="Feed generation method: 'random' or 'similarity'. Default prioritizes unannotated snippets."
        ),
        dataset_id: Optional[int] = Query(default=None, description="Dataset ID to filter snippets"),
        snippet_set_id: Optional[int] = Query(default=None, description="SnippetSet ID (defaults to dataset's default if not specified)"),
        recording_id: Optional[int] = Query(default=None, description="Recording ID to filter snippets"),
        skip: int = Query(default=0, ge=0, description="Number of snippets to skip (pagination)"),
        limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of snippets to return"),
        # Method-specific parameters
        status: Optional[str] = Query(default=None, description="Filter by snippet status (for 'random' method)"),
        annotation_status: Optional[str] = Query(default="any", description="For 'filter' method: 'any' | 'annotated' | 'unannotated'"),
        embedding_model_id: Optional[int] = Query(default=None, description="Embedding model ID (for 'similarity' method)"),
        query_snippet_id: Optional[int] = Query(default=None, description="Snippet ID to use as query (for 'similarity' method)"),
        crop_start_sec: Optional[float] = Query(default=None, description="Crop start time in seconds (for 'similarity' method)"),
        crop_end_sec: Optional[float] = Query(default=None, description="Crop end time in seconds (for 'similarity' method)"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
):
    """
    Get feed of snippets for annotation using various sampling methods.
    
    If no method is specified, defaults to prioritizing unannotated snippets.
    Supported methods: 'random', 'similarity'
    
    If snippet_set_id is not provided, uses the dataset's default SnippetSet.
    Only READY SnippetSets are allowed; PENDING sets are rejected.
    """
    snippet_service = SnippetService(db)
    
    try:
        # Route to appropriate method based on 'method' parameter
        if method is None or method == "":
            # Default: prioritize unannotated snippets
            snippets = snippet_service.get_feed(
                dataset_id=dataset_id,
                snippet_set_id=snippet_set_id,
                recording_id=recording_id,
                skip=skip,
                limit=limit
            )
        elif method == "random":
            snippets = snippet_service.get_feed_random(
                dataset_id=dataset_id,
                snippet_set_id=snippet_set_id,
                recording_id=recording_id,
                status=status,
                skip=skip,
                limit=limit
            )
        elif method == "similarity":
            if dataset_id is None:
                raise HTTPException(
                    status_code=400, 
                    detail="dataset_id is required for 'similarity' method"
                )
            if query_snippet_id is None:
                raise HTTPException(
                    status_code=400, 
                    detail="query_snippet_id is required for 'similarity' method"
                )
            
            # Validate crop parameters if provided
            if crop_start_sec is not None and crop_end_sec is not None:
                if crop_start_sec >= crop_end_sec:
                    raise HTTPException(
                        status_code=400,
                        detail="crop_start_sec must be less than crop_end_sec"
                    )
                if crop_start_sec < 0:
                    raise HTTPException(
                        status_code=400,
                        detail="crop_start_sec must be non-negative"
                    )
            
            snippets = snippet_service.get_feed_similarity(
                dataset_id=dataset_id,
                snippet_set_id=snippet_set_id,
                query_snippet_id=query_snippet_id,
                embedding_model_id=embedding_model_id,
                crop_start_sec=crop_start_sec,
                crop_end_sec=crop_end_sec,
                skip=skip,
                limit=limit
            )
        elif method == "filter":
            snippets = snippet_service.get_feed_filter(
                dataset_id=dataset_id,
                snippet_set_id=snippet_set_id,
                recording_id=recording_id,
                annotation_status=annotation_status,
                skip=skip,
                limit=limit,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown feed method: '{method}'. Supported methods: 'random', 'similarity', 'filter'"
            )
        
        # Persist feed snapshot for the user (last 5 per user+method)
        if method in ("random", "similarity", "filter"):
            request_params: Dict[str, Any] = {
                "method": method,
                "dataset_id": dataset_id,
                "snippet_set_id": snippet_set_id,
                "recording_id": recording_id,
                "skip": skip,
                "limit": limit,
                "status": status,
                "annotation_status": annotation_status,
                "embedding_model_id": embedding_model_id,
                "query_snippet_id": query_snippet_id,
                "crop_start_sec": crop_start_sec,
                "crop_end_sec": crop_end_sec,
            }
            _save_feed_snapshot(db, current_user.id, method, snippets, request_params)

        return snippets
    except HTTPException:
        # Intended HTTP errors (400/404/409/...) must propagate as-is, not be
        # re-wrapped into a 500 by the broad handler below.
        raise
    except ValueError as e:
        # Convert service layer errors to appropriate HTTP exceptions
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        elif "not ready" in error_msg.lower() or "pending" in error_msg.lower():
            raise HTTPException(
                status_code=409,  # Conflict - resource not in correct state
                detail=error_msg
            )
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during feed generation: {str(e)}"
        )

def _snapshot_to_response_model(snap: UserFeed) -> UserFeedSnapshot:
    snippets = [Snippet.model_validate(item) for item in (snap.response or [])]
    return UserFeedSnapshot(
        id=snap.id,
        method=snap.method,
        created_at=snap.created_at,
        response=snippets,
        request_params=snap.request_params,
    )


def _dataset_id_from_request_params(rp: Optional[Dict[str, Any]]) -> Optional[int]:
    if not rp or not isinstance(rp, dict):
        return None
    v = rp.get("dataset_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_CLASSIC_FEED_METHODS = ("random", "similarity")


@router.get("/history", response_model=List[UserFeedSnapshot])
def get_feed_history(
    method: Optional[str] = Query(
        default=None,
        description="If set, only snapshots with this method (e.g. random, similarity).",
    ),
    dataset_id: Optional[int] = Query(
        default=None,
        description="If set, only snapshots for this dataset_id (from stored request_params).",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return stored classic feed snapshots for the user.

    Retention on write is last 5 rows per user **and** method. This endpoint returns up to
    5 **per** random and 5 **per** similarity (newest first when merged), not a single
    global cap of 5 across methods.

    Optional filters narrow results for clients that only need one mode and/or dataset.
    """
    if method:
        methods = (method,)
    else:
        methods = _CLASSIC_FEED_METHODS

    rows: List[UserFeed] = []
    for m in methods:
        part = (
            db.query(UserFeed)
            .filter(UserFeed.user_id == current_user.id, UserFeed.method == m)
            .order_by(UserFeed.created_at.desc(), UserFeed.id.desc())
            .limit(5)
            .all()
        )
        rows.extend(part)

    rows.sort(key=lambda s: (s.created_at, s.id), reverse=True)

    if dataset_id is not None:
        want = int(dataset_id)
        rows = [s for s in rows if _dataset_id_from_request_params(s.request_params) == want]

    return [_snapshot_to_response_model(s) for s in rows]


@router.post("/similarity-search", response_model=List[Snippet])
async def search_by_audio_upload(
        audio_file: UploadFile = File(..., description="Audio file to search with"),
        start_time: float = Form(..., description="Start time in seconds for the audio snippet"),
        end_time: float = Form(..., description="End time in seconds for the audio snippet"),
        dataset_id: int = Form(..., description="Dataset ID to search within"),
        embedding_model_id: int = Form(default=1, description="Embedding model ID to use"),
        snippet_set_id: Optional[int] = Form(default=None, description="Specific snippet set to search in (leave empty to use dataset's default)"),
        limit: int = Form(default=10, ge=1, le=100, description="Maximum number of similar snippets to return"),
        skip: int = Form(default=0, ge=0, description="Number of snippets to skip for pagination"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
):
    """
    Find similar snippets by uploading an audio file and selecting a time region.
    
    This endpoint allows users to:
    1. Upload an audio file (WAV, MP3, FLAC, etc.)
    2. Specify a time range [start_time, end_time] within that file
    3. Search for acoustically similar snippets in the dataset
    
    The audio snippet is embedded on-the-fly and compared against all snippets in the dataset.
    The uploaded file is processed in memory and not permanently stored.
    
    **Requirements**:
    - Audio duration (end_time - start_time) should match the embedding model's window size
    - For BirdNET: typically 3 seconds
    - File format: Any format supported by librosa (WAV, MP3, FLAC, OGG, etc.)
    
    """
    snippet_service = SnippetService(db)
    temp_file_path = None
    
    try:
        # Validate time parameters
        if start_time < 0:
            raise HTTPException(
                status_code=400,
                detail="start_time must be non-negative"
            )
        
        if end_time <= start_time:
            raise HTTPException(
                status_code=400,
                detail="end_time must be greater than start_time"
            )
        
        # Validate audio file
        if not audio_file.filename:
            raise HTTPException(
                status_code=400,
                detail="No audio file provided"
            )
        
        # Check file extension
        allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        audio_file.file.seek(0, 2)  # Seek to end
        file_size = audio_file.file.tell()
        audio_file.file.seek(0)  # Reset to beginning
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum size: 100MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            prefix="yapat_upload_"
        ) as temp_file:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await audio_file.read(chunk_size):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        # Generate embedding for the uploaded audio snippet
        try:
            from app.services.birdnet_model import BirdNetEmbedder

            # Use BirdNET embedder to generate embedding
            # Note: This assumes BirdNET model. For other models, add model selection logic.
            query_vector = BirdNetEmbedder.embed(
                audio_path=temp_file_path,
                start_time=start_time
            )
            
            if query_vector is None:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to generate embedding from audio snippet. "
                           "Please check that the audio region contains valid audio data."
                )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embedding: {str(e)}"
            )
        
        # Perform similarity search with the generated embedding
        try:
            # Convert snippet_set_id=0 to None (IDs start from 1, so 0 means "not specified")
            resolved_snippet_set_id = None if snippet_set_id == 0 else snippet_set_id
            
            similar_snippets = snippet_service.get_feed_similarity(
                dataset_id=dataset_id,
                snippet_set_id=resolved_snippet_set_id,
                query_embedding=query_vector,
                embedding_model_id=embedding_model_id,
                skip=skip,
                limit=limit
            )
            
            # Persist feed snapshot for similarity search (last 5 per user)
            request_params: Dict[str, Any] = {
                "method": "similarity",
                "dataset_id": dataset_id,
                "snippet_set_id": resolved_snippet_set_id,
                "embedding_model_id": embedding_model_id,
                "skip": skip,
                "limit": limit,
                "audio_upload": True,
            }
            _save_feed_snapshot(db, current_user.id, "similarity", similar_snippets, request_params)
            
            return similar_snippets
            
        except ValueError as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                raise HTTPException(status_code=404, detail=error_msg)
            elif "not ready" in error_msg.lower() or "pending" in error_msg.lower():
                raise HTTPException(status_code=409, detail=error_msg)
            else:
                raise HTTPException(status_code=400, detail=error_msg)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during similarity search: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                # Need to Log this in production, but don't fail the request
                pass
