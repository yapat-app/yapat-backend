"""
Dataset endpoints
"""

from typing import List, Optional, Literal
from datetime import datetime
from io import StringIO
import csv

from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.api.deps import get_db, get_current_active_user
from app.models.user import User, UserRole
from app.models.annotation import Annotation as AnnotationModel
from app.models.snippet import Snippet
from app.models.recording import Recording
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.schemas.dataset import Dataset, DatasetCreate, DatasetCreationResponse
from app.schemas.annotation import AnnotationExport
from app.services.dataset_service import DatasetService
from app.tasks.processing_tasks import process_dataset

router = APIRouter()


@router.post("/", response_model=DatasetCreationResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
        dataset_in: DatasetCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)

    if current_user.role != UserRole.ADMIN and dataset_in.team_id is None:
        raise HTTPException(status_code=400, detail="team_id is required for non-admin users")

    try:
        dataset = svc.create_dataset(dataset_in, current_user)
    except ValueError as e:
        if str(e) == "duplicate_dataset":
            raise HTTPException(status_code=409, detail="Dataset already exists")
        if str(e) == "team_not_found":
            raise HTTPException(status_code=404, detail="Team not found")
        if str(e) == "invalid_source_uri":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid dataset path: {dataset_in.source_uri} does not exist or is not a directory"
            )
        raise

    # Dispatch background task for dataset processing (scanning + snippet generation)
    # Returns task ID for client tracking; None if task dispatch fails (backward compatible)
    try:
        task = process_dataset.delay(dataset.id)
        task_id = task.id
    except Exception:
        task_id = None

    # Create dataset response with recording_count and feed readiness
    dataset_dict = {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_uri": dataset.source_uri,
        "team_id": dataset.team_id,
        "default_snippet_set_id": dataset.default_snippet_set_id,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "recording_count": 0,  # New datasets start with 0 recordings
        "is_ready_for_feed": False,  # New datasets don't have snippet sets yet
    }
    dataset_response = Dataset(**dataset_dict)

    return DatasetCreationResponse(
        dataset=dataset_response,
        process_task_id=task_id,
        snippet_config_id=None,
        embedding_job_id=None,
    )


@router.get("/", response_model=List[Dataset])
def read_datasets(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    datasets = svc.list_datasets(current_user=current_user, skip=skip, limit=limit)
    
    # Add recording count for each dataset
    dataset_ids = [ds.id for ds in datasets]
    if dataset_ids:
        recording_counts = (
            db.query(Recording.dataset_id, func.count(Recording.id).label('count'))
            .filter(Recording.dataset_id.in_(dataset_ids))
            .group_by(Recording.dataset_id)
            .all()
        )
        count_map = {ds_id: count for ds_id, count in recording_counts}
    else:
        count_map = {}
    
    # Check feed readiness for datasets with default snippet sets
    snippet_set_ids = [ds.default_snippet_set_id for ds in datasets if ds.default_snippet_set_id]
    if snippet_set_ids:
        ready_snippet_sets = (
            db.query(SnippetSet.id)
            .filter(
                SnippetSet.id.in_(snippet_set_ids),
                SnippetSet.status == SnippetSetStatus.READY
            )
            .all()
        )
        ready_set_ids = {ss_id for (ss_id,) in ready_snippet_sets}
    else:
        ready_set_ids = set()
    
    # Convert to schema and add recording_count and feed readiness
    result = []
    for dataset in datasets:
        is_ready = (
            dataset.default_snippet_set_id is not None 
            and dataset.default_snippet_set_id in ready_set_ids
        )
        
        dataset_dict = {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "source_uri": dataset.source_uri,
            "team_id": dataset.team_id,
            "default_snippet_set_id": dataset.default_snippet_set_id,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "recording_count": count_map.get(dataset.id, 0),
            "is_ready_for_feed": is_ready,
        }
        result.append(Dataset(**dataset_dict))
    
    return result


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    dataset = svc.get_dataset_for_user(dataset_id, current_user)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Add recording count
    recording_count = (
        db.query(func.count(Recording.id))
        .filter(Recording.dataset_id == dataset_id)
        .scalar()
    ) or 0
    
    # Check if dataset is ready for feed generation
    is_ready_for_feed = False
    if dataset.default_snippet_set_id:
        snippet_set = (
            db.query(SnippetSet)
            .filter(SnippetSet.id == dataset.default_snippet_set_id)
            .first()
        )
        if snippet_set and snippet_set.status == SnippetSetStatus.READY:
            is_ready_for_feed = True
    
    dataset_dict = {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_uri": dataset.source_uri,
        "team_id": dataset.team_id,
        "default_snippet_set_id": dataset.default_snippet_set_id,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
        "recording_count": recording_count,
        "is_ready_for_feed": is_ready_for_feed,
    }
    return Dataset(**dataset_dict)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    dataset = svc.get_dataset_for_user(dataset_id, current_user)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not svc.can_delete_dataset(dataset, current_user):
        raise HTTPException(status_code=403, detail="Only team owners and admins can delete a dataset")

    svc.delete_dataset(dataset)
    return None


@router.get("/{dataset_id}/annotations/export")
def export_dataset_annotations(
        dataset_id: int,
        format: Literal["json", "csv"] = Query("json", description="Export format: json or csv"),
        taxon_id: Optional[str] = Query(None, description="Filter by taxon_id"),
        user_id: Optional[int] = Query(None, description="Filter by user_id (created_by)"),
        created_after: Optional[datetime] = Query(None, description="Filter annotations created after this datetime"),
        created_before: Optional[datetime] = Query(None, description="Filter annotations created before this datetime"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """
    Export all annotations for a dataset with recording and snippet metadata.
    
    Supports filtering by:
    - taxon_id: Filter by specific taxon
    - user_id: Filter by annotation creator
    - created_after: Filter annotations created after datetime
    - created_before: Filter annotations created before datetime
    
    Returns either JSON (default) or CSV format.
    """
    # Verify dataset exists and user has access
    svc = DatasetService(db)
    dataset = svc.get_dataset_for_user(dataset_id, current_user)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Build query with joins
    query = (
        db.query(
            AnnotationModel.id.label('annotation_id'),
            Recording.dataset_id.label('dataset_id'),
            AnnotationModel.snippet_id,
            AnnotationModel.taxon_id,
            AnnotationModel.resolved_name_snapshot,
            AnnotationModel.confidence,
            AnnotationModel.created_at,
            AnnotationModel.user_id.label('created_by'),
            Recording.file_name.label('recording_file_name'),
            Recording.file_path.label('recording_file_path'),
            Snippet.start_time.label('snippet_start_time'),
            Snippet.end_time.label('snippet_end_time'),
            Snippet.duration.label('snippet_duration'),
        )
        .join(Snippet, AnnotationModel.snippet_id == Snippet.id)
        .join(Recording, Snippet.recording_id == Recording.id)
        .filter(Recording.dataset_id == dataset_id)
    )
    
    # Apply filters
    if taxon_id:
        query = query.filter(AnnotationModel.taxon_id == taxon_id)
    if user_id:
        query = query.filter(AnnotationModel.user_id == user_id)
    if created_after:
        query = query.filter(AnnotationModel.created_at >= created_after)
    if created_before:
        query = query.filter(AnnotationModel.created_at <= created_before)
    
    # Execute query
    results = query.all()
    
    # Convert to dict for easy processing
    annotations_data = [
        {
            'annotation_id': row.annotation_id,
            'dataset_id': row.dataset_id,
            'snippet_id': row.snippet_id,
            'taxon_id': row.taxon_id,
            'resolved_name_snapshot': row.resolved_name_snapshot,
            'confidence': row.confidence,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'created_by': row.created_by,
            'recording_file_name': row.recording_file_name,
            'recording_file_path': row.recording_file_path,
            'snippet_start_time': row.snippet_start_time,
            'snippet_end_time': row.snippet_end_time,
            'snippet_duration': row.snippet_duration,
        }
        for row in results
    ]
    
    # Return based on format
    if format == "csv":
        # Generate CSV
        if not annotations_data:
            # Return empty CSV with headers
            csv_headers = [
                'annotation_id', 'dataset_id', 'snippet_id', 'taxon_id', 
                'resolved_name_snapshot', 'confidence', 'created_at', 'created_by',
                'recording_file_name', 'recording_file_path', 'snippet_start_time',
                'snippet_end_time', 'snippet_duration'
            ]
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=csv_headers)
            writer.writeheader()
            csv_content = output.getvalue()
        else:
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=annotations_data[0].keys())
            writer.writeheader()
            writer.writerows(annotations_data)
            csv_content = output.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=dataset_{dataset_id}_annotations.csv"
            }
        )
    else:
        # Return JSON (using Pydantic for validation)
        return [AnnotationExport(**data) for data in annotations_data]
