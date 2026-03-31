"""
Annotation endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.schemas.annotation import (
    Annotation, AnnotationCreate, AnnotationBatchCreate, 
    DatasetAnnotationStats, AllDatasetsAnnotationStats
)
from app.models.annotation import Annotation as AnnotationModel
from app.models.snippet import Snippet
from app.models.recording import Recording
from app.models.dataset import Dataset
from app.models.user import User
from app.core import taxonomy
from sqlalchemy import func

router = APIRouter()


@router.post("/", response_model=Annotation, status_code=status.HTTP_201_CREATED)
def create_annotation(
    annotation_in: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new annotation.
    
    You can provide either:
    - `taxon_id`: A namespaced identifier (e.g., 'gbif:2420576')
    - `species_name`: A scientific or common name (e.g., 'Turdus merula' or 'Common Blackbird')
    
    If `species_name` is provided, it will be automatically resolved to a taxon_id via the taxonomy service.
    The scientific name is automatically resolved and snapshotted.
    """
    taxon_id = annotation_in.taxon_id
    resolved = None
    
    # If species_name is provided, resolve it to taxon_id
    if annotation_in.species_name:
        matched = taxonomy.match_species_name(annotation_in.species_name)
        if not matched:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not resolve species name: {annotation_in.species_name}. Please check the spelling or provide a taxon_id instead."
            )
        taxon_id = matched['taxon_id']
        resolved = matched
    else:
        # Validate and resolve taxon_id (pass db session for custom taxonomies)
        resolved = taxonomy.resolve_taxon_id(taxon_id, db_session=db)
        if not resolved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or unresolvable taxon_id: {taxon_id}"
            )
        
        # Check access for custom taxonomies
        if resolved.get("taxonomy_type") == "custom":
            from app.core.permissions import check_team_member, check_admin
            from app.services.custom_taxonomy_service import get_taxonomy_by_id
            from app.models.custom_taxonomy import TaxonomyStatus
            
            custom_taxonomy = get_taxonomy_by_id(taxon_id, db)
            if not custom_taxonomy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Custom taxonomy {taxon_id} not found"
                )
            
            # Check if taxonomy is active
            if custom_taxonomy.status != TaxonomyStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Taxonomy {taxon_id} is not active and cannot be used for annotation"
                )
            
            # Check if user has access: must be team member or global taxonomy
            if not custom_taxonomy.is_global:
                if not check_team_member(current_user, custom_taxonomy.team_id, db) and not check_admin(current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"You don't have access to use taxonomy {taxon_id}"
                    )
    
    # Reject duplicate: same snippet already has this taxon_id
    existing = (
        db.query(AnnotationModel)
        .filter(
            AnnotationModel.snippet_id == annotation_in.snippet_id,
            AnnotationModel.taxon_id == taxon_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This snippet already has an annotation for this taxon ({taxon_id}). Duplicate annotations are not allowed.",
        )

    # Create annotation with resolved name snapshot
    annotation_data = annotation_in.model_dump(exclude={'species_name', 'display_name'})
    annotation_data['taxon_id'] = taxon_id
    # Prefer client-provided display_name (e.g. from Taxonomy Assistant) for wiki/envo/ols
    snapshot = (
        annotation_in.display_name
        or resolved.get('canonical_name')
        or resolved.get('scientific_name')
    )
    annotation_data['resolved_name_snapshot'] = snapshot or taxon_id
    annotation_data['user_id'] = current_user.id
    
    annotation = AnnotationModel(**annotation_data)
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


@router.post("/batch", response_model=List[Annotation], status_code=status.HTTP_201_CREATED)
def create_annotations_batch(
    batch_in: AnnotationBatchCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create multiple annotations for a single snippet.

    Useful for multi-label annotations where multiple taxa are present
    in the same audio snippet. Duplicate taxon_id for the same snippet are not allowed.
    """
    snippet_id = batch_in.snippet_id
    # Existing taxon_ids on this snippet (no duplicates allowed)
    existing_taxon_ids = {
        a.taxon_id
        for a in db.query(AnnotationModel).filter(
            AnnotationModel.snippet_id == snippet_id
        ).all()
    }
    seen_in_batch = set()
    created_annotations = []

    for annotation_in in batch_in.annotations:
        taxon_id = annotation_in.taxon_id
        resolved = None

        # If species_name is provided, resolve it to taxon_id
        if annotation_in.species_name:
            matched = taxonomy.match_species_name(annotation_in.species_name)
            if not matched:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not resolve species name: {annotation_in.species_name}. Please check the spelling or provide a taxon_id instead."
                )
            taxon_id = matched['taxon_id']
            resolved = matched
        else:
            # Validate and resolve taxon_id (pass db session for custom taxonomies)
            resolved = taxonomy.resolve_taxon_id(taxon_id, db_session=db)
            if not resolved:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid or unresolvable taxon_id: {taxon_id}"
                )
            
            # Check access for custom taxonomies
            if resolved.get("taxonomy_type") == "custom":
                from app.core.permissions import check_team_member, check_admin
                from app.services.custom_taxonomy_service import get_taxonomy_by_id
                from app.models.custom_taxonomy import TaxonomyStatus
                
                custom_taxonomy = get_taxonomy_by_id(taxon_id, db)
                if not custom_taxonomy:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Custom taxonomy {taxon_id} not found"
                    )
                
                # Check if taxonomy is active
                if custom_taxonomy.status != TaxonomyStatus.ACTIVE:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Taxonomy {taxon_id} is not active and cannot be used for annotation"
                    )
                
                # Check if user has access: must be team member or global taxonomy
                if not custom_taxonomy.is_global:
                    if not check_team_member(current_user, custom_taxonomy.team_id, db) and not check_admin(current_user):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"You don't have access to use taxonomy {taxon_id}"
                        )

        # Reject duplicate: already on snippet or repeated in this batch
        if taxon_id in existing_taxon_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"This snippet already has an annotation for this taxon ({taxon_id}). Duplicate annotations are not allowed.",
            )
        if taxon_id in seen_in_batch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Duplicate taxon in request: {taxon_id} appears more than once.",
            )
        seen_in_batch.add(taxon_id)

        # Create annotation with resolved name snapshot
        annotation_data = annotation_in.model_dump(exclude={'species_name', 'display_name'})
        annotation_data['taxon_id'] = taxon_id
        annotation_data['snippet_id'] = batch_in.snippet_id
        snapshot = (
            annotation_in.display_name
            or resolved.get('canonical_name')
            or resolved.get('scientific_name')
        )
        annotation_data['resolved_name_snapshot'] = snapshot or taxon_id
        annotation_data['user_id'] = current_user.id
        
        annotation = AnnotationModel(**annotation_data)
        db.add(annotation)
        created_annotations.append(annotation)
    
    db.commit()
    for annotation in created_annotations:
        db.refresh(annotation)
    
    return created_annotations


@router.get("/", response_model=List[Annotation])
def read_annotations(
    snippet_id: Optional[int] = Query(None, description="Filter by snippet ID"),
    taxon_id: Optional[str] = Query(None, description="Filter by taxon ID"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of annotations with optional filtering.
    
    Supports filtering by snippet_id, taxon_id, user_id, and dataset_id.
    """
    query = db.query(AnnotationModel)
    
    if snippet_id:
        query = query.filter(AnnotationModel.snippet_id == snippet_id)
    if taxon_id:
        query = query.filter(AnnotationModel.taxon_id == taxon_id)
    if user_id:
        query = query.filter(AnnotationModel.user_id == user_id)
    if dataset_id:
        # Join through Snippet -> Recording -> Dataset to filter by dataset_id
        query = query.join(Snippet).join(Recording).filter(Recording.dataset_id == dataset_id)
    
    annotations = query.offset(skip).limit(limit).all()
    return annotations


@router.get("/{annotation_id}", response_model=Annotation)
def read_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific annotation by ID"""
    annotation = db.query(AnnotationModel).filter(AnnotationModel.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return annotation


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete an annotation.
    
    Users can only delete their own annotations.
    Team owners can delete any annotation in their teams (future enhancement).
    """
    annotation = db.query(AnnotationModel).filter(AnnotationModel.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Check ownership - users can only delete their own annotations
    if annotation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own annotations"
        )
    
    db.delete(annotation)
    db.commit()
    return None


@router.get("/datasets/stats", response_model=AllDatasetsAnnotationStats)
def get_all_datasets_annotation_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get annotation statistics for all datasets.
    
    Returns a list of annotation statistics for each dataset including:
    - Total number of snippets in each dataset
    - Number of annotated snippets (with at least one annotation)
    - Number of not annotated snippets (with zero annotations)
    - Annotation percentage
    - Total number of annotations
    """
    # Get all datasets
    datasets = db.query(Dataset).all()
    
    dataset_stats_list = []
    
    for dataset in datasets:
        dataset_id = dataset.id
        
        # Count total snippets for this dataset
        total_snippets = (
            db.query(func.count(Snippet.id))
            .join(Recording)
            .filter(Recording.dataset_id == dataset_id)
            .scalar()
        ) or 0
        
        # Count snippets with at least one annotation
        annotated_snippets = (
            db.query(func.count(func.distinct(Snippet.id)))
            .join(Recording)
            .join(AnnotationModel, AnnotationModel.snippet_id == Snippet.id)
            .filter(Recording.dataset_id == dataset_id)
            .scalar()
        ) or 0
        
        # Count total annotations
        total_annotations = (
            db.query(func.count(AnnotationModel.id))
            .join(Snippet)
            .join(Recording)
            .filter(Recording.dataset_id == dataset_id)
            .scalar()
        ) or 0
        
        # Calculate not annotated snippets
        not_annotated_snippets = total_snippets - annotated_snippets
        
        # Calculate percentage
        annotation_percentage = (annotated_snippets / total_snippets * 100) if total_snippets > 0 else 0.0
        
        dataset_stats_list.append(
            DatasetAnnotationStats(
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                total_snippets=total_snippets,
                annotated_snippets=annotated_snippets,
                not_annotated_snippets=not_annotated_snippets,
                annotation_percentage=round(annotation_percentage, 2),
                total_annotations=total_annotations
            )
        )
    
    return AllDatasetsAnnotationStats(
        datasets=dataset_stats_list,
        total_datasets=len(datasets)
    )

