"""
Annotation endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.schemas.annotation import Annotation, AnnotationCreate, AnnotationBatchCreate
from app.models.annotation import Annotation as AnnotationModel
from app.models.snippet import Snippet
from app.models.recording import Recording
from app.models.user import User
from app.core import taxonomy

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
        # Validate and resolve taxon_id
        resolved = taxonomy.resolve_taxon_id(taxon_id)
        if not resolved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or unresolvable taxon_id: {taxon_id}"
            )
    
    # Create annotation with resolved name snapshot
    annotation_data = annotation_in.model_dump(exclude={'species_name'})
    annotation_data['taxon_id'] = taxon_id
    annotation_data['resolved_name_snapshot'] = resolved.get('canonical_name') or resolved.get('scientific_name')
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
    in the same audio snippet.
    """
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
            # Validate and resolve taxon_id
            resolved = taxonomy.resolve_taxon_id(taxon_id)
            if not resolved:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid or unresolvable taxon_id: {taxon_id}"
                )
        
        # Create annotation with resolved name snapshot
        annotation_data = annotation_in.model_dump(exclude={'species_name'})
        annotation_data['taxon_id'] = taxon_id
        annotation_data['snippet_id'] = batch_in.snippet_id
        annotation_data['resolved_name_snapshot'] = resolved.get('canonical_name') or resolved.get('scientific_name')
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

