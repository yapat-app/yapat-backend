"""
Annotation endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
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
from app.models.pam_active_learning import ALSnippetAnnotation, ALAnnotationSource
from app.core import taxonomy
from sqlalchemy import func

_MAX_ANNOTATION_SNIPPET_IDS_FILTER = 400

router = APIRouter()


def _mirror_to_al(db: Session, annotation: AnnotationModel) -> None:
    """Mirror a canonical annotation into al_snippet_annotation (USER source)."""
    snippet = db.get(Snippet, annotation.snippet_id)
    if snippet is None or snippet.recording is None:
        return
    dataset_id = snippet.recording.dataset_id
    label = (annotation.resolved_name_snapshot or "").strip()
    if not label:
        return
    exists = (
        db.query(ALSnippetAnnotation)
        .filter(
            ALSnippetAnnotation.snippet_id == annotation.snippet_id,
            ALSnippetAnnotation.label == label,
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.user_id == annotation.user_id,
            ALSnippetAnnotation.model_checkpoint_id.is_(None),
        )
        .first()
    )
    if exists is None:
        db.add(ALSnippetAnnotation(
            dataset_id=dataset_id,
            snippet_id=annotation.snippet_id,
            label=label,
            source=ALAnnotationSource.USER,
            user_id=annotation.user_id,
            model_checkpoint_id=None,
        ))


def _unmirror_from_al(db: Session, annotation: AnnotationModel) -> None:
    """Remove the mirrored al_snippet_annotation row for a deleted annotation."""
    label = (annotation.resolved_name_snapshot or "").strip()
    if not label:
        return
    (
        db.query(ALSnippetAnnotation)
        .filter(
            ALSnippetAnnotation.snippet_id == annotation.snippet_id,
            ALSnippetAnnotation.label == label,
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.user_id == annotation.user_id,
            ALSnippetAnnotation.model_checkpoint_id.is_(None),
        )
        .delete(synchronize_session=False)
    )


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
    _mirror_to_al(db, annotation)
    db.commit()
    from app.schemas.annotation import Annotation as AnnotationSchema
    result = AnnotationSchema.model_validate(annotation)
    result.username = current_user.username
    return result


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
    for annotation in created_annotations:
        _mirror_to_al(db, annotation)
    db.commit()
    from app.schemas.annotation import Annotation as AnnotationSchema
    result = []
    for ann in created_annotations:
        obj = AnnotationSchema.model_validate(ann)
        obj.username = current_user.username
        result.append(obj)
    return result


@router.get("/", response_model=List[Annotation])
def read_annotations(
    snippet_id: Optional[int] = Query(None, description="Filter by snippet ID"),
    snippet_ids: Optional[str] = Query(
        None,
        description="Comma-separated snippet IDs (max 400). Batched load for annotate hub.",
    ),
    taxon_id: Optional[str] = Query(None, description="Filter by taxon ID"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of annotations with optional filtering.
    
    Supports filtering by snippet_id, snippet_ids (batch), taxon_id, user_id, and dataset_id.
    """
    parsed_id_list: List[int] = []
    if snippet_ids:
        for part in snippet_ids.split(","):
            p = part.strip()
            if p.isdigit():
                parsed_id_list.append(int(p))
        parsed_id_list = parsed_id_list[:_MAX_ANNOTATION_SNIPPET_IDS_FILTER]

    max_limit = 2000 if parsed_id_list else 500
    eff_limit = min(limit, max_limit)

    query = db.query(AnnotationModel).join(User, AnnotationModel.user_id == User.id)

    if parsed_id_list:
        query = query.filter(AnnotationModel.snippet_id.in_(parsed_id_list))
    elif snippet_id:
        query = query.filter(AnnotationModel.snippet_id == snippet_id)
    if taxon_id:
        query = query.filter(AnnotationModel.taxon_id == taxon_id)
    if user_id:
        query = query.filter(AnnotationModel.user_id == user_id)
    if dataset_id:
        # Join through Snippet -> Recording -> Dataset to filter by dataset_id
        query = query.join(Snippet, AnnotationModel.snippet_id == Snippet.id).join(Recording).filter(Recording.dataset_id == dataset_id)

    rows = query.offset(skip).limit(eff_limit).all()

    from app.schemas.annotation import Annotation as AnnotationSchema
    result = []
    for ann in rows:
        obj = AnnotationSchema.model_validate(ann)
        obj.username = ann.user.username if ann.user else None
        result.append(obj)
    return result


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
    from app.schemas.annotation import Annotation as AnnotationSchema
    result = AnnotationSchema.model_validate(annotation)
    result.username = annotation.user.username if annotation.user else None
    return result


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete an annotation.
    
    Team members can delete annotations made by other users within the same team.
    """
    annotation = db.query(AnnotationModel).filter(AnnotationModel.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    # Determine the owning team for the snippet's recording dataset.
    # If dataset has no team (admin-created personal dataset), only allow self-deletes or admins.
    snippet_row = db.query(Snippet).filter(Snippet.id == annotation.snippet_id).first()
    if not snippet_row:
        raise HTTPException(status_code=404, detail="Snippet not found")

    recording_row = db.query(Recording).filter(Recording.id == snippet_row.recording_id).first()
    if not recording_row:
        raise HTTPException(status_code=404, detail="Recording not found")

    dataset_row = db.query(Dataset).filter(Dataset.id == recording_row.dataset_id).first()
    if not dataset_row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from app.core.permissions import check_admin, check_team_member

    if dataset_row.team_id is None:
        if annotation.user_id != current_user.id and not check_admin(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this annotation",
            )
    else:
        if not (check_admin(current_user) or check_team_member(current_user, dataset_row.team_id, db)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete annotations in this team",
            )

    _unmirror_from_al(db, annotation)
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

