"""
Annotation endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user
from app.schemas.annotation import Annotation, AnnotationCreate
from app.models.annotation import Annotation as AnnotationModel
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Annotation, status_code=status.HTTP_201_CREATED)
def create_annotation(
    annotation_in: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new annotation"""
    annotation = AnnotationModel(
        **annotation_in.dict(),
        user_id=current_user.id
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


@router.get("/", response_model=List[Annotation])
def read_annotations(
    snippet_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of annotations"""
    query = db.query(AnnotationModel)
    if snippet_id:
        query = query.filter(AnnotationModel.snippet_id == snippet_id)
    annotations = query.offset(skip).limit(limit).all()
    return annotations


@router.get("/{annotation_id}", response_model=Annotation)
def read_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific annotation"""
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
    """Hard delete an annotation"""
    annotation = db.query(AnnotationModel).filter(AnnotationModel.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Optional: Check if user has permission to delete (e.g., own annotation or team owner)
    # For now, allowing any authenticated user to delete any annotation
    # You may want to add permission checks here
    
    db.delete(annotation)
    db.commit()
    return None

