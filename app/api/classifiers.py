"""
Classifier endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user
from app.schemas.classifier import Classifier, ClassifierCreate, ClassifierUpdate
from app.models.classifier import Classifier as ClassifierModel
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Classifier, status_code=status.HTTP_201_CREATED)
def create_classifier(
    classifier_in: ClassifierCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new classifier"""
    classifier = ClassifierModel(**classifier_in.dict())
    db.add(classifier)
    db.commit()
    db.refresh(classifier)
    return classifier


@router.get("/", response_model=List[Classifier])
def read_classifiers(
    team_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of classifiers"""
    query = db.query(ClassifierModel)
    if team_id:
        query = query.filter(ClassifierModel.team_id == team_id)
    classifiers = query.offset(skip).limit(limit).all()
    return classifiers


@router.get("/{classifier_id}", response_model=Classifier)
def read_classifier(
    classifier_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific classifier"""
    classifier = db.query(ClassifierModel).filter(ClassifierModel.id == classifier_id).first()
    if not classifier:
        raise HTTPException(status_code=404, detail="Classifier not found")
    return classifier

