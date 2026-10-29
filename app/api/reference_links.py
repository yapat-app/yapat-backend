"""
Reference data pool link endpoints.

Admin-only for now (reference datasets are provisioned/ops-managed for the
user study). See docs/reference-data-pool-design.md.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_admin_user
from app.models.dataset import Dataset as DatasetModel
from app.models.team import Team as TeamModel
from app.models.reference_link import DatasetReferenceLink as DatasetReferenceLinkModel
from app.models.user import User
from app.schemas.reference_link import DatasetReferenceLink, DatasetReferenceLinkCreate

router = APIRouter()


@router.post("/", response_model=DatasetReferenceLink, status_code=status.HTTP_201_CREATED)
def create_reference_link(
    body: DatasetReferenceLinkCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    ref_ds = db.query(DatasetModel).filter(DatasetModel.id == body.reference_dataset_id).first()
    if ref_ds is None:
        raise HTTPException(status_code=404, detail="Reference dataset not found")
    if not ref_ds.is_reference:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {ref_ds.id} is not marked is_reference=True",
        )

    if body.dataset_id is not None:
        target = db.query(DatasetModel).filter(DatasetModel.id == body.dataset_id).first()
        if target is None:
            raise HTTPException(status_code=404, detail="Target dataset not found")
        if target.is_reference:
            raise HTTPException(status_code=400, detail="Target dataset cannot itself be reference-only")
    else:
        team = db.query(TeamModel).filter(TeamModel.id == body.team_id).first()
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

    existing = (
        db.query(DatasetReferenceLinkModel)
        .filter(
            DatasetReferenceLinkModel.reference_dataset_id == body.reference_dataset_id,
            DatasetReferenceLinkModel.dataset_id == body.dataset_id,
            DatasetReferenceLinkModel.team_id == body.team_id,
        )
        .first()
    )
    if existing is not None:
        return existing

    link = DatasetReferenceLinkModel(
        reference_dataset_id=body.reference_dataset_id,
        dataset_id=body.dataset_id,
        team_id=body.team_id,
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


@router.get("/", response_model=List[DatasetReferenceLink])
def list_reference_links(
    dataset_id: Optional[int] = Query(None, description="Filter to links targeting this dataset"),
    team_id: Optional[int] = Query(None, description="Filter to links targeting this team"),
    reference_dataset_id: Optional[int] = Query(None, description="Filter to links sourced from this reference dataset"),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    query = db.query(DatasetReferenceLinkModel)
    if dataset_id is not None:
        query = query.filter(DatasetReferenceLinkModel.dataset_id == dataset_id)
    if team_id is not None:
        query = query.filter(DatasetReferenceLinkModel.team_id == team_id)
    if reference_dataset_id is not None:
        query = query.filter(DatasetReferenceLinkModel.reference_dataset_id == reference_dataset_id)
    return query.order_by(DatasetReferenceLinkModel.created_at.desc()).all()


@router.delete("/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reference_link(
    link_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    link = db.query(DatasetReferenceLinkModel).filter(DatasetReferenceLinkModel.id == link_id).first()
    if link is None:
        raise HTTPException(status_code=404, detail="Reference link not found")
    db.delete(link)
    db.commit()
    return None
