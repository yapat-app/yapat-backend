"""Access control for the WSSED (focal recordings) workflow."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.user import User
from app.services.dataset_service import DatasetService

WSSED_ACCESS_DENIED_DETAIL = (
    "WSSED is only available for teams with focal recordings datasets."
)
WSSED_DATASET_DENIED_DETAIL = (
    "This dataset is not a focal recordings dataset or you do not have access to it."
)


def user_has_wssed_access(db: Session, user: User) -> bool:
    return DatasetService(db).user_has_wssed_access(user)


def require_wssed_access(db: Session, user: User) -> None:
    if not user_has_wssed_access(db, user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=WSSED_ACCESS_DENIED_DETAIL,
        )


def require_wssed_dataset(db: Session, user: User, dataset_id: int) -> Dataset:
    dataset = DatasetService(db).get_focal_dataset_for_user(user, dataset_id)
    if dataset is None:
        raw = DatasetService(db).get_dataset(dataset_id)
        if raw is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=WSSED_DATASET_DENIED_DETAIL,
        )
    return dataset
