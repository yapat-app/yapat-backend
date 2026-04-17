from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.services.visualisation_service import VISService
from app.schemas.visualisation import (
    FPVRequest,
    FPVDatasetRequest,
    FPVResponse,
    FPVColorField,
    FPVVisibilityField,
    FPVVisibilityRangeResponse,
)
router = APIRouter()

@router.post("/fpv", response_model=FPVResponse)
def get_or_create_fpv(body: FPVRequest, db: Session = Depends(get_db)):
    service = VISService(db)
    try:
        return service.get_or_create_fpv(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV fetch/generation failed: {str(e)}")


@router.get("/fpv", response_model=FPVResponse)
def get_fpv(
    dataset_id: int,
    model_family_name: str,
    run_3d: bool = False,
    color_filter_value: FPVColorField = FPVColorField.PREDICTED_LABEL,
    visibility_filter_value: FPVVisibilityField = FPVVisibilityField.COMPOSITE,
    visibility_range_min: float | None = None,
    visibility_range_max: float | None = None,
    db: Session = Depends(get_db),
):
    service = VISService(db)
    try:
        body = FPVRequest(
            dataset_id=dataset_id,
            model_family_name=model_family_name,
            run_3d=run_3d,
            color_filter_value=color_filter_value,
            visibility_filter_value=visibility_filter_value,
            visibility_range_min=visibility_range_min,
            visibility_range_max=visibility_range_max,
        )
        return service.get_or_create_fpv(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV fetch/generation failed: {str(e)}")


@router.post("/fpv-dataset", response_model=FPVResponse)
def generate_fpv_dataset(body: FPVDatasetRequest, db: Session = Depends(get_db)):
    service = VISService(db)
    try:
        return service.generate_fpv_for_dataset_embeddings(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV dataset generation failed: {str(e)}")


@router.get("/fpv-dataset", response_model=FPVResponse)
def get_fpv_dataset(
    dataset_id: int,
    embedding_model_id: int,
    run_3d: bool = False,
    db: Session = Depends(get_db),
):
    service = VISService(db)
    try:
        body = FPVDatasetRequest(
            dataset_id=dataset_id,
            embedding_model_id=embedding_model_id,
            run_3d=run_3d,
        )
        return service.get_fpv_for_dataset_embeddings(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV dataset fetch failed: {str(e)}")


@router.get("/fpv_vis_range", response_model=FPVVisibilityRangeResponse)
def get_fpv_vis_range(
    visibility_filter_value: FPVVisibilityField = Query(..., description="Visibility filter field"),
    db: Session = Depends(get_db),
):
    service = VISService(db)
    try:
        return service.get_fpv_vis_range(visibility_filter_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV fetch failed: {str(e)}")
