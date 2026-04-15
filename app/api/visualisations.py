from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_active_user
from app.services.visualisation_service import VISService
from app.schemas.visualisation import FPVRequest, FPVResponse, FPVVisibilityField, FPVVisibilityRangeResponse
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
        raise HTTPException(status_code=500, detail=f"FPV visibility range fetch failed: {str(e)}")