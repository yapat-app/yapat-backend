from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_active_user
from app.services.visualisation_service import VISService
from app.schemas.visualisation import FPVRequest, FPVResponse
router = APIRouter()

@router.post("/fpv", response_model=FPVResponse)
def generate_fpv(body: FPVRequest, db: Session = Depends(get_db)):
    service = VISService(db)
    try:
        return service.generate_fpv_for_checkpoint(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV generation failed: {str(e)}")

@router.get("/fpv", response_model=FPVResponse)
def get_fpv(
    dataset_id: int,
    model_family_name: str,
    run_3d: bool = False,
    db: Session = Depends(get_db),
):
    service = VISService(db)
    try:
        body = FPVRequest(
            dataset_id=dataset_id,
            model_family_name=model_family_name,
            run_3d=run_3d,
        )
        return service.get_fpv_for_checkpoint(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPV fetch failed: {str(e)}")