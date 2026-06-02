from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.services.visualisation_service import VISService
from app.services.fpv_cache import get_cached_fpv, set_cached_fpv
from app.schemas.visualisation import (
    FPVRequest,
    FPVDatasetRequest,
    FPVResponse,
    FPVColorField,
    FPVMethod,
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


@router.get("/fpv-dataset")
def get_fpv_dataset(
    dataset_id: int,
    embedding_model_id: int,
    run_3d: bool = False,
    method: FPVMethod | None = None,
    db: Session = Depends(get_db),
):
    # Dataset-level projections are static until regenerated. Large datasets
    # (>100k snippets) take 15-25s to build/serialize and the work is CPU-bound,
    # so recomputing per request per user starves the workers. Serve the
    # pre-serialized JSON from Redis; the first request warms it, every
    # subsequent request (any user) is served in milliseconds. We also return a
    # raw Response to skip FastAPI's response_model re-validation of ~130k points.
    method_key = method.value if method is not None else "all"
    cached = get_cached_fpv(dataset_id, embedding_model_id, method_key, run_3d)
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    service = VISService(db)
    try:
        body = FPVDatasetRequest(
            dataset_id=dataset_id,
            embedding_model_id=embedding_model_id,
            run_3d=run_3d,
            method=method,
        )
        result = service.get_fpv_for_dataset_embeddings(body)
        # Pydantic v2's Rust serializer is fast and avoids FastAPI re-encoding.
        payload = result.model_dump_json().encode("utf-8")
        set_cached_fpv(dataset_id, embedding_model_id, method_key, run_3d, payload)
        return Response(content=payload, media_type="application/json")
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
