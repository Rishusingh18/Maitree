from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from pydantic import BaseModel, Field
from ..services.ml_service import predict_crossing, predict_gesture, predict_behaviour
from ..config import settings

api_router = APIRouter()

# Operator authorization dependency
def get_api_key(
    key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    if settings.API_KEY:
        provided = key or x_api_key
        if provided != settings.API_KEY:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return True

class PredictImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image data")

class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class or status label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata and inference details")

@api_router.post("/predict/crossing", response_model=PredictionResponse, tags=["ML Safety"])
async def crossing_predict(req: PredictImageRequest, authorized: bool = Depends(get_api_key)):
    """Detect yellow-line crossing, track intrusion, and platform proximity."""
    try:
        label, confidence, details = await predict_crossing(req.image)
        return PredictionResponse(label=label, confidence=confidence, details=details)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Crossing inference failed: {e}")

@api_router.post("/predict/gesture", response_model=PredictionResponse, tags=["ML Gesture"])
async def gesture_predict(req: PredictImageRequest, authorized: bool = Depends(get_api_key)):
    """Detect silent distress signals (Signal for Help, Wrists crossed overhead)."""
    try:
        label, confidence, details = await predict_gesture(req.image)
        return PredictionResponse(label=label, confidence=confidence, details=details)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gesture inference failed: {e}")

@api_router.post("/predict/behaviour", response_model=PredictionResponse, tags=["ML Behaviour"])
async def behaviour_predict(req: PredictImageRequest, authorized: bool = Depends(get_api_key)):
    """Detect unusual body gesture and potential threats (theft, fights, violence)."""
    try:
        label, confidence, details = await predict_behaviour(req.image)
        return PredictionResponse(label=label, confidence=confidence, details=details)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Behaviour inference failed: {e}")
