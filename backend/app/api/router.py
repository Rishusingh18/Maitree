from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from ..services.ml_service import predict_crossing, predict_gesture
from ..config import settings

api_router = APIRouter()

# Simple API key dependency (for demo)
def get_api_key(key: str = ""):
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return True

class CrossingRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image data for crossing detection")

class GestureRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image data for gesture/threat detection")

class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0‑1)")

@api_router.post("/predict/crossing", response_model=PredictionResponse, tags=["ML"])
async def crossing_predict(req: CrossingRequest, authorized: bool = Depends(get_api_key)):
    label, confidence = await predict_crossing(req.image)
    return PredictionResponse(label=label, confidence=confidence)

@api_router.post("/predict/gesture", response_model=PredictionResponse, tags=["ML"])
async def gesture_predict(req: GestureRequest, authorized: bool = Depends(get_api_key)):
    label, confidence = await predict_gesture(req.image)
    return PredictionResponse(label=label, confidence=confidence)
