# backend/api/fusion.py
"""FastAPI router for Multi-Modal Fusion.

POST /predict/fusion
    - Accepts JSON payload containing predictions from image, video, audio endpoints.
    - Fuses available modalities using weighted score average.
    - Ignores missing/untrained modalities without fabricating data.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services.fusion_service import fuse_modalities

router = APIRouter()

class FusionRequest(BaseModel):
    image: Optional[Dict[str, Any]] = None
    video: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, float]] = None

@router.post("/predict/fusion")
async def fuse_predictions(request: FusionRequest):
    modalities = {}
    if request.image:
        modalities["image"] = request.image
    if request.video:
        modalities["video"] = request.video
    if request.audio:
        modalities["audio"] = request.audio

    result = fuse_modalities(modalities, request.weights)
    if "error" in result:
        return JSONResponse(content=result, status_code=200)

    return JSONResponse(content=result)
