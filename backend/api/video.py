# backend/api/video.py
"""FastAPI router for video DeepFake detection.

POST /predict/video
    - Accepts a multipart video file (max 100 MB).
    - Validates extension (mp4, avi, mov, mkv).
    - Uses backend.services.video_service to sample frames.
    - Loads model checkpoint from models/video_cnn_lstm.pth.
    - Returns JSON with label, confidence, fake_probability, risk, sampled_frames.
    - Logs successful prediction to SQLite database.
    - If checkpoint missing -> {"error": "MODEL_NOT_TRAINED"}.
"""

import os
import time
from pathlib import Path
from typing import List, Dict

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.services.video_service import prepare_batch, get_frame_info
from ml.video.model import get_video_model
from backend.history.db import log_prediction

router = APIRouter()

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
RISK_LEVELS = [(0.0, 0.4, "LOW"), (0.4, 0.7, "MEDIUM"), (0.7, 1.0, "HIGH")]

def _risk_from_conf(conf: float) -> str:
    for lo, hi, label in RISK_LEVELS:
        if lo <= conf < hi:
            return label
    return "HIGH"

@router.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    start = time.time()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_dir = Path("temp_videos")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{int(time.time()*1000)}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            if len(content) > MAX_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Video file too large (max 100 MB)")
            buffer.write(content)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

    checkpoint_path = Path("models/video_cnn_lstm.pth")
    if not checkpoint_path.is_file():
        temp_path.unlink(missing_ok=True)
        return JSONResponse(content={"error": "MODEL_NOT_TRAINED"}, status_code=200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_video_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return JSONResponse(content={"error": "MODEL_LOAD_ERROR", "detail": str(e)}, status_code=200)
    model.eval()

    try:
        batch = prepare_batch(str(temp_path), device, num_frames=16)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video preprocessing error: {e}")

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        fake_conf = probs[0, 1].item()
    label = "FAKE" if fake_conf >= 0.5 else "REAL"
    risk = _risk_from_conf(fake_conf)
    processing_time = round(time.time() - start, 3)

    frame_info = get_frame_info(str(temp_path), num_frames=16)
    temp_path.unlink(missing_ok=True)

    # Log prediction
    try:
        log_prediction(file.filename, "video", label, fake_conf if label == "FAKE" else 1.0 - fake_conf, risk)
    except Exception:
        pass

    response = {
        "label": label,
        "confidence": fake_conf if label == "FAKE" else 1.0 - fake_conf,
        "fake_probability": fake_conf,
        "risk": risk,
        "processing_time_sec": processing_time,
        "sampled_frames": frame_info,
    }
    return JSONResponse(content=response)
