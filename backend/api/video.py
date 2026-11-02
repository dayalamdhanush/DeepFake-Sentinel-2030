# backend/api/video.py
"""FastAPI router for video DeepFake detection.

POST /predict/video
    - Accepts a multipart video file (mp4, avi, mov, mkv - max 100 MB).
    - Uses backend.services.video_service to sample 16 sequential uniform frames.
    - Loads model checkpoint from models/video_cnn_lstm.pth.
    - Evaluates temporal sequence with ResNet-18 + LSTM.
    - Accurately maps class 0 (Fake/Synthetic) and class 1 (Real/Authentic).
    - Returns JSON with label, confidence, fake_probability, risk, sampled_frames.
    - Logs successful prediction to SQLite database.
    - If checkpoint missing -> {"error": "MODEL_NOT_TRAINED"}.
"""

import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.services.video_service import prepare_batch, get_frame_info
from ml.video.model import get_video_model
from backend.history.db import log_prediction

router = APIRouter()

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

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
        # Class 0: Synthetic / DeepFake, Class 1: Authentic Human Video
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()

    if real_prob >= fake_prob:
        label = "REAL"
        confidence = round(real_prob, 4)
        risk = "LOW" if real_prob >= 0.65 else "MEDIUM"
    else:
        label = "FAKE"
        confidence = round(fake_prob, 4)
        risk = "HIGH" if fake_prob >= 0.65 else "MEDIUM"

    processing_time = round(time.time() - start, 3)

    frame_info = get_frame_info(str(temp_path), num_frames=16)
    temp_path.unlink(missing_ok=True)

    # Log prediction
    try:
        log_prediction(file.filename, "video", label, confidence, risk)
    except Exception:
        pass

    response = {
        "label": label,
        "confidence": confidence,
        "fake_probability": round(fake_prob, 4),
        "risk": risk,
        "processing_time_sec": processing_time,
        "sampled_frames": frame_info,
    }
    return JSONResponse(content=response)
