# backend/main.py
"""Main FastAPI entry point for DeepFake Sentinel 2030.

Provides endpoints for:
- Health check (/health)
- Image detection (/predict/image)
- Video detection (/predict/video)
- Audio detection (/predict/audio)
- Multi-modal fusion (/predict/fusion)
- Prediction history (/history, /history/{id})
- Model status (/model-status)
"""

import os
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.api import image, video, audio, fusion
from backend.history.db import get_history, get_history_by_id

app = FastAPI(
    title="DeepFake Sentinel 2030",
    description="Explainable Multi-Modal DeepFake Detection System API",
    version="1.0.0"
)

# Enable CORS for local dev and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Modality Routers
app.include_router(image.router, prefix="", tags=["image"])
app.include_router(video.router, prefix="", tags=["video"])
app.include_router(audio.router, prefix="", tags=["audio"])
app.include_router(fusion.router, prefix="", tags=["fusion"])

@app.get("/health")
async def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "system": "DeepFake Sentinel 2030", "device": device}

@app.get("/history")
async def fetch_history(limit: int = Query(default=50, ge=1, le=500)):
    records = get_history(limit=limit)
    return {"history": records, "count": len(records)}

@app.get("/history/{record_id}")
async def fetch_history_by_id(record_id: int):
    record = get_history_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction history record not found")
    return record

@app.get("/model-status")
async def check_model_status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models_info = [
        {
            "name": "EfficientNet-B0 Image Model",
            "modality": "image",
            "architecture": "EfficientNet-B0 (Transfer Learning)",
            "checkpoint_path": "models/efficientnet_b0_deepfake.pth",
            "status": "TRAINED" if Path("models/efficientnet_b0_deepfake.pth").is_file() else "NOT_TRAINED",
        },
        {
            "name": "ResNet-18 + LSTM Video Model",
            "modality": "video",
            "architecture": "ResNet-18 CNN + LSTM (Temporal Sequence Modeling)",
            "checkpoint_path": "models/video_cnn_lstm.pth",
            "status": "TRAINED" if Path("models/video_cnn_lstm.pth").is_file() else "NOT_TRAINED",
        },
        {
            "name": "Log-Mel Spectrogram Audio CNN Model",
            "modality": "audio",
            "architecture": "Log-Mel Spectrogram (128 bins) + 3-Block CNN",
            "checkpoint_path": "models/audio_cnn.pth",
            "status": "TRAINED" if Path("models/audio_cnn.pth").is_file() else "NOT_TRAINED",
        }
    ]

    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models": models_info
    }

# Mount static frontend directory if present
frontend_dir = Path("frontend")
if frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

