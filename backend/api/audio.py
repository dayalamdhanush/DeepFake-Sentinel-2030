# backend/api/audio.py
"""FastAPI router for audio DeepFake detection.

POST /predict/audio
    - Accepts an audio file (wav, mp3, ogg, flac, m4a - max 20 MB).
    - Uses backend.services.audio_service to build log-Mel spectrogram.
    - Runs AudioCNN model if checkpoint exists.
    - Returns JSON with label, confidence, fake_probability, risk, processing_time.
    - Logs successful prediction to SQLite database.
    - If checkpoint missing -> {"error": "MODEL_NOT_TRAINED"}.
"""

import os
import time
from pathlib import Path

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.services.audio_service import prepare_batch
from backend.models.audio_cnn import get_audio_model
from backend.history.db import log_prediction

router = APIRouter()

ALLOWED_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
RISK_LEVELS = [(0.0, 0.4, "LOW"), (0.4, 0.7, "MEDIUM"), (0.7, 1.0, "HIGH")]

def _risk_from_conf(conf: float) -> str:
    for lo, hi, label in RISK_LEVELS:
        if lo <= conf < hi:
            return label
    return "HIGH"

@router.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    start = time.time()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    temp_dir = Path("temp_audios")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{int(time.time()*1000)}_{file.filename}"
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 20 MB)")
    with open(temp_path, "wb") as f:
        f.write(content)

    checkpoint_path = Path("models/audio_cnn.pth")
    if not checkpoint_path.is_file():
        temp_path.unlink(missing_ok=True)
        return JSONResponse(content={"error": "MODEL_NOT_TRAINED"}, status_code=200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return JSONResponse(content={"error": "MODEL_LOAD_ERROR", "detail": str(e)}, status_code=200)
    model.eval()

    try:
        batch = prepare_batch(str(temp_path), device)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Audio preprocessing error: {e}")

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        fake_conf = probs[0, 1].item()
    label = "FAKE" if fake_conf >= 0.5 else "REAL"
    risk = _risk_from_conf(fake_conf)
    processing_time = round(time.time() - start, 3)
    temp_path.unlink(missing_ok=True)

    # Log prediction
    try:
        log_prediction(file.filename, "audio", label, fake_conf if label == "FAKE" else 1.0 - fake_conf, risk)
    except Exception:
        pass

    response = {
        "label": label,
        "confidence": fake_conf if label == "FAKE" else 1.0 - fake_conf,
        "fake_probability": fake_conf,
        "risk": risk,
        "processing_time_sec": processing_time,
    }
    return JSONResponse(content=response)
