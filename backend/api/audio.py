# backend/api/audio.py
"""FastAPI router for audio DeepFake detection.

POST /predict/audio
    - Accepts an audio file (wav, mp3, ogg, flac, m4a - max 20 MB).
    - Uses ml.audio.preprocessing (via audio_service) for standardized
      log-Mel spectrogram chunks: 16 kHz, 3s, 128-bin, fixed 128 frames.
    - CLASS MAPPING: FAKE=0, REAL=1 (consistent with training).
    - Prefers models/audio_cnn_repaired.pth; falls back to audio_cnn.pth.
    - Returns JSON with label, confidence, fake_probability, real_probability,
      risk, chunks_analyzed, processing_time_sec.
    - Logs successful prediction to SQLite database.
    - If no checkpoint exists -> {"error": "MODEL_NOT_TRAINED"}.
"""

import os
import time
from pathlib import Path

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.services.audio_service import prepare_batch, aggregate_logits
from backend.models.audio_cnn import get_audio_model
from backend.history.db import log_prediction

router = APIRouter()

ALLOWED_EXT    = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Preferred checkpoint (repaired), fallback to original
_REPAIRED       = Path("models/audio_cnn_repaired.pth")
_ORIGINAL       = Path("models/audio_cnn.pth")
_THRESHOLD_FILE = Path("metrics/audio_threshold.json")


def _get_threshold() -> float:
    """Return calibrated threshold from metrics/audio_threshold.json if present, else 0.50."""
    if _THRESHOLD_FILE.is_file():
        try:
            import json
            with open(_THRESHOLD_FILE, "r") as f:
                data = json.load(f)
                return float(data.get("calibrated_threshold", 0.50))
        except Exception:
            pass
    return 0.50


def _get_checkpoint() -> Path | None:
    """Return the best available checkpoint, or None if none exist."""
    if _REPAIRED.is_file():
        return _REPAIRED
    if _ORIGINAL.is_file():
        return _ORIGINAL
    return None


@router.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    start = time.time()

    # ── Validate extension ─────────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # ── Save temp file ─────────────────────────────────────────────────────────
    temp_dir  = Path("temp_audios")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{int(time.time() * 1000)}_{file.filename}"
    content   = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413,
                            detail="Audio file too large (max 20 MB)")
    with open(temp_path, "wb") as f:
        f.write(content)

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    checkpoint_path = _get_checkpoint()
    if checkpoint_path is None:
        temp_path.unlink(missing_ok=True)
        return JSONResponse(content={"error": "MODEL_NOT_TRAINED"}, status_code=200)

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_audio_model(num_classes=2).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return JSONResponse(
            content={"error": "MODEL_LOAD_ERROR", "detail": str(e)},
            status_code=200,
        )
    model.eval()

    # ── Preprocess ─────────────────────────────────────────────────────────────
    try:
        batch = prepare_batch(str(temp_path), device)   # (N, 1, 128, 128)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500,
                            detail=f"Audio preprocessing error: {e}")
    finally:
        temp_path.unlink(missing_ok=True)

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(batch)                            # (N, 2)

    # Use the standardized aggregation (FAKE=0, REAL=1) with calibrated threshold
    threshold  = _get_threshold()
    result     = aggregate_logits(logits.cpu(), threshold=threshold)
    fake_prob  = result["fake_probability"]
    real_prob  = result["real_probability"]
    confidence = result["confidence"]
    label      = result["label"].upper()                 # 'FAKE' or 'REAL'
    n_chunks   = int(batch.shape[0])

    if fake_prob >= 0.7:
        risk = "HIGH"
    elif fake_prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    processing_time = round(time.time() - start, 3)

    # ── Log prediction ─────────────────────────────────────────────────────────
    try:
        log_prediction(file.filename, "audio", label, confidence, risk)
    except Exception:
        pass  # Never let logging failures break the response

    response = {
        "label":               label,
        "confidence":          round(confidence, 4),
        "fake_probability":    round(fake_prob,  4),
        "real_probability":    round(real_prob,  4),
        "risk":                risk,
        "threshold_used":      round(threshold,  4),
        "chunks_analyzed":     n_chunks,
        "processing_time_sec": processing_time,
        "checkpoint_used":     str(checkpoint_path),
    }
    return JSONResponse(content=response)
