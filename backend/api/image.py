# backend/api/image.py
"""FastAPI router for image DeepFake detection.

POST /predict/image
    - Accepts an image file (JPEG, PNG, WEBP, BMP - max 10 MB).
    - Runs EfficientNet-B0 model if trained checkpoint exists.
    - Accurately maps class 0 (Fake/Synthetic) and class 1 (Real/Authentic).
    - Generates real Grad-CAM heatmap visualization.
    - Returns JSON response with label, confidence, fake_probability, risk, gradcam (base64 data URI).
    - Logs successful prediction to SQLite database.
    - If checkpoint missing -> {"error": "MODEL_NOT_TRAINED"}.
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from ml.image.model import get_image_model
from ml.image.gradcam import generate_gradcam_heatmap
from backend.history.db import log_prediction

router = APIRouter()

MODEL_PATH = Path("models/efficientnet_b0_deepfake.pth")
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

@router.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()
    
    if file.content_type not in ALLOWED_MIME and not any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        raise HTTPException(status_code=400, detail="Invalid image format. Supported formats: JPEG, PNG, WEBP, BMP.")
    
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Image file too large (max 10 MB).")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    if not MODEL_PATH.is_file():
        return JSONResponse(content={"error": "MODEL_NOT_TRAINED"}, status_code=200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_image_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        return JSONResponse(content={"error": "MODEL_LOAD_ERROR", "detail": str(e)}, status_code=200)
    model.eval()

    # Preprocessing matching model training pipeline (Resize 224x224)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        # Class 0: Synthetic / DeepFake, Class 1: Authentic Human
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()

    if real_prob >= fake_prob:
        label = "REAL"
        confidence = round(real_prob, 4)
        risk = "LOW" if real_prob >= 0.70 else "MEDIUM"
    else:
        label = "FAKE"
        confidence = round(fake_prob, 4)
        risk = "HIGH" if fake_prob >= 0.70 else "MEDIUM"

    processing_time = round(time.time() - start_time, 3)

    # Grad-CAM heatmap generation: visualize activations for the predicted class
    class_idx = 0 if label == "FAKE" else 1
    heatmap_img = generate_gradcam_heatmap(model, input_tensor, target_layer="features[7]", class_idx=class_idx)
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Log to SQLite history
    try:
        log_prediction(file.filename, "image", label, confidence, risk)
    except Exception:
        pass

    response: Dict[str, Any] = {
        "label": label,
        "confidence": confidence,
        "fake_probability": round(fake_prob, 4),
        "risk": risk,
        "processing_time_sec": processing_time,
        "gradcam": f"data:image/png;base64,{heatmap_b64}",
    }
    return JSONResponse(content=response)
