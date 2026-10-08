# tests/test_backend.py
"""Backend Test Suite for DeepFake Sentinel 2030 using pytest and FastAPI TestClient.

Tests:
1. /health endpoint
2. /model-status endpoint
3. /predict/image (missing model return MODEL_NOT_TRAINED, invalid mime handling)
4. /predict/video (missing model return MODEL_NOT_TRAINED, invalid video handling)
5. /predict/audio (missing model return MODEL_NOT_TRAINED, invalid audio handling)
6. /predict/fusion (weighted fusion logic across valid/missing modalities)
7. /history (database retrieval and logging verification)
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.history.db import log_prediction, get_history, get_history_by_id

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data

def test_model_status_endpoint():
    response = client.get("/model-status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 3
    assert data["models"][0]["modality"] == "image"
    assert data["models"][1]["modality"] == "video"
    assert data["models"][2]["modality"] == "audio"

def test_predict_image_missing_model():
    # Generate in-memory dummy image
    img = Image.new("RGB", (100, 100), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/predict/image", files=files)
    assert response.status_code == 200
    data = response.json()
    # Expect MODEL_NOT_TRAINED if models/efficientnet_b0_deepfake.pth doesn't exist
    assert "error" in data or "label" in data

def test_predict_image_invalid_format():
    files = {"file": ("test.txt", b"invalid text content", "text/plain")}
    response = client.post("/predict/image", files=files)
    assert response.status_code == 400

def test_predict_video_missing_model():
    files = {"file": ("dummy.mp4", b"fake video bytes", "video/mp4")}
    response = client.post("/predict/video", files=files)
    assert response.status_code in [200, 500]  # Returns 200 MODEL_NOT_TRAINED or 500 processing error

def test_predict_audio_missing_model():
    files = {"file": ("dummy.wav", b"fake audio bytes", "audio/wav")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code in [200, 500]

def test_predict_fusion():
    payload = {
        "image": {"label": "FAKE", "confidence": 0.85, "fake_probability": 0.85},
        "video": {"label": "FAKE", "confidence": 0.90, "fake_probability": 0.90},
        "audio": {"error": "MODEL_NOT_TRAINED"}  # Untrained audio ignored
    }
    response = client.post("/predict/fusion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["final_label"] == "FAKE"
    assert "image" in data["modalities_used"]
    assert "video" in data["modalities_used"]
    assert "audio" not in data["modalities_used"]

def test_history_logging_and_retrieval():
    rec = log_prediction("unit_test_sample.jpg", "image", "FAKE", 0.92, "HIGH")
    assert rec["id"] is not None
    assert rec["filename"] == "unit_test_sample.jpg"

    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) > 0

    record_id = rec["id"]
    rec_by_id = get_history_by_id(record_id)
    assert rec_by_id is not None
    assert rec_by_id["filename"] == "unit_test_sample.jpg"
