# scripts/predict_audio.py
"""Standalone CLI prediction script for a single audio file using AudioCNN.

Usage:
    python scripts/predict_audio.py \
        --model models/audio_cnn.pth \
        --audio path/to/sample.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.models.audio_cnn import get_audio_model
from backend.services.audio_service import prepare_batch

RISK_LEVELS = [(0.0, 0.4, "LOW"), (0.4, 0.7, "MEDIUM"), (0.7, 1.0, "HIGH")]

def get_risk(confidence: float) -> str:
    for low, high, label in RISK_LEVELS:
        if low <= confidence < high:
            return label
    return "HIGH"

def parse_args():
    parser = argparse.ArgumentParser(description="Predict DeepFake for a single audio file")
    parser.add_argument("--model", type=str, default="models/audio_cnn.pth", help="Path to audio checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    checkpoint = Path(args.model)
    if not checkpoint.is_file():
        print(json.dumps({"error": "MODEL_NOT_TRAINED"}))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    except Exception as e:
        print(json.dumps({"error": "MODEL_LOAD_ERROR", "detail": str(e)}))
        return
    model.eval()

    try:
        batch = prepare_batch(args.audio, device)
    except Exception as e:
        print(json.dumps({"error": f"PREPROCESSING_ERROR: {e}"}))
        return

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        fake_conf = probs[0, 1].item()

    label = "FAKE" if fake_conf >= 0.5 else "REAL"
    risk = get_risk(fake_conf)
    processing_time = round(time.time() - start_time, 3)

    res = {
        "label": label,
        "confidence": fake_conf,
        "fake_probability": fake_conf,
        "risk": risk,
        "processing_time_sec": processing_time
    }
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
