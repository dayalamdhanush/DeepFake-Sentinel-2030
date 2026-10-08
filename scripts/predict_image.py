# scripts/predict_image.py
"""Standalone prediction script for a single image using the trained EfficientNet‑B0 model.

Usage:
    python scripts/predict_image.py \
        --model models/efficientnet_b0_deepfake.pth \
        --image path/to/image.jpg
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as transforms
from PIL import Image

from ml.image.model import get_image_model
from ml.image.gradcam import generate_gradcam_heatmap

RISK_LEVELS = [(0.0, 0.4, "LOW"), (0.4, 0.7, "MEDIUM"), (0.7, 1.0, "HIGH")]

def parse_args():
    parser = argparse.ArgumentParser(description="Predict DeepFake for a single image")
    parser.add_argument("--model", type=str, default="models/efficientnet_b0_deepfake.pth", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    return parser.parse_args()

def load_image(image_path: str, device: torch.device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def get_risk(confidence: float) -> str:
    for low, high, label in RISK_LEVELS:
        if low <= confidence < high:
            return label
    return "HIGH"

def main():
    args = parse_args()
    checkpoint = Path(args.model)
    if not checkpoint.is_file():
        print(json.dumps({"error": "MODEL_NOT_TRAINED"}))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_image_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    input_tensor = load_image(args.image, device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        fake_conf = probs[0, 1].item()  # index 1 -> FAKE class
    label = "FAKE" if fake_conf >= 0.5 else "REAL"
    risk = get_risk(fake_conf)

    # Grad‑CAM heatmap (only if model present)
    heatmap_img = generate_gradcam_heatmap(model, input_tensor, class_idx=1)
    buffered = io.BytesIO()
    if isinstance(heatmap_img, Image.Image):
        heatmap_img.save(buffered, format="PNG")
    else:
        Image.fromarray(heatmap_img).save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    result = {
        "label": label,
        "confidence": fake_conf,
        "risk": risk,
        "heatmap": heatmap_b64,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
