# scripts/evaluate_image.py
"""Evaluation script for the trained EfficientNet‑B0 image model.

Usage:
    python scripts/evaluate_image.py \
        --model models/efficientnet_b0_deepfake.pth \
        --data-dir datasets/image
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml.image.model import get_image_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet‑B0 DeepFake image classifier")
    parser.add_argument("--model", type=str, default="models/efficientnet_b0_deepfake.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="datasets/image", help="Root folder of image dataset (test subfolder)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_path = Path(args.model)
    if not checkpoint_path.is_file():
        print("MODEL_NOT_TRAINED")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_image_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError("Test directory not found. Ensure dataset structure matches the specification.")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    os.makedirs("metrics", exist_ok=True)
    out_path = Path("metrics/image_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
