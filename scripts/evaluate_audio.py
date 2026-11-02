# scripts/evaluate_audio.py
"""Evaluation script for the AudioCNN model on test split.

Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, and saves to metrics/audio_metrics.json.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from backend.models.audio_cnn import get_audio_model
from scripts.train_audio import AudioFolderDataset, pad_collate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake model")
    parser.add_argument("--model", type=str, default="models/audio_cnn.pth", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="datasets/audio", help="Root dataset dir")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint = Path(args.model)
    if not checkpoint.is_file():
        print("MODEL_NOT_TRAINED")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError("Test directory not found under datasets/audio")

    test_dataset = AudioFolderDataset(str(test_dir))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)

            if specs.ndim == 5:
                batch_size, segments, channels, height, width = specs.shape
                specs = specs.reshape(
                    batch_size * segments, channels, height, width
                )
                outputs = model(specs)
                outputs = outputs.reshape(batch_size, segments, -1).mean(dim=1)
            else:
                outputs = model(specs)

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
    out_path = Path("metrics/audio_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
