# scripts/evaluate_video.py
"""Evaluation script for the video deepfake detector.

It loads the test split from ``datasets/video/test`` and computes
accuracy, precision, recall, F1 and a confusion matrix.
Results are saved to ``metrics/video_metrics.json``.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml.video.model import get_video_model
from backend.services.video_service import load_video_frames

class VideoFolderDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int = 16):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for video_path in cls_dir.rglob("*.*"):
                if video_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    self.samples.append((video_path, self.class_to_idx[cls_name]))
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = load_video_frames(str(video_path), num_frames=self.num_frames)
        return frames, label

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video deepfake model")
    parser.add_argument("--model", type=str, default="models/video_cnn_lstm.pth", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="datasets/video", help="Root dataset directory (expects test subfolder)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames sampled per video")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint = Path(args.model)
    if not checkpoint.is_file():
        print("MODEL_NOT_TRAINED")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_video_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError("Test directory not found")

    test_dataset = VideoFolderDataset(str(test_dir), num_frames=args.num_frames)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            logits = model(frames)
            _, preds = torch.max(logits, 1)
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
    out_path = Path("metrics/video_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
