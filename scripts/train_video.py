# scripts/train_video.py
"""Training script for video deepfake detection (CNN+LSTM).

Usage example:
    python scripts/train_video.py \
        --epochs 20 \
        --batch-size 8 \
        --learning-rate 1e-4 \
        --num-workers 2 \
        --data-dir datasets/video
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ml.video.model import get_video_model
from backend.services.video_service import load_video_frames

# Simple Dataset that loads video paths and labels from ImageFolder‑style dirs
class VideoFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, num_frames: int = 16):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []  # List[Tuple[Path, int]]
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for video_path in cls_dir.rglob("*.*"):
                if video_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    self.samples.append((video_path, self.class_to_idx[cls_name]))
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # Load frames as tensor (seq_len, 3, 224, 224)
        frames = load_video_frames(str(video_path), num_frames=self.num_frames)
        if self.transform:
            frames = self.transform(frames)
        return frames, label

def parse_args():
    parser = argparse.ArgumentParser(description="Train video deepfake detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="datasets/video", help="Root folder containing train/val subfolders")
    parser.add_argument("--output-dir", type=str, default="models", help="Where to save checkpoint")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to sample per video")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Train/val directories missing under the provided data-dir")

    train_dataset = VideoFolderDataset(str(train_dir), num_frames=args.num_frames)
    val_dataset = VideoFolderDataset(str(val_dir), num_frames=args.num_frames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_video_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for frames, labels in train_loader:
            frames = frames.to(device)  # (B, seq, 3, 224, 224)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[Epoch {epoch}/{args.epochs}] Train loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"[Epoch {epoch}] Validation Acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(args.output_dir) / "video_cnn_lstm.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

if __name__ == "__main__":
    main()
