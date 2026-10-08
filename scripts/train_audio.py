# scripts/train_audio.py
"""Training script for Audio DeepFake detection using AudioCNN.

Usage:
    python scripts/train_audio.py \
        --epochs 20 \
        --batch-size 16 \
        --learning-rate 1e-3 \
        --data-dir datasets/audio
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from backend.models.audio_cnn import get_audio_model
from backend.services.audio_service import load_audio_tensor

class AudioFolderDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        allowed_ext = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for audio_path in cls_dir.rglob("*.*"):
                if audio_path.suffix.lower() in allowed_ext:
                    self.samples.append((audio_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        spec_tensor = load_audio_tensor(str(audio_path))  # (1, 1, 128, T)
        spec_tensor = spec_tensor.squeeze(0)  # (1, 128, T)
        return spec_tensor, label

def pad_collate(batch):
    # Batch is list of tuples (tensor(1, 128, T), label)
    tensors, labels = zip(*batch)
    max_t = max(t.size(-1) for t in tensors)
    padded_tensors = []
    for t in tensors:
        pad_len = max_t - t.size(-1)
        if pad_len > 0:
            t = nn.functional.pad(t, (0, pad_len))
        padded_tensors.append(t)
    stacked = torch.stack(padded_tensors)  # (B, 1, 128, max_t)
    labels = torch.tensor(labels, dtype=torch.long)
    return stacked, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Train AudioCNN deepfake detector")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="datasets/audio")
    parser.add_argument("--output-dir", type=str, default="models")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Train/val directories missing under datasets/audio")

    train_dataset = AudioFolderDataset(str(train_dir))
    val_dataset = AudioFolderDataset(str(val_dir))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=args.num_workers)

    model = get_audio_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)
                outputs = model(specs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"[Epoch {epoch}] Validation Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(args.output_dir) / "audio_cnn.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

if __name__ == "__main__":
    main()
