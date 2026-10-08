# scripts/train_image.py
"""Training script for Image DeepFake detection using EfficientNet-B0.

Usage:
    python scripts/train_image.py \
        --epochs 20 \
        --batch-size 32 \
        --learning-rate 1e-4 \
        --num-workers 4 \
        --fine-tune
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from ml.image.model import get_image_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for DeepFake image detection")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--fine-tune", action="store_true", help="Fine‑tune the whole backbone (default freezes backbone)")
    parser.add_argument("--data-dir", type=str, default="datasets/image", help="Root folder of image dataset (train/val/test subfolders)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save the trained checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Training/validation directories not found. Ensure dataset structure matches the specification.")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_image_model(num_classes=2)
    model = model.to(device)

    if not args.fine_tune:
        # Freeze backbone parameters
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[Epoch {epoch}/{args.epochs}] Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"[Epoch {epoch}] Validation Acc: {val_acc:.4f}")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(args.output_dir) / "efficientnet_b0_deepfake.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

if __name__ == "__main__":
    main()
