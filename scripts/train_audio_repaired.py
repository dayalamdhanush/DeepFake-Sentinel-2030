"""
scripts/train_audio_repaired.py
================================
Repaired audio deepfake training script.

Key design decisions
--------------------
* Uses ml.audio.preprocessing as the SINGLE source of truth for preprocessing.
* Explicit class mapping: FAKE=0, REAL=1 (never relies on sort order).
* Fixed 128×128 spectrogram tensors — consistent with inference.
* All chunks from each file are used as independent training samples.
  File-level labels are carried to each chunk, so no label leakage.
* Validation: file-level prediction (average chunk logits per file).
* Early stopping on validation accuracy (patience=10 epochs).
* Saves ONLY to models/audio_cnn_repaired.pth.
  NEVER overwrites audio_cnn.pth / audio_cnn_previous.pth /
  audio_cnn_experimental.pth.
* Reproducible: seed=42.

Run from project root:
    python scripts/train_audio_repaired.py
"""

import os
import sys
import random
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.audio.preprocessing import (
    CLASS_TO_IDX, IDX_TO_CLASS, FAKE_IDX, REAL_IDX,
    audio_path_to_tensor, aggregate_logits,
    N_MELS, FIXED_FRAMES,
)
from backend.models.audio_cnn import get_audio_model

# ── Constants ──────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 16
EPOCHS       = 60
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10           # early-stopping patience (epochs)

SAVE_PATH    = os.path.join(PROJECT_ROOT, "models", "audio_cnn_repaired.pth")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "audio")

# Files that must NEVER be overwritten
PROTECTED = {
    os.path.join(PROJECT_ROOT, "models", "audio_cnn.pth"),
    os.path.join(PROJECT_ROOT, "models", "audio_cnn_previous.pth"),
    os.path.join(PROJECT_ROOT, "models", "audio_cnn_experimental.pth"),
}


# ── Reproducibility ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Safety check ───────────────────────────────────────────────────────────────
def check_save_path(path: str):
    abs_path = os.path.abspath(path)
    for p in PROTECTED:
        if os.path.abspath(p) == abs_path:
            raise RuntimeError(
                f"SAFETY: attempted to write to protected checkpoint: {abs_path}"
            )


# ── Dataset ────────────────────────────────────────────────────────────────────
class AudioChunkDataset(Dataset):
    """
    Each item is a single (chunk_spectrogram, label) pair.
    All chunks from every file are enumerated at construction time.

    Parameters
    ----------
    split : str   'train' or 'val'
    """

    def __init__(self, split: str):
        super().__init__()
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        self.samples = []   # list of (tensor_shape_1x128x128, label_idx)
        counts       = {cls: 0 for cls in CLASS_TO_IDX}

        for cls_name, label_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
            )
            for fname in files:
                fpath = os.path.join(cls_dir, fname)
                try:
                    tensor = audio_path_to_tensor(fpath)   # (N, 1, 128, 128)
                except Exception as e:
                    print(f"  [WARN] Skipping {fpath}: {e}")
                    continue
                for chunk_idx in range(tensor.shape[0]):
                    self.samples.append((tensor[chunk_idx], label_idx))
                counts[cls_name] += 1

        print(f"[Dataset:{split}] files per class: {counts}")
        print(f"[Dataset:{split}] total chunks  : {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(f"No audio samples found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec, label = self.samples[idx]
        return spec, label


class AudioFileDataset(Dataset):
    """
    File-level dataset used for validation.
    Returns all chunks for a file and the label.
    Used to produce file-level predictions.
    """

    def __init__(self, split: str):
        super().__init__()
        split_dir = os.path.join(DATASET_ROOT, split)
        self.files = []   # list of (path, label_idx)
        for cls_name, label_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    self.files.append(
                        (os.path.join(cls_dir, fname), label_idx)
                    )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        tensor = audio_path_to_tensor(path)   # (N, 1, 128, 128)
        return tensor, label


# ── File-level validation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_file_level(model, file_dataset, device):
    """
    Evaluate on a file-level dataset.
    Returns accuracy (float) and a list of (true, pred) pairs.
    """
    model.eval()
    correct = 0
    pairs   = []

    for tensor, label in file_dataset:
        tensor = tensor.to(device)          # (N, 1, 128, 128)
        logits = model(tensor)              # (N, 2)
        result = aggregate_logits(logits.cpu())
        pred   = result["predicted_class_idx"]
        pairs.append((int(label), pred))
        if pred == int(label):
            correct += 1

    acc = correct / len(file_dataset) if len(file_dataset) > 0 else 0.0
    return acc, pairs


# ── Training loop ──────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    check_save_path(SAVE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("DeepFake Sentinel 2030 — Audio Repair Training")
    print(f"{'='*60}")
    print(f"Device     : {device}")
    print(f"Save path  : {SAVE_PATH}")
    print(f"Dataset    : {DATASET_ROOT}")
    print(f"Class map  : {CLASS_TO_IDX}")
    print(f"{'='*60}\n")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = AudioChunkDataset("train")
    val_dataset   = AudioFileDataset("val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_audio_model()
    model = model.to(device)

    # ── Loss, optimizer ────────────────────────────────────────────────────────
    # Balanced dataset (50/50), equal weights
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training state ─────────────────────────────────────────────────────────
    best_val_acc  = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    history       = []

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct    = 0
        n_samples  = 0

        for specs, labels in train_loader:
            specs  = specs.to(device)           # (B, 1, 128, 128)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(specs)               # (B, 2)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * specs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            n_samples  += specs.size(0)

        train_loss = total_loss / n_samples
        train_acc  = correct   / n_samples

        # ── Validate (file-level) ─────────────────────────────────────────────
        val_acc, _ = evaluate_file_level(model, val_dataset, device)
        scheduler.step(val_acc)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_PATH)
            marker = "  ← SAVED"
        else:
            patience_cnt += 1
            marker = f"  (patience {patience_cnt}/{PATIENCE})"

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_acc":    round(val_acc,    4),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}"
            f"{marker}"
        )

        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val_acc : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint   : {SAVE_PATH}")
    print(f"{'='*60}\n")

    # Save training history
    metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, "audio_training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {"best_val_acc": best_val_acc, "best_epoch": best_epoch,
             "history": history},
            f, indent=2,
        )
    print(f"History saved: {history_path}")

    # Verify protected checkpoints were not touched
    for p in PROTECTED:
        if os.path.exists(p):
            print(f"[OK] Protected checkpoint intact: {os.path.basename(p)}")
        else:
            print(f"[WARN] Protected checkpoint missing (was already absent): {os.path.basename(p)}")


if __name__ == "__main__":
    train()
