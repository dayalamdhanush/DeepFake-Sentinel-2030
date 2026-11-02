"""
scripts/calibrate_audio_threshold.py
======================================
Validation-based threshold calibration for DeepFake Sentinel 2030 audio model.

Rules:
1. Evaluates ONLY the validation dataset (datasets/audio/val).
2. Never accesses or optimizes on the test dataset.
3. Tests candidate decision thresholds between 0.30 and 0.70.
4. Objective: Maximize validation Macro F1 score (with Accuracy and Macro Recall as secondary tie-breakers).
5. Saves selected threshold to metrics/audio_threshold.json.
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.audio.preprocessing import (
    CLASS_TO_IDX, IDX_TO_CLASS, FAKE_IDX, REAL_IDX,
    audio_path_to_tensor, aggregate_logits
)
from backend.models.audio_cnn import get_audio_model

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "audio_cnn_repaired.pth")
VAL_DIR         = os.path.join(PROJECT_ROOT, "datasets", "audio", "val")
OUTPUT_JSON     = os.path.join(PROJECT_ROOT, "metrics", "audio_threshold.json")


def load_val_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_model().to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    fake_probs = []
    files = []

    for cls_name in ["fake", "real"]:
        cls_idx = CLASS_TO_IDX[cls_name]
        d = os.path.join(VAL_DIR, cls_name)
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                fpath = os.path.join(d, fname)
                tensor = audio_path_to_tensor(fpath).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                res = aggregate_logits(logits.cpu())
                y_true.append(cls_idx)
                fake_probs.append(res["fake_probability"])
                files.append(f"{cls_name}/{fname}")

    return np.array(y_true), np.array(fake_probs), files


def run_calibration():
    print(f"\n{'='*70}")
    print("DeepFake Sentinel 2030 — Validation-Based Threshold Calibration")
    print(f"{'='*70}")
    print(f"Checkpoint  : {CHECKPOINT_PATH}")
    print(f"Dataset     : {VAL_DIR} (ONLY validation split)")
    print(f"Objective   : Maximize Macro F1")
    print(f"{'='*70}\n")

    y_true, fake_probs, files = load_val_predictions()
    print(f"Loaded {len(y_true)} validation samples ({sum(y_true==0)} fake, {sum(y_true==1)} real).")

    candidate_thresholds = np.round(np.arange(0.30, 0.71, 0.02), 2)
    results = []

    best_thresh = 0.50
    best_f1 = -1.0
    best_acc = -1.0

    print("\n--- Candidate Threshold Grid Search (Step: 0.02) ---")
    print(f"{'Threshold':<10} | {'Val Acc':<8} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10} | {'CM [TN, FP / FN, TP]'}")
    print("-" * 75)

    for thresh in candidate_thresholds:
        # If fake_prob >= thresh -> FAKE (0), else REAL (1)
        y_pred = np.where(fake_probs >= thresh, FAKE_IDX, REAL_IDX)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        row = {
            "threshold": float(thresh),
            "val_accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec), 4),
            "macro_recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm
        }
        results.append(row)

        print(f"{thresh:<10.2f} | {acc:<8.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {cm}")

        # Selection logic: highest macro_f1, tie-break on accuracy, tie-break closest to 0.5
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc) or (f1 == best_f1 and acc == best_acc and abs(thresh - 0.5) < abs(best_thresh - 0.5)):
            best_f1 = f1
            best_acc = acc
            best_thresh = float(thresh)

    best_row = next(r for r in results if r["threshold"] == best_thresh)
    orig_row = next(r for r in results if r["threshold"] == 0.50)

    print("\n" + "="*70)
    print("CALIBRATION SUMMARY ON VALIDATION SET")
    print("="*70)
    print(f"Default 0.50 Threshold : Acc={orig_row['val_accuracy']:.4f}, Macro F1={orig_row['macro_f1']:.4f}, CM={orig_row['confusion_matrix']}")
    print(f"Selected Threshold ({best_thresh:.2f}): Acc={best_row['val_accuracy']:.4f}, Macro F1={best_row['macro_f1']:.4f}, CM={best_row['confusion_matrix']}")
    print("="*70)

    output = {
        "checkpoint": CHECKPOINT_PATH,
        "validation_set": VAL_DIR,
        "n_val_samples": len(y_true),
        "selection_objective": "Maximized Validation Macro F1 with Validation Accuracy tie-breaker",
        "default_threshold": 0.50,
        "calibrated_threshold": best_thresh,
        "validation_performance_at_default": orig_row,
        "validation_performance_at_calibrated": best_row,
        "all_candidates": results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibrated threshold successfully saved to: {OUTPUT_JSON}")
    return best_thresh


if __name__ == "__main__":
    run_calibration()
