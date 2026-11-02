"""
scripts/evaluate_audio_repaired.py
====================================
Evaluate models/audio_cnn_repaired.pth on the untouched test split.

* File-level predictions (average chunk logits per file).
* Reports accuracy, macro precision/recall/F1, per-class metrics,
  confusion matrix, and class distributions.
* Flags model collapse if all predictions are the same class.
* Saves results to metrics/audio_metrics_repaired.json.
* Does NOT touch the test set during training — evaluation only.

Run from project root:
    python scripts/evaluate_audio_repaired.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.audio.preprocessing import (
    CLASS_TO_IDX, IDX_TO_CLASS, FAKE_IDX, REAL_IDX,
    audio_path_to_tensor, aggregate_logits,
)
from backend.models.audio_cnn import get_audio_model

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "audio_cnn_repaired.pth")
DATASET_ROOT    = os.path.join(PROJECT_ROOT, "datasets", "audio", "test")
METRICS_PATH    = os.path.join(PROJECT_ROOT, "metrics", "audio_metrics_repaired.json")
THRESHOLD_PATH  = os.path.join(PROJECT_ROOT, "metrics", "audio_threshold.json")


def get_calibrated_threshold() -> float:
    if os.path.isfile(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r") as f:
                d = json.load(f)
                return float(d.get("calibrated_threshold", 0.50))
        except Exception:
            pass
    return 0.50


def collect_files(split_dir: str):
    """Return list of (path, label_idx) for the test split."""
    items = []
    for cls_name, label_idx in CLASS_TO_IDX.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing class dir: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                items.append((os.path.join(cls_dir, fname), label_idx))
    return items


@torch.no_grad()
def evaluate():
    calibrated_thresh = get_calibrated_threshold()
    print(f"\n{'='*60}")
    print("DeepFake Sentinel 2030 — Audio Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Test set   : {DATASET_ROOT}")
    print(f"Class map  : {CLASS_TO_IDX}")
    print(f"Calibrated Threshold : {calibrated_thresh}")
    print(f"{'='*60}\n")

    if not os.path.isfile(CHECKPOINT_PATH):
        print("[ERROR] Checkpoint not found. Run train_audio_repaired.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_audio_model()
    state  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model  = model.to(device)
    model.eval()

    test_files = collect_files(DATASET_ROOT)
    if not test_files:
        print("[ERROR] No test files found.")
        sys.exit(1)

    print(f"Test files found: {len(test_files)}")

    y_true          = []
    fake_probs_all  = []
    details_default = []
    details_calib   = []
    t0              = time.time()

    for path, label_idx in test_files:
        try:
            tensor = audio_path_to_tensor(path).to(device)   # (N, 1, 128, 128)
            logits = model(tensor)                            # (N, 2)
            res_default = aggregate_logits(logits.cpu(), threshold=0.50)
            res_calib   = aggregate_logits(logits.cpu(), threshold=calibrated_thresh)
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")
            continue

        true_label = IDX_TO_CLASS[label_idx]
        y_true.append(label_idx)
        fake_probs_all.append(res_default["fake_probability"])

        # Default 0.50
        details_default.append({
            "file":       os.path.basename(path),
            "true_class": true_label,
            "pred_class": res_default["label"],
            "fake_prob":  res_default["fake_probability"],
            "real_prob":  res_default["real_probability"],
            "confidence": res_default["confidence"],
            "correct":    res_default["predicted_class_idx"] == label_idx,
        })

        # Calibrated
        details_calib.append({
            "file":       os.path.basename(path),
            "true_class": true_label,
            "pred_class": res_calib["label"],
            "fake_prob":  res_calib["fake_probability"],
            "real_prob":  res_calib["real_probability"],
            "confidence": res_calib["confidence"],
            "correct":    res_calib["predicted_class_idx"] == label_idx,
        })

        mark = "[PASS]" if (res_calib["predicted_class_idx"] == label_idx) else "[FAIL]"
        print(
            f"  {mark} {os.path.basename(path):<25s}  "
            f"true={true_label:<4s}  "
            f"pred_calib={res_calib['label']:<4s} (p_fake={res_calib['fake_probability']:.3f})"
        )

    elapsed = time.time() - t0
    y_true  = np.array(y_true)

    def calc_metrics(details):
        y_pred   = np.array([CLASS_TO_IDX[d["pred_class"]] for d in details])
        acc      = accuracy_score(y_true, y_pred)
        prec_mac = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_mac  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mac   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
        actual_d = {IDX_TO_CLASS[i]: int((y_true == i).sum()) for i in [0, 1]}
        pred_d   = {IDX_TO_CLASS[i]: int((y_pred == i).sum()) for i in [0, 1]}
        collapsed = len(set(y_pred.tolist())) == 1
        return {
            "accuracy":        round(float(acc), 4),
            "macro_precision": round(float(prec_mac), 4),
            "macro_recall":    round(float(rec_mac), 4),
            "macro_f1":        round(float(f1_mac), 4),
            "per_class": {
                "fake": {
                    "precision": round(float(prec_per[0]), 4),
                    "recall":    round(float(rec_per[0]), 4),
                    "f1":        round(float(f1_per[0]), 4),
                },
                "real": {
                    "precision": round(float(prec_per[1]), 4),
                    "recall":    round(float(rec_per[1]), 4),
                    "f1":        round(float(f1_per[1]), 4),
                }
            },
            "confusion_matrix": cm.tolist(),
            "actual_distribution": actual_d,
            "predicted_distribution": pred_d,
            "collapsed": collapsed,
        }

    m_default = calc_metrics(details_default)
    m_calib   = calc_metrics(details_calib)

    print(f"\n{'='*60}")
    print("TEST EVALUATION COMPARISON")
    print(f"{'='*60}")
    print(f"Metric               | Default (0.50)  | Calibrated ({calibrated_thresh:.2f})")
    print("-" * 55)
    print(f"Accuracy             | {m_default['accuracy']:<15.4f} | {m_calib['accuracy']:<15.4f}")
    print(f"Macro Precision      | {m_default['macro_precision']:<15.4f} | {m_calib['macro_precision']:<15.4f}")
    print(f"Macro Recall         | {m_default['macro_recall']:<15.4f} | {m_calib['macro_recall']:<15.4f}")
    print(f"Macro F1             | {m_default['macro_f1']:<15.4f} | {m_calib['macro_f1']:<15.4f}")
    print(f"Confusion Matrix     | {str(m_default['confusion_matrix']):<15s} | {str(m_calib['confusion_matrix']):<15s}")
    print(f"Predicted Dist.      | {str(m_default['predicted_distribution']):<15s} | {str(m_calib['predicted_distribution']):<15s}")
    print(f"{'='*60}\n")

    result_dict = {
        "checkpoint":          CHECKPOINT_PATH,
        "test_set":            DATASET_ROOT,
        "n_samples":           len(y_true),
        "calibrated_threshold": calibrated_thresh,
        "metrics_at_default_0_50": m_default,
        "metrics_at_calibrated":   m_calib,
        "evaluation_time_sec": round(elapsed, 2),
        "per_file_details":    details_calib,
    }

    # Backward compatibility flat keys
    result_dict.update({
        "accuracy":        m_calib["accuracy"],
        "macro_precision": m_calib["macro_precision"],
        "macro_recall":    m_calib["macro_recall"],
        "macro_f1":        m_calib["macro_f1"],
        "per_class":       m_calib["per_class"],
        "confusion_matrix": m_calib["confusion_matrix"],
        "actual_distribution": m_calib["actual_distribution"],
        "predicted_distribution": m_calib["predicted_distribution"],
        "collapsed":       m_calib["collapsed"],
    })

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Metrics saved: {METRICS_PATH}")
    return result_dict

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Metrics saved: {METRICS_PATH}")
    return result_dict


if __name__ == "__main__":
    evaluate()
