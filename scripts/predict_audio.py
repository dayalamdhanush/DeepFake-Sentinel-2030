import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from backend.models.audio_cnn import get_audio_model

# Use the standardized preprocessing module
from ml.audio.preprocessing import (
    audio_path_to_tensor,
    aggregate_logits,
    IDX_TO_CLASS,
)

# Repaired checkpoint is preferred; falls back to original if not present
REPAIRED_CHECKPOINT = str(PROJECT_ROOT / "models" / "audio_cnn_repaired.pth")
DEFAULT_CHECKPOINT  = REPAIRED_CHECKPOINT
THRESHOLD_FILE      = PROJECT_ROOT / "metrics" / "audio_threshold.json"


def get_default_threshold() -> float:
    """Load calibrated threshold from metrics/audio_threshold.json if present, else 0.50."""
    if THRESHOLD_FILE.is_file():
        try:
            with open(THRESHOLD_FILE, "r") as f:
                data = json.load(f)
                return float(data.get("calibrated_threshold", 0.50))
        except Exception:
            pass
    return 0.50


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an audio file is FAKE or REAL."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint (.pth). Defaults to repaired checkpoint.",
    )
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for fake probability. Defaults to calibrated threshold.",
    )
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else get_default_threshold()

    start_time = time.time()
    checkpoint = Path(args.model)

    if not checkpoint.is_file():
        print(json.dumps({"error": "MODEL_NOT_TRAINED",
                          "detail": f"Checkpoint not found: {args.model}"}))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_audio_model(num_classes=2).to(device)

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()

        # ── Standardized preprocessing ────────────────────────────────────────
        tensor = audio_path_to_tensor(str(args.audio))   # (N, 1, 128, 128)
        tensor = tensor.to(device)
        n_chunks = tensor.shape[0]

        with torch.no_grad():
            logits = model(tensor)      # (N, 2)

        result = aggregate_logits(logits.cpu(), threshold=threshold)

        fake_probability = result["fake_probability"]
        real_probability = result["real_probability"]
        label            = result["label"].upper()   # 'FAKE' or 'REAL'
        confidence       = result["confidence"]

        if fake_probability >= 0.7:
            risk = "HIGH"
        elif fake_probability >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        output = {
            "label":               label,
            "confidence":          round(confidence,       4),
            "fake_probability":    round(fake_probability, 4),
            "real_probability":    round(real_probability, 4),
            "risk":                risk,
            "threshold_used":      round(threshold,        4),
            "chunks_analyzed":     n_chunks,
            "processing_time_sec": round(time.time() - start_time, 3),
            "checkpoint_used":     str(checkpoint),
        }

        print(json.dumps(output, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
