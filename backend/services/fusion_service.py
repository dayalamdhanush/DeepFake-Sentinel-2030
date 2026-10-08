# backend/services/fusion_service.py
"""Multi-Modal Fusion Service for DeepFake Sentinel 2030.

Combines available predictions from Image, Video, and Audio modalities
using weighted score fusion.
"""

from typing import Dict, Any, Optional

DEFAULT_WEIGHTS = {
    "image": 1.0,
    "video": 1.0,
    "audio": 1.0,
}

RISK_THRESHOLDS = [
    (0.0, 0.4, "LOW"),
    (0.4, 0.7, "MEDIUM"),
    (0.7, 1.0, "HIGH"),
]

def get_risk_level(fake_prob: float) -> str:
    for lo, hi, label in RISK_THRESHOLDS:
        if lo <= fake_prob < hi:
            return label
    return "HIGH"

def fuse_modalities(
    modality_results: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Fuse available modality inference results.

    :param modality_results: Dict with keys 'image', 'video', 'audio' (where provided).
                             Each inner dict should contain 'fake_probability' or 'confidence' & 'label'.
    :param weights: Optional weight override dictionary.
    :return: Comprehensive fusion result.
    """
    if not weights:
        weights = DEFAULT_WEIGHTS

    active_modalities = []
    weighted_sum = 0.0
    total_weight = 0.0

    valid_results = {}

    for mod_name in ["image", "video", "audio"]:
        if mod_name in modality_results and modality_results[mod_name]:
            res = modality_results[mod_name]
            # Check if this modality returned an error like MODEL_NOT_TRAINED
            if "error" in res or res.get("error"):
                continue

            # Extract fake probability
            if "fake_probability" in res:
                prob = float(res["fake_probability"])
            elif "confidence" in res and "label" in res:
                lbl = str(res["label"]).upper()
                conf = float(res["confidence"])
                prob = conf if lbl in ["FAKE", "DEEPFAKE"] else (1.0 - conf)
            else:
                continue

            w = float(weights.get(mod_name, 1.0))
            weighted_sum += w * prob
            total_weight += w
            active_modalities.append(mod_name)
            valid_results[mod_name] = {
                "fake_probability": prob,
                "label": "FAKE" if prob >= 0.5 else "REAL",
                "weight": w
            }

    if not active_modalities or total_weight == 0.0:
        return {
            "error": "NO_VALID_MODALITIES",
            "message": "No valid trained modality predictions were provided for fusion."
        }

    fused_fake_prob = weighted_sum / total_weight
    final_label = "FAKE" if fused_fake_prob >= 0.5 else "REAL"
    final_conf = fused_fake_prob if final_label == "FAKE" else (1.0 - fused_fake_prob)
    risk_level = get_risk_level(fused_fake_prob)
    risk_score = round(fused_fake_prob * 100, 1)

    return {
        "final_label": final_label,
        "final_confidence": round(final_conf, 4),
        "fake_probability": round(fused_fake_prob, 4),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "modalities_used": active_modalities,
        "modality_results": valid_results,
        "fusion_weights": {m: weights.get(m, 1.0) for m in active_modalities},
        "threshold_notice": "Risk levels are based on project-defined thresholds (0-40 LOW, 40-70 MEDIUM, 70-100 HIGH)."
    }
