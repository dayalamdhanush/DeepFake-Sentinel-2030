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
            
            mod_label = res.get("label", "FAKE" if prob >= 0.5 else "REAL")

            valid_results[mod_name] = {
                "fake_probability": round(prob, 4),
                "label": mod_label,
                "weight": w
            }

    if not active_modalities or total_weight == 0.0:
        return {
            "error": "NO_VALID_MODALITIES",
            "message": "No valid trained modality predictions were provided for fusion."
        }

    fused_fake_prob = weighted_sum / total_weight
    if fused_fake_prob < 0.50:
        final_label = "REAL"
        final_conf = round(1.0 - fused_fake_prob, 4)
        risk_level = "LOW" if fused_fake_prob < 0.35 else "MEDIUM"
    else:
        final_label = "FAKE"
        final_conf = round(fused_fake_prob, 4)
        risk_level = "HIGH" if fused_fake_prob >= 0.70 else "MEDIUM"

    risk_score = round(fused_fake_prob * 100, 1)

    return {
        "final_label": final_label,
        "final_confidence": final_conf,
        "fake_probability": round(fused_fake_prob, 4),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "modalities_used": active_modalities,
        "modality_results": valid_results,
        "fusion_weights": {m: weights.get(m, 1.0) for m in active_modalities},
        "threshold_notice": "Forensic evaluation: Authentic < 50%, DeepFake >= 50%."
    }
