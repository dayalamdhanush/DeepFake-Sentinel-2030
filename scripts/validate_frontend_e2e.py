# scripts/validate_frontend_e2e.py
"""End-to-End Application & Frontend Flow Validator for DeepFake Sentinel 2030.

Verifies:
1. Static asset serving (index.html, style.css, script.js)
2. Backend health and model checkpoint statuses
3. Modality uploads with validation & error handling
4. Real PyTorch model inferences (Image, Video, Audio)
5. Grad-CAM visual heatmap generation
6. Multi-Modal probability score fusion
7. SQLite audit logging and record retrieval
"""

import os
import sys
import glob
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.history.db import get_history, get_history_by_id

client = TestClient(app)

def run_e2e_validation():
    print("=====================================================")
    print(" DEEPFAKE SENTINEL 2030 - END-TO-END FLOW VALIDATION ")
    print("=====================================================")

    results = {}

    # 1. Static Assets
    print("\n--- 1. STATIC ASSETS & FRONTEND SERVING ---")
    r_index = client.get("/")
    r_css = client.get("/style.css")
    r_js = client.get("/script.js")
    
    index_pass = r_index.status_code == 200 and "SENTINEL" in r_index.text
    css_pass = r_css.status_code == 200 and "DeepFake Sentinel" in r_css.text
    js_pass = r_js.status_code == 200 and "API_BASE" in r_js.text

    print(f" * GET / (index.html): [{'PASS' if index_pass else 'FAIL'}] HTTP {r_index.status_code}")
    print(f" * GET /style.css:     [{'PASS' if css_pass else 'FAIL'}] HTTP {r_css.status_code}")
    print(f" * GET /script.js:     [{'PASS' if js_pass else 'FAIL'}] HTTP {r_js.status_code}")
    results["static_frontend"] = index_pass and css_pass and js_pass

    # 2. Health & Models
    print("\n--- 2. SYSTEM HEALTH & CHECKPOINT AVAILABILITY ---")
    r_health = client.get("/health")
    health_pass = r_health.status_code == 200 and r_health.json().get("status") == "ok"
    print(f" * Health Endpoint:     [{'PASS' if health_pass else 'FAIL'}] {r_health.json()}")

    r_models = client.get("/model-status")
    models_json = r_models.json()
    all_models_trained = all(m["status"] == "TRAINED" for m in models_json.get("models", []))
    print(f" * Checkpoint Status:  [{'PASS' if all_models_trained else 'FAIL'}]")
    for m in models_json.get("models", []):
        print(f"   - {m['modality'].upper()}: {m['name']} -> {m['status']} ({m['checkpoint_path']})")
    results["model_checkpoints"] = all_models_trained

    # 3. Image Detection & Grad-CAM
    print("\n--- 3. IMAGE DETECTION & GRAD-CAM FLOW ---")
    # Error handling check
    r_bad_img = client.post("/predict/image", files={"file": ("test.txt", b"invalid text data", "text/plain")})
    img_err_pass = r_bad_img.status_code == 400
    print(f" * Invalid Format Rejection: [{'PASS' if img_err_pass else 'FAIL'}] HTTP {r_bad_img.status_code}")

    # Real Inference check
    test_imgs = glob.glob("datasets/image/test/*/*.jpg") + glob.glob("datasets/image/test/*/*.png")
    test_img_path = test_imgs[0] if test_imgs else None
    img_res_data = None
    if test_img_path and os.path.isfile(test_img_path):
        with open(test_img_path, "rb") as f:
            r_img = client.post("/predict/image", files={"file": (os.path.basename(test_img_path), f.read(), "image/jpeg")})
        if r_img.status_code == 200:
            img_res_data = r_img.json()
            gradcam_present = "gradcam" in img_res_data and len(img_res_data["gradcam"]) > 50
            print(f" * Image Inference:          [PASS] Label: {img_res_data.get('label')} | Conf: {round(img_res_data.get('confidence',0),4)} | Risk: {img_res_data.get('risk')}")
            print(f" * Grad-CAM Heatmap:         [{'PASS' if gradcam_present else 'FAIL'}] Base64 Data URI length: {len(img_res_data.get('gradcam',''))}")
            results["image_inference"] = True
            results["gradcam"] = gradcam_present
        else:
            print(f" * Image Inference:          [FAIL] HTTP {r_img.status_code}")
            results["image_inference"] = False
            results["gradcam"] = False

    # 4. Video Detection Flow
    print("\n--- 4. VIDEO DETECTION FLOW ---")
    # Error handling check
    r_bad_vid = client.post("/predict/video", files={"file": ("test.exe", b"invalid binary", "application/octet-stream")})
    vid_err_pass = r_bad_vid.status_code == 400
    print(f" * Invalid Format Rejection: [{'PASS' if vid_err_pass else 'FAIL'}] HTTP {r_bad_vid.status_code}")

    test_vids = glob.glob("datasets/video/test/*/*.mp4")
    test_vid_path = test_vids[0] if test_vids else None
    vid_res_data = None
    if test_vid_path and os.path.isfile(test_vid_path):
        with open(test_vid_path, "rb") as f:
            r_vid = client.post("/predict/video", files={"file": (os.path.basename(test_vid_path), f.read(), "video/mp4")})
        if r_vid.status_code == 200:
            vid_res_data = r_vid.json()
            frames_ok = len(vid_res_data.get("sampled_frames", [])) == 16
            print(f" * Video Inference:          [PASS] Label: {vid_res_data.get('label')} | Conf: {round(vid_res_data.get('confidence',0),4)} | Risk: {vid_res_data.get('risk')}")
            print(f" * 16 Uniform Frame Decimation: [{'PASS' if frames_ok else 'FAIL'}] Total Frames Sampled: {len(vid_res_data.get('sampled_frames', []))}")
            results["video_inference"] = True
        else:
            print(f" * Video Inference:          [FAIL] HTTP {r_vid.status_code}")
            results["video_inference"] = False

    # 5. Audio Detection Flow
    print("\n--- 5. AUDIO DETECTION FLOW ---")
    # Error handling check
    r_bad_aud = client.post("/predict/audio", files={"file": ("test.pdf", b"pdf stream", "application/pdf")})
    aud_err_pass = r_bad_aud.status_code == 400
    print(f" * Invalid Format Rejection: [{'PASS' if aud_err_pass else 'FAIL'}] HTTP {r_bad_aud.status_code}")

    test_auds = glob.glob("datasets/audio/test/*/*.wav")
    test_aud_path = test_auds[0] if test_auds else None
    aud_res_data = None
    if test_aud_path and os.path.isfile(test_aud_path):
        with open(test_aud_path, "rb") as f:
            r_aud = client.post("/predict/audio", files={"file": (os.path.basename(test_aud_path), f.read(), "audio/wav")})
        if r_aud.status_code == 200:
            aud_res_data = r_aud.json()
            print(f" * Audio Inference:          [PASS] Label: {aud_res_data.get('label')} | Conf: {round(aud_res_data.get('confidence',0),4)} | Risk: {aud_res_data.get('risk')}")
            results["audio_inference"] = True
        else:
            print(f" * Audio Inference:          [FAIL] HTTP {r_aud.status_code}")
            results["audio_inference"] = False

    # 6. Multi-Modal Fusion Flow
    print("\n--- 6. MULTI-MODAL FUSION FLOW ---")
    fusion_payload = {
        "image": img_res_data,
        "video": vid_res_data,
        "audio": aud_res_data
    }
    r_fuse = client.post("/predict/fusion", json=fusion_payload)
    if r_fuse.status_code == 200:
        fuse_res = r_fuse.json()
        fuse_pass = "final_label" in fuse_res and len(fuse_res.get("modalities_used", [])) == 3
        print(f" * Multi-Modal Fusion:       [{'PASS' if fuse_pass else 'FAIL'}] Final Label: {fuse_res.get('final_label')} | Confidence: {fuse_res.get('final_confidence')} | Risk: {fuse_res.get('risk_level')} (Score: {fuse_res.get('risk_score')}/100)")
        print(f" * Contributing Modalities:  {fuse_res.get('modalities_used')}")
        results["multimodal_fusion"] = fuse_pass
    else:
        print(f" * Multi-Modal Fusion:       [FAIL] HTTP {r_fuse.status_code}")
        results["multimodal_fusion"] = False

    # 7. SQLite History Flow
    print("\n--- 7. SQLITE DATABASE AUDIT LOGGING ---")
    r_hist = client.get("/history?limit=10")
    if r_hist.status_code == 200:
        hist_data = r_hist.json()
        count = hist_data.get("count", 0)
        hist_pass = count > 0 and len(hist_data.get("history", [])) > 0
        print(f" * History Retrieval:        [{'PASS' if hist_pass else 'FAIL'}] Total Records: {count}")
        if hist_pass:
            first = hist_data["history"][0]
            print(f"   - Latest Log: ID={first['id']} | File={first['filename']} | Media={first['media_type']} | Label={first['label']} | Conf={round(first['confidence'],4)} | Risk={first['risk']}")
        results["sqlite_history"] = hist_pass
    else:
        print(f" * History Retrieval:        [FAIL] HTTP {r_hist.status_code}")
        results["sqlite_history"] = False

    print("\n=====================================================")
    print("               FINAL VALIDATION SUMMARY              ")
    print("=====================================================")
    for k, v in results.items():
        print(f" - {k.replace('_', ' ').upper()}: {'PASS' if v else 'FAIL'}")
    print("=====================================================")

if __name__ == "__main__":
    run_e2e_validation()
