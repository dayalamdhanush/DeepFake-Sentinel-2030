# DeepFake Sentinel 2030

## Overview
DeepFake Sentinel 2030 is a full‑stack explainable multi‑modal deepfake detection system. It detects manipulated **images**, **videos**, and **audio** using state‑of‑the‑art deep learning models and provides:
- Real / Deepfake classification
- Confidence score
- Risk level
- Grad‑CAM visual explanations for images
- Multi‑modal fusion for combined decisions

The project is organized as a **backend** FastAPI service and a **vanilla HTML/JS** frontend. All inference code runs on CPU by default and automatically uses CUDA if available.

## Repository Layout
```
deepfake-sentinel-2030/
├─ backend/                # FastAPI service
│   ├─ main.py
│   └─ api/
│       ├─ image.py
│       ├─ video.py
│       └─ audio.py
├─ frontend/               # Simple UI (HTML/CSS/JS)
│   ├─ index.html
│   ├─ style.css
│   └─ script.js
├─ ml/                     # Model definitions & utilities
│   └─ image/
│       ├─ model.py
│       ├─ predict.py
│       ├─ train.py
│       └─ gradcam.py
├─ scripts/                # Training scripts
│   └─ train_image.py
├─ models/                 # Trained model checkpoints (to be added later)
├─ datasets/               # Expected dataset structure (user‑provided)
├─ Dockerfile              # Backend container
├─ docker-compose.yml
├─ requirements.txt        # Backend Python deps
└─ README.md               # This file
```

## Setup & Run
### Prerequisites
- Python 3.10+ (recommended via `uv` or `venv`)
- Docker (optional, see Dockerfile)
- No GPU required – the code will fall back to CPU.

### Install Python deps
```bash
pip install -r requirements.txt
```

### Start the backend (development)
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### Run the frontend
Open `frontend/index.html` in a browser.

### Docker (optional)
```bash
docker compose up --build
```

## Model Training
Training scripts are provided under `scripts/`. Example for the image model:
```bash
python scripts/train_image.py --data_dir datasets/image --epochs 20 --batch_size 32
```
The script will save `models/efficientnet_b0_deepfake.pth`. If no model file exists, the API will reply with *"Model not trained"*.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict/image` | Upload an image and receive classification, confidence, risk, and Grad‑CAM heatmap (base64). |
| POST | `/predict/video` | Upload a video file (MP4). Returns classification and confidence (model placeholder). |
| POST | `/predict/audio` | Upload an audio file (WAV). Returns classification and confidence (model placeholder). |
| POST | `/predict/fusion` | Submit JSON with any combination of modalities and receive fused decision. |

## License
MIT – feel free to adapt for academic projects.
