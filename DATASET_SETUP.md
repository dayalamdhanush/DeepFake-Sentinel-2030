# DeepFake Sentinel 2030 — Dataset Setup & Preparation Guide

This guide explains how to acquire, inspect, and structure real datasets for training and evaluating **DeepFake Sentinel 2030** without data leakage or file corruption.

---

## 1. Required Standardized Dataset Hierarchy

The machine learning pipelines expect the following standardized directory tree under `datasets/`:

```text
datasets/
├── image/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── video/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
└── audio/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

---

## 2. Where to Place Raw Downloaded Datasets

Do **not** extract raw datasets directly into `datasets/`. Instead, download and extract them to an external or temporary staging folder, e.g.:

- `data_raw/image/` (e.g. FaceForensics++ Face crops, Celeb-DF frames, 140k Real/Fake Faces)
- `data_raw/video/` (e.g. FaceForensics++ videos, Celeb-DF v2 `.mp4` videos, DFDC sample videos)
- `data_raw/audio/` (e.g. ASVspoof 2019/2021 logical access, WaveFake `.wav` files)

---

## 3. Step 1: Inspecting the Raw Dataset

Before preparation or splitting, use `scripts/inspect_dataset.py` to check file counts, detect corrupted files, identify empty folders, and verify class balance:

```powershell
# Basic folder hierarchy & class distribution check
python scripts/inspect_dataset.py --data_dir data_raw/image

# Deep integrity scan (verifies PIL decodability for images, OpenCV for video, soundfile/librosa for audio)
python scripts/inspect_dataset.py --data_dir data_raw/image --check_integrity
```

---

## 4. Step 2: Running the Dataset Preparation Pipeline

The preparation scripts convert raw datasets into the `datasets/` structure with deterministic train/val/test splits (default: 70% Train, 15% Validation, 15% Test).

### 🖼️ Image Dataset Preparation
```powershell
python scripts/prepare_image_dataset.py `
  --source_dir data_raw/image `
  --output_dir datasets/image `
  --train_ratio 0.70 `
  --val_ratio 0.15 `
  --test_ratio 0.15 `
  --seed 42
```
*Optional `--link` flag creates NTFS hardlinks instead of copies to save disk space on Windows.*

---

### 🎥 Video Dataset Preparation (Identity Grouping)
To prevent **identity/subject leakage** (where the same person appears in both training and testing sets), the video script groups videos sharing identity IDs (e.g. `id0123_...`, `person01_...`):

```powershell
python scripts/prepare_video_dataset.py `
  --source_dir data_raw/video `
  --output_dir datasets/video `
  --train_ratio 0.70 `
  --val_ratio 0.15 `
  --test_ratio 0.15 `
  --seed 42
```

---

### 🎙️ Audio Dataset Preparation (Speaker Grouping)
The audio script validates all audio files using `soundfile`/`librosa` and groups by speaker/session:

```powershell
python scripts/prepare_audio_dataset.py `
  --source_dir data_raw/audio `
  --output_dir datasets/audio `
  --train_ratio 0.70 `
  --val_ratio 0.15 `
  --test_ratio 0.15 `
  --seed 42
```

---

## 5. Avoiding Data Leakage Checklist

1. **Identity / Speaker Isolation**: Keep all frames or audio clips originating from the same person or synthetic generator strictly inside **one split** (Train, Validation, or Test).
2. **Fixed Random Seed**: Always maintain `--seed 42` so dataset splits are deterministic and reproducible.
3. **No Overlapping Source Files**: The preparation scripts automatically separate files into distinct directory trees without data collision.

---

## 6. Recommended Benchmark Datasets (Legally Obtainable)

- **Images**:
  - *140k Real and Fake Faces* (Kaggle Academic Dataset)
  - *FaceForensics++* (Extracted pristine and manipulated face crops)
- **Videos**:
  - *Celeb-DF (v2)* (High-quality deepfake video dataset)
  - *FaceForensics++* (DeepFakes, Face2Face, FaceSwap, NeuralTextures)
  - *Deepfake Detection Challenge (DFDC)* (Preview / Public dataset)
- **Audio**:
  - *ASVspoof 2019 / 2021* (Logical Access partition)
  - *WaveFake* (Multi-architecture synthetic voice dataset)
