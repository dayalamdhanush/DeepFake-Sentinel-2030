#!/usr/bin/env python3
"""Dataset Inspection Tool for DeepFake Sentinel 2030.

Safely inspects any local dataset folder without modifying source files.
Features:
- Detects folder hierarchy and split structures (train/val/test).
- Identifies REAL vs FAKE sample distributions.
- Categorizes supported vs unsupported file extensions.
- Detects empty directories.
- Optional deep integrity scan (PIL for images, OpenCV for videos, soundfile/librosa for audio).
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALL_SUPPORTED = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

REAL_KEYWORDS = {"real", "authentic", "original", "pristine", "0", "bonafide", "live"}
FAKE_KEYWORDS = {"fake", "manipulated", "deepfake", "synthetic", "1", "spoof", "generated"}


def classify_label(path: Path) -> str:
    """Infer class label ('REAL', 'FAKE', or 'UNKNOWN') from path components."""
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts):
        if part in REAL_KEYWORDS:
            return "REAL"
        if part in FAKE_KEYWORDS:
            return "FAKE"
    # Check filename prefix/suffix
    stem = path.stem.lower()
    for kw in REAL_KEYWORDS:
        if kw in stem:
            return "REAL"
    for kw in FAKE_KEYWORDS:
        if kw in stem:
            return "FAKE"
    return "UNKNOWN"


def infer_modality(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "unsupported"


def check_file_integrity(path: Path, modality: str) -> Tuple[bool, str]:
    """Test if file can actually be opened and parsed by respective library."""
    try:
        if modality == "image":
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            return True, "OK"
        elif modality == "video":
            import cv2
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False, "Cannot open video stream"
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return False, "Failed to read first frame"
            return True, "OK"
        elif modality == "audio":
            import soundfile as sf
            try:
                info = sf.info(str(path))
                if info.frames <= 0:
                    return False, "Audio has 0 frames"
                return True, "OK"
            except Exception:
                # Fallback check with librosa or wave
                import librosa
                y, sr = librosa.load(str(path), sr=None, duration=0.5)
                if len(y) == 0:
                    return False, "Loaded audio waveform is empty"
                return True, "OK"
        return True, "Not scanned"
    except Exception as e:
        return False, str(e)


def inspect_dataset(data_dir: Path, check_integrity: bool = False, max_corrupt_report: int = 10):
    print("=" * 70)
    print("DEEPFAKE SENTINEL 2030 -- DATASET INSPECTOR")
    print(f"Target Directory: {data_dir.resolve()}")
    print("=" * 70)

    if not data_dir.exists():
        print(f"\n[ERROR] Path does not exist: {data_dir}")
        print("Status: DATASET NOT AVAILABLE -- TRAINING NOT PERFORMED")
        return False

    if not data_dir.is_dir():
        print(f"\n[ERROR] Path is not a directory: {data_dir}")
        return False

    all_paths = list(data_dir.rglob("*"))
    all_dirs = [p for p in all_paths if p.is_dir()]
    all_files = [p for p in all_paths if p.is_file() and not p.name.startswith(".")]

    if not all_files:
        print("\n[WARNING] Directory contains 0 files.")
        print("Status: DATASET NOT AVAILABLE -- TRAINING NOT PERFORMED")
        return False

    empty_dirs = [d for d in all_dirs if not any(d.iterdir())]

    # Metrics aggregation
    extension_counts = Counter()
    label_counts = Counter()
    modality_counts = Counter()
    split_counts = defaultdict(lambda: Counter())
    unsupported_files = []
    corrupted_files = []
    total_bytes = 0

    known_splits = {"train", "val", "validation", "valid", "test", "dev"}

    print(f"\nScanning {len(all_files)} files across {len(all_dirs)} subdirectories...")

    for file_path in all_files:
        total_bytes += file_path.stat().st_size
        ext = file_path.suffix.lower()
        extension_counts[ext or "[no_ext]"] += 1

        modality = infer_modality(file_path)
        modality_counts[modality] += 1

        if modality == "unsupported":
            unsupported_files.append(file_path)

        label = classify_label(file_path)
        label_counts[label] += 1

        # Detect split if path contains train/val/test
        split_name = "unsplit"
        for part in file_path.parts:
            part_lower = part.lower()
            if part_lower in known_splits:
                split_name = "val" if part_lower in {"validation", "valid", "dev"} else part_lower
                break
        split_counts[split_name][label] += 1

        # Optional integrity check
        if check_integrity and modality in {"image", "video", "audio"}:
            is_valid, reason = check_file_integrity(file_path, modality)
            if not is_valid:
                corrupted_files.append((file_path, reason))

    # Display Summary
    total_size_mb = total_bytes / (1024 * 1024)
    print("\n" + "-" * 70)
    print("1. SUMMARY OVERVIEW")
    print("-" * 70)
    print(f"Total Files Found      : {len(all_files)}")
    print(f"Total Disk Size        : {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")
    print(f"Empty Directories      : {len(empty_dirs)}")
    print(f"Supported Media Files  : {sum(modality_counts[m] for m in ['image', 'video', 'audio'])}")
    print(f"Unsupported / Misc     : {modality_counts['unsupported']}")

    print("\n" + "-" * 70)
    print("2. CLASS DISTRIBUTION (REAL vs FAKE)")
    print("-" * 70)
    for lbl, cnt in label_counts.most_common():
        pct = (cnt / len(all_files)) * 100
        print(f"  - {lbl:<10} : {cnt:>6} samples ({pct:>5.1f}%)")

    print("\n" + "-" * 70)
    print("3. MODALITY BREAKDOWN")
    print("-" * 70)
    for mod in ["image", "video", "audio", "unsupported"]:
        cnt = modality_counts[mod]
        pct = (cnt / len(all_files)) * 100 if all_files else 0
        print(f"  - {mod.capitalize():<12} : {cnt:>6} files ({pct:>5.1f}%)")

    print("\n" + "-" * 70)
    print("4. SPLIT HIERARCHY")
    print("-" * 70)
    for s_name, counts in sorted(split_counts.items()):
        sub_total = sum(counts.values())
        print(f"  Split: [{s_name.upper()}] (Total: {sub_total})")
        for l_name, l_cnt in counts.items():
            print(f"    - {l_name}: {l_cnt}")

    print("\n" + "-" * 70)
    print("5. FILE EXTENSIONS FOUND")
    print("-" * 70)
    for ext, cnt in extension_counts.most_common():
        status = "SUPPORTED" if ext in ALL_SUPPORTED else "UNSUPPORTED"
        print(f"  {ext:<12} : {cnt:>6} files  [{status}]")

    if empty_dirs:
        print("\n" + "-" * 70)
        print("6. EMPTY DIRECTORIES DETECTED")
        print("-" * 70)
        for ed in empty_dirs[:10]:
            print(f"  - {ed.relative_to(data_dir)}")
        if len(empty_dirs) > 10:
            print(f"  ... and {len(empty_dirs) - 10} more empty folders.")

    if check_integrity:
        print("\n" + "-" * 70)
        print("7. INTEGRITY SCAN RESULTS")
        print("-" * 70)
        if not corrupted_files:
            print("  [SUCCESS] All scanned files passed integrity validation!")
        else:
            print(f"  [ALERT] {len(corrupted_files)} corrupted/unreadable file(s) found:")
            for p, r in corrupted_files[:max_corrupt_report]:
                print(f"  - {p.relative_to(data_dir)}: {r}")
            if len(corrupted_files) > max_corrupt_report:
                print(f"  ... and {len(corrupted_files) - max_corrupt_report} more corrupted files.")
    else:
        print("\n[NOTE] Deep integrity scan skipped. Use --check_integrity to test file decoding.")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inspect local datasets for DeepFake Sentinel 2030 without modifying them."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Path to the dataset directory to inspect (default: 'datasets')",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Run deep file integrity check to verify decodability of images/videos/audio.",
    )
    args = parser.parse_args()

    target_dir = Path(args.data_dir)
    inspect_dataset(target_dir, check_integrity=args.check_integrity)


if __name__ == "__main__":
    main()
