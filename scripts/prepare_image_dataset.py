#!/usr/bin/env python3
"""Image Dataset Preparation Script for DeepFake Sentinel 2030.

Transforms a raw or source image dataset into the standardized hierarchy:
    datasets/image/train/real/
    datasets/image/train/fake/
    datasets/image/val/real/
    datasets/image/val/fake/
    datasets/image/test/real/
    datasets/image/test/fake/

Guarantees:
- Never modifies or overwrites source data (copies or hardlinks safely).
- Strict data separation to prevent train/test leakage.
- Reproducible deterministic splitting via configurable random seed.
- Validates image integrity using PIL before inclusion.
- Supports flat, binary-subfolder (real/fake), or split-based source directories.
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

REAL_KEYWORDS = {"real", "authentic", "original", "pristine", "0", "bonafide", "live"}
FAKE_KEYWORDS = {"fake", "manipulated", "deepfake", "synthetic", "1", "spoof", "generated"}


def classify_file(path: Path) -> str:
    """Classify file as 'real' or 'fake' from path and filename keywords."""
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts):
        if part in REAL_KEYWORDS:
            return "real"
        if part in FAKE_KEYWORDS:
            return "fake"
    stem = path.stem.lower()
    for kw in REAL_KEYWORDS:
        if kw in stem:
            return "real"
    for kw in FAKE_KEYWORDS:
        if kw in stem:
            return "fake"
    return "unknown"


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def copy_or_link(src: Path, dst: Path, link_mode: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link_mode:
        try:
            os.link(src, dst)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)


def prepare_image_dataset(
    source_dir: Path,
    output_dir: Path = Path("datasets/image"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    link_mode: bool = False,
    validate_images: bool = True,
):
    print("=" * 70)
    print("DEEPFAKE SENTINEL 2030 -- IMAGE DATASET PREPARATION")
    print("=" * 70)
    print(f"Source Directory : {source_dir.resolve()}")
    print(f"Output Directory : {output_dir.resolve()}")
    print(f"Split Ratios     : Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random Seed      : {seed}")
    print(f"Mode             : {'Hardlink' if link_mode else 'Copy'}")
    print("-" * 70)

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"\n[ERROR] Source directory does not exist: {source_dir}")
        print("Status: DATASET NOT AVAILABLE -- TRAINING NOT PERFORMED")
        return False

    # Normalize split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    # Discover valid images
    all_files = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS and not p.name.startswith(".")
    ]

    if not all_files:
        print(f"\n[ERROR] No supported images found in {source_dir}")
        return False

    print(f"Discovered {len(all_files)} candidate image files. Classifying & validating...")

    # Group files by class
    class_files: Dict[str, List[Path]] = defaultdict(list)
    invalid_count = 0

    for file_path in all_files:
        if validate_images and not is_valid_image(file_path):
            invalid_count += 1
            continue
        label = classify_file(file_path)
        if label in {"real", "fake"}:
            class_files[label].append(file_path)
        else:
            class_files["unlabeled"].append(file_path)

    if invalid_count > 0:
        print(f"[WARNING] Skipped {invalid_count} corrupted/unreadable images.")

    if not class_files["real"] and not class_files["fake"]:
        print("\n[ERROR] Could not determine REAL/FAKE labels from directory structure or filenames.")
        print("Hint: Organize source images into 'real/' and 'fake/' subfolders.")
        return False

    print(f"\nValid Samples Identified:")
    print(f"  - REAL : {len(class_files['real'])} images")
    print(f"  - FAKE : {len(class_files['fake'])} images")
    if class_files["unlabeled"]:
        print(f"  - Unlabeled (skipped): {len(class_files['unlabeled'])} images")

    # Set deterministic random seed
    random.seed(seed)

    split_counts = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0},
    }

    # Split and organize each class
    for label in ["real", "fake"]:
        samples = sorted(class_files[label])  # Sort first for deterministic shuffling
        random.shuffle(samples)

        n_total = len(samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        split_map = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        for split_name, file_list in split_map.items():
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for idx, src_file in enumerate(file_list):
                # Ensure unique destination filename
                dest_filename = f"{src_file.stem}_{idx:06d}{src_file.suffix.lower()}"
                dest_file = dest_dir / dest_filename
                copy_or_link(src_file, dest_file, link_mode=link_mode)
                split_counts[split_name][label] += 1

    print("\n" + "-" * 70)
    print("IMAGE DATASET PREPARATION SUMMARY")
    print("-" * 70)
    for split_name in ["train", "val", "test"]:
        r_cnt = split_counts[split_name]["real"]
        f_cnt = split_counts[split_name]["fake"]
        tot = r_cnt + f_cnt
        print(f"  [{split_name.upper():<5}] -> REAL: {r_cnt:>6} | FAKE: {f_cnt:>6} | TOTAL: {tot:>6}")

    total_prepared = sum(split_counts[s]["real"] + split_counts[s]["fake"] for s in ["train", "val", "test"])
    print(f"\n[SUCCESS] Successfully prepared {total_prepared} images in {output_dir.resolve()}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare image deepfake dataset for DeepFake Sentinel 2030."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source raw image dataset directory (containing real/fake folders or files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/image",
        help="Destination directory for prepared dataset (default: 'datasets/image').",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proportion of dataset for training (default: 0.70).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of dataset for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of dataset for testing (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Use hardlinks instead of copying files (saves disk space on same volume).",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip PIL image integrity verification.",
    )
    args = parser.parse_args()

    prepare_image_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        link_mode=args.link,
        validate_images=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
