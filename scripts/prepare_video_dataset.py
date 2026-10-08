#!/usr/bin/env python3
"""Video Dataset Preparation Script for DeepFake Sentinel 2030.

Transforms a raw or source video dataset into the standardized hierarchy:
    datasets/video/train/real/
    datasets/video/train/fake/
    datasets/video/val/real/
    datasets/video/val/fake/
    datasets/video/test/real/
    datasets/video/test/fake/

Guarantees:
- Maintains native video container files (.mp4, .avi, .mov, etc.) without inflating disk space.
- Subject/Identity grouping: groups videos sharing subject IDs to eliminate identity leakage across train/val/test splits.
- Never modifies or overwrites source video files.
- Deterministic reproducible splitting via fixed random seed.
- Validates video header / first frame via OpenCV before inclusion.
"""

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

VALID_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

REAL_KEYWORDS = {"real", "authentic", "original", "pristine", "0", "bonafide", "live"}
FAKE_KEYWORDS = {"fake", "manipulated", "deepfake", "synthetic", "1", "spoof", "generated"}


def classify_file(path: Path) -> str:
    """Classify video file as 'real' or 'fake' from path and filename."""
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


def extract_group_id(path: Path) -> str:
    """Extract identity/subject grouping key to prevent cross-split identity leakage."""
    stem = path.stem
    parent = path.parent.name

    # Check for patterns like 'id01234_...', 'person01_...', '001_002' (FaceForensics / Celeb-DF)
    match = re.match(r"^(id\d+|person\d+|\d{3,4})", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Check parent directory name if it represents an identity
    if re.match(r"^(id\d+|person\d+|\d{3,4}|subject\d+)", parent, re.IGNORECASE):
        return parent.lower()

    # Fallback to file stem itself if no shared identity pattern detected
    return stem


def is_valid_video(path: Path) -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
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


def prepare_video_dataset(
    source_dir: Path,
    output_dir: Path = Path("datasets/video"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    link_mode: bool = False,
    validate_videos: bool = True,
    group_by_subject: bool = True,
):
    print("=" * 70)
    print("DEEPFAKE SENTINEL 2030 -- VIDEO DATASET PREPARATION")
    print("=" * 70)
    print(f"Source Directory   : {source_dir.resolve()}")
    print(f"Output Directory   : {output_dir.resolve()}")
    print(f"Split Ratios       : Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random Seed        : {seed}")
    print(f"Subject Grouping   : {group_by_subject}")
    print(f"Mode               : {'Hardlink' if link_mode else 'Copy'}")
    print("-" * 70)

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"\n[ERROR] Source directory does not exist: {source_dir}")
        print("Status: DATASET NOT AVAILABLE -- TRAINING NOT PERFORMED")
        return False

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    # Discover valid videos
    all_files = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXTS and not p.name.startswith(".")
    ]

    if not all_files:
        print(f"\n[ERROR] No supported video files found in {source_dir}")
        return False

    print(f"Discovered {len(all_files)} candidate video files. Classifying & validating...")

    # Classify & group files
    class_groups: Dict[str, Dict[str, List[Path]]] = {
        "real": defaultdict(list),
        "fake": defaultdict(list),
        "unlabeled": defaultdict(list),
    }
    invalid_count = 0

    for file_path in all_files:
        if validate_videos and not is_valid_video(file_path):
            invalid_count += 1
            continue
        label = classify_file(file_path)
        grp = extract_group_id(file_path) if group_by_subject else file_path.stem
        if label in {"real", "fake"}:
            class_groups[label][grp].append(file_path)
        else:
            class_groups["unlabeled"][grp].append(file_path)

    if invalid_count > 0:
        print(f"[WARNING] Skipped {invalid_count} unreadable video files.")

    total_real_videos = sum(len(v) for v in class_groups["real"].values())
    total_fake_videos = sum(len(v) for v in class_groups["fake"].values())

    if total_real_videos == 0 and total_fake_videos == 0:
        print("\n[ERROR] Could not determine REAL/FAKE labels from directory structure or filenames.")
        print("Hint: Organize source videos into 'real/' and 'fake/' folders.")
        return False

    print(f"\nValid Video Samples Identified:")
    print(f"  - REAL : {total_real_videos} videos across {len(class_groups['real'])} groups/subjects")
    print(f"  - FAKE : {total_fake_videos} videos across {len(class_groups['fake'])} groups/subjects")

    random.seed(seed)

    split_counts = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0},
    }

    # Split groups deterministically
    for label in ["real", "fake"]:
        groups = sorted(class_groups[label].keys())
        random.shuffle(groups)

        n_groups = len(groups)
        n_train_g = max(1, int(n_groups * train_ratio)) if n_groups >= 3 else n_groups
        n_val_g = max(1, int(n_groups * val_ratio)) if n_groups >= 3 else 0
        if n_train_g + n_val_g >= n_groups and n_groups >= 3:
            n_train_g = n_groups - 2
            n_val_g = 1

        train_groups = set(groups[:n_train_g])
        val_groups = set(groups[n_train_g:n_train_g + n_val_g])
        test_groups = set(groups[n_train_g + n_val_g:])

        for grp_id, video_list in class_groups[label].items():
            if grp_id in train_groups:
                split_name = "train"
            elif grp_id in val_groups:
                split_name = "val"
            else:
                split_name = "test"

            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for idx, src_video in enumerate(video_list):
                dest_filename = f"{src_video.stem}_{idx:05d}{src_video.suffix.lower()}"
                dest_file = dest_dir / dest_filename
                copy_or_link(src_video, dest_file, link_mode=link_mode)
                split_counts[split_name][label] += 1

    print("\n" + "-" * 70)
    print("VIDEO DATASET PREPARATION SUMMARY")
    print("-" * 70)
    for split_name in ["train", "val", "test"]:
        r_cnt = split_counts[split_name]["real"]
        f_cnt = split_counts[split_name]["fake"]
        tot = r_cnt + f_cnt
        print(f"  [{split_name.upper():<5}] -> REAL: {r_cnt:>6} | FAKE: {f_cnt:>6} | TOTAL: {tot:>6}")

    total_prepared = sum(split_counts[s]["real"] + split_counts[s]["fake"] for s in ["train", "val", "test"])
    print(f"\n[SUCCESS] Successfully prepared {total_prepared} videos in {output_dir.resolve()}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video deepfake dataset for DeepFake Sentinel 2030."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source raw video dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/video",
        help="Destination directory for prepared video dataset (default: 'datasets/video').",
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
        help="Use hardlinks instead of copying files.",
    )
    parser.add_argument(
        "--no_grouping",
        action="store_true",
        help="Disable subject/identity grouping (splits purely on individual video files).",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip OpenCV video validation.",
    )
    args = parser.parse_args()

    prepare_video_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        link_mode=args.link,
        validate_videos=not args.skip_validation,
        group_by_subject=not args.no_grouping,
    )


if __name__ == "__main__":
    main()
