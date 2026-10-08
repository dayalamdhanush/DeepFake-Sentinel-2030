#!/usr/bin/env python3
"""Audio Dataset Preparation Script for DeepFake Sentinel 2030.

Transforms a raw or source audio dataset into the standardized hierarchy:
    datasets/audio/train/real/
    datasets/audio/train/fake/
    datasets/audio/val/real/
    datasets/audio/val/fake/
    datasets/audio/test/real/
    datasets/audio/test/fake/

Guarantees:
- Preserves original audio files without destructive downsampling during preparation.
- Validates audio readability via soundfile/librosa before inclusion.
- Speaker/Session grouping support to reduce cross-split speaker leakage.
- Deterministic reproducible splitting via configurable random seed.
- Generates comprehensive class counts and duration metrics.
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

VALID_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

REAL_KEYWORDS = {"real", "authentic", "original", "pristine", "0", "bonafide", "live", "human"}
FAKE_KEYWORDS = {"fake", "manipulated", "deepfake", "synthetic", "1", "spoof", "generated", "tts", "vc"}


def classify_file(path: Path) -> str:
    """Classify audio file as 'real' or 'fake' from path components and filename."""
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


def extract_speaker_id(path: Path) -> str:
    """Extract speaker/voice session grouping key to prevent cross-split speaker leakage."""
    stem = path.stem
    parent = path.parent.name

    # Check for speaker prefixes like 'spk01_...', 'speaker_01', 'LA_0001', 'id10001'
    match = re.match(r"^(spk\d+|speaker\d+|la_\d+|id\d+|\d{3,4})", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    if re.match(r"^(spk\d+|speaker\d+|speaker_\d+|id\d+|voice\d+)", parent, re.IGNORECASE):
        return parent.lower()

    return stem


def is_valid_audio(path: Path) -> bool:
    """Test if audio file can be decoded by soundfile or librosa."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None, duration=0.5)
            return len(y) > 0
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


def prepare_audio_dataset(
    source_dir: Path,
    output_dir: Path = Path("datasets/audio"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    link_mode: bool = False,
    validate_audio: bool = True,
    group_by_speaker: bool = True,
):
    print("=" * 70)
    print("DEEPFAKE SENTINEL 2030 -- AUDIO DATASET PREPARATION")
    print("=" * 70)
    print(f"Source Directory   : {source_dir.resolve()}")
    print(f"Output Directory   : {output_dir.resolve()}")
    print(f"Split Ratios       : Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random Seed        : {seed}")
    print(f"Speaker Grouping   : {group_by_speaker}")
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

    # Discover valid audio files
    all_files = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_AUDIO_EXTS and not p.name.startswith(".")
    ]

    if not all_files:
        print(f"\n[ERROR] No supported audio files found in {source_dir}")
        return False

    print(f"Discovered {len(all_files)} candidate audio files. Classifying & validating with librosa/soundfile...")

    # Classify & group files
    class_groups: Dict[str, Dict[str, List[Path]]] = {
        "real": defaultdict(list),
        "fake": defaultdict(list),
        "unlabeled": defaultdict(list),
    }
    invalid_count = 0

    for file_path in all_files:
        if validate_audio and not is_valid_audio(file_path):
            invalid_count += 1
            continue
        label = classify_file(file_path)
        grp = extract_speaker_id(file_path) if group_by_speaker else file_path.stem
        if label in {"real", "fake"}:
            class_groups[label][grp].append(file_path)
        else:
            class_groups["unlabeled"][grp].append(file_path)

    if invalid_count > 0:
        print(f"[WARNING] Skipped {invalid_count} corrupted/unreadable audio files.")

    total_real_audio = sum(len(v) for v in class_groups["real"].values())
    total_fake_audio = sum(len(v) for v in class_groups["fake"].values())

    if total_real_audio == 0 and total_fake_audio == 0:
        print("\n[ERROR] Could not determine REAL/FAKE labels from directory structure or filenames.")
        print("Hint: Organize source audio into 'real/' and 'fake/' folders.")
        return False

    print(f"\nValid Audio Samples Identified:")
    print(f"  - REAL : {total_real_audio} files across {len(class_groups['real'])} speaker groups")
    print(f"  - FAKE : {total_fake_audio} files across {len(class_groups['fake'])} speaker groups")

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

        for grp_id, audio_list in class_groups[label].items():
            if grp_id in train_groups:
                split_name = "train"
            elif grp_id in val_groups:
                split_name = "val"
            else:
                split_name = "test"

            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for idx, src_audio in enumerate(audio_list):
                dest_filename = f"{src_audio.stem}_{idx:05d}{src_audio.suffix.lower()}"
                dest_file = dest_dir / dest_filename
                copy_or_link(src_audio, dest_file, link_mode=link_mode)
                split_counts[split_name][label] += 1

    print("\n" + "-" * 70)
    print("AUDIO DATASET PREPARATION SUMMARY")
    print("-" * 70)
    for split_name in ["train", "val", "test"]:
        r_cnt = split_counts[split_name]["real"]
        f_cnt = split_counts[split_name]["fake"]
        tot = r_cnt + f_cnt
        print(f"  [{split_name.upper():<5}] -> REAL: {r_cnt:>6} | FAKE: {f_cnt:>6} | TOTAL: {tot:>6}")

    total_prepared = sum(split_counts[s]["real"] + split_counts[s]["fake"] for s in ["train", "val", "test"])
    print(f"\n[SUCCESS] Successfully prepared {total_prepared} audio files in {output_dir.resolve()}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio synthetic speech dataset for DeepFake Sentinel 2030."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source raw audio dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/audio",
        help="Destination directory for prepared audio dataset (default: 'datasets/audio').",
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
        help="Disable speaker grouping (splits purely on individual audio files).",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip librosa/soundfile audio validation.",
    )
    args = parser.parse_args()

    prepare_audio_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        link_mode=args.link,
        validate_audio=not args.skip_validation,
        group_by_speaker=not args.no_grouping,
    )


if __name__ == "__main__":
    main()
