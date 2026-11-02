# backend/services/audio_service.py
"""Audio processing service for DeepFake Sentinel 2030.

All preprocessing is delegated to ml.audio.preprocessing, which is the
single source of truth.  This module provides thin wrappers used by the
backend API so that existing import paths continue to work unchanged.

Preprocessing parameters (defined in ml.audio.preprocessing):
  - 16 kHz mono audio
  - 3-second non-overlapping chunks (all chunks, not just 5)
  - 128-bin log-Mel spectrogram (n_fft=1024, hop_length=512)
  - Fixed 128-frame time axis (pad/truncate)
  - Per-spectrogram mean/std normalization
  - FAKE=0, REAL=1
"""

import sys
import os
from pathlib import Path

import torch

# Ensure project root is on sys.path so ml.audio.preprocessing is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.audio.preprocessing import (
    audio_path_to_tensor,
    # Re-export constants so any code that imported them from here still works
    SR, N_MELS, N_FFT, HOP_LENGTH, FIXED_FRAMES,
    CHUNK_DURATION, CHUNK_SAMPLES,
    CLASS_TO_IDX, IDX_TO_CLASS, FAKE_IDX, REAL_IDX,
    aggregate_logits,
)

MAX_LEN = FIXED_FRAMES   # alias for backward compatibility


def load_audio_tensor(audio_path: str) -> torch.Tensor:
    """
    Load audio file and return normalized log-Mel spectrogram tensor.

    Returns
    -------
    torch.Tensor  shape (num_chunks, 1, 128, 128)
      All non-overlapping 3-second chunks are returned (no cap at 5).
      Short audio (< 3s) is zero-padded to produce exactly 1 chunk.
    """
    if not Path(audio_path).is_file():
        raise ValueError(f"Audio file not found: {audio_path}")
    try:
        return audio_path_to_tensor(audio_path)
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")


def prepare_batch(audio_path: str, device: torch.device) -> torch.Tensor:
    """
    Convenience wrapper — loads spectrogram chunks and moves to device.

    Returns
    -------
    torch.Tensor  shape (num_chunks, 1, 128, 128) on `device`
    """
    return load_audio_tensor(audio_path).to(device)
