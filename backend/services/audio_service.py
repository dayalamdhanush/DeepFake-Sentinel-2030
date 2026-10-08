# backend/services/audio_service.py
"""Audio processing service for DeepFake Sentinel 2030.

- Loads an audio file with librosa.
- Resamples to 16 kHz.
- Generates a log‑Mel spectrogram (128 mel bins).
- Normalizes to zero‑mean, unit‑variance (per‑sample).
- Returns a torch tensor of shape (1, 1, 128, time_frames).
"""

import os
import numpy as np
import torch
import librosa
from pathlib import Path

# Constants for spectrogram generation
SR = 16000  # target sampling rate
N_MELS = 128
HOP_LENGTH = 512
WINDOW = "hann"

def load_audio_tensor(audio_path: str) -> torch.Tensor:
    """Load ``audio_path`` and return a normalized log‑Mel spectrogram tensor.

    Returns ``(1, 1, N_MELS, T)`` where ``T`` depends on audio length.
    Raises ``ValueError`` on failure.
    """
    if not Path(audio_path).is_file():
        raise ValueError("Audio file not found")
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        raise ValueError(f"Librosa failed to load audio: {e}")
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, window=WINDOW)
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize to zero mean / unit variance (per‑sample)
    mean = log_mel.mean()
    std = log_mel.std() if log_mel.std() > 0 else 1.0
    norm_log_mel = (log_mel - mean) / std
    # Convert to tensor (1, 1, N_MELS, T)
    tensor = torch.from_numpy(norm_log_mel).unsqueeze(0).unsqueeze(0).float()
    return tensor

def prepare_batch(audio_path: str, device: torch.device) -> torch.Tensor:
    """Convenience wrapper that loads the spectrogram and moves it to ``device``.
    Returns ``(1, 1, N_MELS, T)``.
    """
    tensor = load_audio_tensor(audio_path)
    return tensor.to(device)
