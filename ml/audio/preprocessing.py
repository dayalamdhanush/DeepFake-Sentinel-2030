"""
ml/audio/preprocessing.py
==========================
Single source of truth for ALL audio preprocessing in DeepFake Sentinel 2030.
Used by: training, evaluation, inference, and backend service.

CLASS MAPPING (explicit, fixed everywhere):
  FAKE = 0
  REAL = 1
"""

import os
import numpy as np
import librosa
import torch

# ── Audio constants ────────────────────────────────────────────────────────────
SR            = 16_000          # Sample rate (Hz)
N_MELS        = 128             # Mel bins
N_FFT         = 1024            # FFT window size
HOP_LENGTH    = 512             # Hop length
FIXED_FRAMES  = 128             # Time-frames per spectrogram after pad/truncate
CHUNK_DURATION = 3.0            # Seconds per chunk
CHUNK_SAMPLES  = int(SR * CHUNK_DURATION)  # 48 000 samples

# ── Class mapping ──────────────────────────────────────────────────────────────
CLASS_TO_IDX = {"fake": 0, "real": 1}
IDX_TO_CLASS = {0: "fake", 1: "real"}
FAKE_IDX = 0
REAL_IDX = 1


# ── Core helpers ───────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """
    Load an audio file at SR=16 kHz, mono.
    Returns a 1-D float32 array.
    Raises FileNotFoundError if path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def audio_to_chunks(path: str) -> list:
    """
    Load audio and split into non-overlapping 3-second chunks.

    Rules
    -----
    * If the recording is shorter than CHUNK_SAMPLES, pad with zeros and
      return ONE chunk.  (No recording is silently discarded.)
    * Otherwise split into floor(len / CHUNK_SAMPLES) chunks; any trailing
      samples shorter than one full chunk are padded to form a final chunk.

    Returns
    -------
    list[np.ndarray]  – each element is shape (CHUNK_SAMPLES,)
    """
    y = load_audio(path)

    # Pad short recordings to at least one chunk
    if len(y) < CHUNK_SAMPLES:
        y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="constant")

    n_full   = len(y) // CHUNK_SAMPLES
    chunks   = [y[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                for i in range(n_full)]

    # Handle trailing samples (not exactly divisible)
    remainder = len(y) - n_full * CHUNK_SAMPLES
    if remainder > 0:
        tail = y[n_full * CHUNK_SAMPLES:]
        tail = np.pad(tail, (0, CHUNK_SAMPLES - len(tail)), mode="constant")
        chunks.append(tail)

    return chunks


def chunk_to_spectrogram(chunk: np.ndarray) -> np.ndarray:
    """
    Convert a single CHUNK_SAMPLES-length audio chunk to a
    normalized log-Mel spectrogram.

    Output shape: (1, N_MELS, FIXED_FRAMES)  i.e. (1, 128, 128)

    Steps
    -----
    1. Compute mel spectrogram
    2. Convert to dB (power_to_db)
    3. Truncate or zero-pad time axis to FIXED_FRAMES
    4. Per-spectrogram mean/std normalization
    """
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # shape (N_MELS, T)

    # Pad or truncate time axis
    T = log_mel.shape[1]
    if T >= FIXED_FRAMES:
        log_mel = log_mel[:, :FIXED_FRAMES]
    else:
        pad_width = FIXED_FRAMES - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")

    # Normalize
    mean = log_mel.mean()
    std  = log_mel.std()
    log_mel = (log_mel - mean) / (std + 1e-8)

    return log_mel[np.newaxis, :, :]   # (1, 128, 128)


def audio_path_to_tensor(path: str) -> torch.Tensor:
    """
    Load audio and return a tensor of all chunk spectrograms.

    Returns
    -------
    torch.Tensor  shape (N_chunks, 1, N_MELS, FIXED_FRAMES)
                  = (N_chunks, 1, 128, 128)
    """
    chunks = audio_to_chunks(path)
    specs  = [chunk_to_spectrogram(c) for c in chunks]
    arr    = np.stack(specs, axis=0)          # (N, 1, 128, 128)
    return torch.from_numpy(arr).float()


def aggregate_logits(logits: torch.Tensor, threshold: float = 0.50) -> dict:
    """
    Aggregate per-chunk logits to a single file-level prediction.

    Parameters
    ----------
    logits : torch.Tensor  shape (N_chunks, 2)
    threshold : float      decision threshold for fake_probability (default: 0.50)

    Returns
    -------
    dict with keys:
        label          – 'fake' or 'real'
        fake_probability
        real_probability
        confidence
        predicted_class_idx
        threshold_used
    """
    avg_logits = logits.mean(dim=0)                    # (2,)
    probs      = torch.softmax(avg_logits, dim=0)      # (2,)
    fake_prob  = probs[FAKE_IDX].item()
    real_prob  = probs[REAL_IDX].item()

    if fake_prob >= threshold:
        pred_idx = FAKE_IDX
    else:
        pred_idx = REAL_IDX

    label      = IDX_TO_CLASS[pred_idx]
    confidence = max(fake_prob, real_prob)

    return {
        "label":               label,
        "fake_probability":    round(fake_prob,  4),
        "real_probability":    round(real_prob,  4),
        "confidence":          round(confidence, 4),
        "predicted_class_idx": pred_idx,
        "threshold_used":      round(threshold,  4),
    }
