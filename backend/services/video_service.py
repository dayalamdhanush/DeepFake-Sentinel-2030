# backend/services/video_service.py
"""Video processing service for DeepFake Sentinel 2030.

- Reads a video file using OpenCV
- Samples a fixed number of frames uniformly (default 16)
- Resizes frames to 224x224
- Normalizes with ImageNet statistics
- Returns a torch tensor of shape (1, seq_len, 3, 224, 224)
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms

# ImageNet normalization values
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

def _sample_frame_indices(frame_count: int, num_samples: int = 16) -> list[int]:
    """Return a list of frame indices (0‑based) uniformly spaced.
    If the video has fewer frames than ``num_samples`` we repeat the last frame.
    """
    if frame_count <= 0:
        return []
    if frame_count < num_samples:
        # repeat last frame to reach required count
        indices = list(range(frame_count)) + [frame_count - 1] * (num_samples - frame_count)
        return indices[:num_samples]
    step = frame_count / float(num_samples)
    return [int(step * i) for i in range(num_samples)]

def load_video_frames(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """Load ``num_frames`` uniformly sampled frames from ``video_path``.

    Returns a tensor ``(num_frames, 3, 224, 224)`` ready for model input.
    Raises ``ValueError`` on any failure (e.g., cannot open video).
    """
    if not os.path.isfile(video_path):
        raise ValueError("Video file not found")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no frames")
    indices = _sample_frame_indices(total_frames, num_frames)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # fallback to last successfully read frame
            if frames:
                frame = frames[-1]
            else:
                cap.release()
                raise ValueError(f"Failed to read frame {idx}")
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Apply torchvision transform (to Tensor + Normalization)
        tensor = _transform(frame)  # shape (3, 224, 224)
        frames.append(tensor)
    cap.release()
    video_tensor = torch.stack(frames)  # (num_frames, 3, 224, 224)
    return video_tensor

def prepare_batch(video_path: str, device: torch.device, num_frames: int = 16) -> torch.Tensor:
    """Convenience wrapper that loads frames and adds batch dimension.
    Returns ``(1, num_frames, 3, 224, 224)`` on ``device``.
    """
    video_tensor = load_video_frames(video_path, num_frames=num_frames)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # batch dim
    return video_tensor

# Helper to extract simple metadata for the response
def get_frame_info(video_path: str, num_frames: int = 16) -> list[dict]:
    """Return a list of dictionaries describing sampled frames (index and timestamp).
    Timestamp is approximated using FPS.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_frame_indices(total_frames, num_frames)
    info = []
    for i, idx in enumerate(indices):
        timestamp = idx / fps
        info.append({"frame_index": idx, "timestamp_sec": round(timestamp, 3)})
    cap.release()
    return info
