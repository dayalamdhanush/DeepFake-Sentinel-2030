# backend/services/video_service.py
"""Video processing service for DeepFake Sentinel 2030.

- Reads video files using OpenCV sequential frame decoding (matching training pipeline).
- Uniformly samples 16 frames across the entire video duration.
- Resizes frames to 224x224 RGB.
- Normalizes with ImageNet statistics.
- Returns a torch tensor of shape (1, seq_len=16, 3, 224, 224).
"""

import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

def load_video_frames(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """Load ``num_frames`` uniformly sampled frames from ``video_path``.

    Uses sequential decoding to prevent GOP/keyframe seek corruption in MP4 codecs.
    Returns a tensor ``(num_frames, 3, 224, 224)``.
    """
    if not os.path.isfile(video_path):
        raise ValueError("Video file not found")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames")

    target_indices = set(np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int))
    frames: List[torch.Tensor] = []
    current_idx = 0

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if current_idx in target_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            tensor = _transform(frame_resized)
            frames.append(tensor)
        current_idx += 1
    cap.release()

    if not frames:
        raise ValueError("Failed to extract any readable video frames")

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    video_tensor = torch.stack(frames[:num_frames])  # (num_frames, 3, 224, 224)
    return video_tensor

def prepare_batch(video_path: str, device: torch.device, num_frames: int = 16) -> torch.Tensor:
    """Loads video frames and adds batch dimension (1, num_frames, 3, 224, 224)."""
    video_tensor = load_video_frames(video_path, num_frames=num_frames)
    return video_tensor.unsqueeze(0).to(device)

def get_frame_info(video_path: str, num_frames: int = 16) -> List[Dict[str, float]]:
    """Return sampled frame indices and estimated timestamps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int))
    info = []
    for idx in indices:
        timestamp = idx / fps
        info.append({"frame_index": int(idx), "timestamp_sec": round(timestamp, 3)})
    cap.release()
    return info
