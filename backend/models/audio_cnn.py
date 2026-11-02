# backend/models/audio_cnn.py
"""Simple CNN for audio mel-spectrogram classification (binary).

Architecture:
- 3 Conv2d + BatchNorm + ReLU blocks with MaxPool
- Global Average Pooling
- Dropout(0.3) for regularization
- Fully Connected layer to 2 logits

CLASS MAPPING: FAKE=0, REAL=1
"""

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout     = nn.Dropout(p=dropout)
        self.classifier  = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, N_MELS, FIXED_FRAMES)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)   # (batch, 64)
        x = self.dropout(x)
        return self.classifier(x)


def get_audio_model(num_classes: int = 2, dropout: float = 0.3) -> AudioCNN:
    """Return an AudioCNN instance ready for binary classification.

    FAKE=0, REAL=1 — consistent with ml.audio.preprocessing.CLASS_TO_IDX
    """
    return AudioCNN(num_classes=num_classes, dropout=dropout)
