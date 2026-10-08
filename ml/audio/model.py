import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """Simple CNN for mel‑spectrogram based deepfake audio detection.

    Input shape: (1, n_mels, time) – we treat the mel‑spectrogram as a single‑channel image.
    The network consists of three convolutional blocks followed by global average pooling
    and a linear classifier for binary output.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (B, 1, n_mels, T)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
