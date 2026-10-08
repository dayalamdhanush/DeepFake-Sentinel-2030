# ml/video/model.py
"""Video model: CNN (ResNet‑18) backbone + LSTM for temporal aggregation.
Returns a binary classifier (REAL=0, FAKE=1).
"""

import torch
import torch.nn as nn
import torchvision.models as models

class VideoCNNLSTM(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        # Use pretrained ResNet-18 as frame feature extractor (exclude final FC)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # output 512-dim per frame
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        batch, seq_len = x.shape[:2]
        # Merge batch and seq_len for CNN processing
        x = x.view(batch * seq_len, *x.shape[2:])  # (B*seq, C, H, W)
        frame_feats = self.cnn(x)  # (B*seq, 512)
        frame_feats = frame_feats.view(batch, seq_len, -1)  # (B, seq, 512)
        lstm_out, _ = self.lstm(frame_feats)  # (B, seq, hidden)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

def get_video_model(num_classes: int = 2) -> nn.Module:
    """Return a VideoCNNLSTM model ready for binary classification."""
    model = VideoCNNLSTM(num_classes=num_classes)
    return model
