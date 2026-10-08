import argparse
import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# -------------------------------
# Dataset for video files
# -------------------------------
class VideoDataset(Dataset):
    """Loads videos from a directory with the following structure:
    root/train/real/*.mp4
    root/train/fake/*.mp4
    Same for val/ and test/.
    Samples ``num_frames`` uniformly spaced frames from each video.
    Returns a tensor of shape (num_frames, C, H, W) and the class index.
    """
    def __init__(self, root_dir, split='train', num_frames=16, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []  # list of (video_path, label)
        class_names = ['real', 'fake']
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(class_dir, f), label))
        self.class_names = class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f'Failed to read video {video_path}')
        # Determine frame indices to sample uniformly
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame)
                if self.transform:
                    frame_tensor = self.transform(frame_img)  # C,H,W
                else:
                    frame_tensor = transforms.ToTensor()(frame_img)
                frames.append(frame_tensor)
        cap.release()
        # If not enough frames (e.g., video shorter), duplicate last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        video_tensor = torch.stack(frames)  # (T,C,H,W)
        return video_tensor, label

# -------------------------------
# Model (EfficientNet-B0 + LSTM)
# -------------------------------
class VideoDeepFakeModel(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=1, fine_tune=False):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        # Remove final classifier
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # output (B,1280,1,1)
        if not fine_tune:
            for p in self.cnn.parameters():
                p.requires_grad = False
        self.cnn_output_dim = 1280
        self.lstm = nn.LSTM(input_size=self.cnn_output_dim, hidden_size=lstm_hidden,
                             num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x shape (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)  # (B*T,1280,1,1)
        feats = feats.view(B, T, self.cnn_output_dim)
        lstm_out, _ = self.lstm(feats)
        # Use last timestep representation
        last = lstm_out[:, -1, :]
        logits = self.classifier(last)
        return logits

# -------------------------------
# Training utilities
# -------------------------------
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for videos, labels in tqdm(loader, desc='Training', leave=False):
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc='Eval', leave=False):
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description='Train video deepfake detector')
    parser.add_argument('--data-dir', type=str, default='datasets/video', help='Root of video dataset')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to sample per video')
    parser.add_argument('--fine-tune', action='store_true', help='Fine‑tune CNN backbone')
    parser.add_argument('--output-model', type=str, default='models/video_cnn_lstm.pth')
    parser.add_argument('--history-json', type=str, default='metrics/video_training_history.json')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    transform = build_transforms()
    train_set = VideoDataset(args.data_dir, split='train', num_frames=args.num_frames, transform=transform)
    val_set = VideoDataset(args.data_dir, split='val', num_frames=args.num_frames, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VideoDeepFakeModel(fine_tune=args.fine_tune).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}')
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print(f'Model saved to {args.output_model}')
    os.makedirs(os.path.dirname(args.history_json), exist_ok=True)
    with open(args.history_json, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to {args.history_json}')

if __name__ == '__main__':
    main()
