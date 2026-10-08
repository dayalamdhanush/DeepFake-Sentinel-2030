import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml.audio.model import AudioCNN
from tqdm import tqdm

class AudioDataset(Dataset):
    """Dataset for audio deepfake detection.
    Expects directory structure:
    root/train/real/*.wav
    root/train/fake/*.wav
    Same for val/ and test/.
    """
    def __init__(self, root_dir, split='train', sr=16000, n_mels=128, duration=3, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration  # seconds to load per clip
        self.transform = transform
        self.samples = []
        class_names = ['real', 'fake']
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith('.wav'):
                    self.samples.append((os.path.join(class_dir, f), label))
        self.class_names = class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load audio, pad/truncate to fixed length
        y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        if len(y) < self.sr * self.duration:
            pad_len = self.sr * self.duration - len(y)
            y = np.pad(y, (0, pad_len))
        # Compute log‑mel spectrogram
        mel = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [0,1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        # Add channel dimension
        spec = np.expand_dims(log_mel, axis=0).astype(np.float32)  # (1, n_mels, time)
        if self.transform:
            spec = self.transform(spec)
        return torch.from_numpy(spec), label

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for specs, labels in tqdm(loader, desc='Training', leave=False):
        specs = specs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * specs.size(0)
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
        for specs, labels in tqdm(loader, desc='Eval', leave=False):
            specs = specs.to(device)
            labels = labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description='Train audio deepfake detector')
    parser.add_argument('--data-dir', type=str, default='datasets/audio')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-model', type=str, default='models/audio_cnn.pth')
    parser.add_argument('--history-json', type=str, default='metrics/audio_training_history.json')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    train_set = AudioDataset(args.data_dir, split='train')
    val_set = AudioDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} - Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}')
        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
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
