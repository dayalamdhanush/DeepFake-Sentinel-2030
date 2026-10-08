import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from ml.video.model import VideoDeepFakeModel
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from tqdm import tqdm

# Reuse VideoDataset from training script (copy minimal version)
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='test', num_frames=16, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
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
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame)
                if self.transform:
                    frames.append(self.transform(pil))
                else:
                    frames.append(transforms.ToTensor()(pil))
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor, label

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device, fine_tune=False):
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model not trained')
    model = VideoDeepFakeModel(fine_tune=fine_tune)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc='Evaluating', leave=False):
            videos = videos.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description='Evaluate video deepfake detector')
    parser.add_argument('--data-dir', type=str, default='datasets/video', help='Root of video dataset')
    parser.add_argument('--model', type=str, default='models/video_cnn_lstm.pth', help='Trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--output', type=str, default='metrics/video_metrics.json')
    args = parser.parse_args()

    device = get_device()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = VideoDataset(args.data_dir, split='test', num_frames=args.num_frames, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.model, device)
    true, pred = evaluate(model, test_loader, device)

    metrics = {
        'accuracy': accuracy_score(true, pred),
        'precision': precision_score(true, pred, average='binary', pos_label=1),
        'recall': recall_score(true, pred, average='binary', pos_label=1),
        'f1': f1_score(true, pred, average='binary', pos_label=1),
        'confusion_matrix': confusion_matrix(true, pred).tolist(),
        'class_names': ['real', 'fake'],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Video metrics saved to {args.output}')

if __name__ == '__main__':
    main()
