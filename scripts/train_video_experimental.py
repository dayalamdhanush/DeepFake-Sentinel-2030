from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.video.model import get_video_model
from scripts.train_video import VideoFolderDataset

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = VideoFolderDataset("datasets/video/train", num_frames=16)
val_ds = VideoFolderDataset("datasets/video/val", num_frames=16)

print("Training videos:", len(train_ds))
print("Validation videos:", len(val_ds))
print("Training classes:", train_ds.class_to_idx)
print("Device:", device)

train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=2, shuffle=False, num_workers=0
)

model = get_video_model(num_classes=2).to(device)

checkpoint = torch.load(
    "models/video_cnn_lstm.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint)

# Freeze the pretrained CNN initially.
for param in model.cnn.parameters():
    param.requires_grad = False

# Fine-tune only the final ResNet block.
for param in model.cnn.layer4.parameters():
    param.requires_grad = True

# Keep BatchNorm running statistics stable on the tiny dataset.
for module in model.cnn.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

# Train the LSTM and classifier.
for param in model.lstm.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print("Trainable parameters:", f"{trainable:,}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.00001,
    weight_decay=0.01
)

output = Path("models/video_cnn_lstm_experimental.pth")
best_val_acc = -1.0
best_state = None
patience = 5
epochs_without_improvement = 0

for epoch in range(1, 21):
    model.train()

    # Keep frozen CNN BatchNorm layers in evaluation mode.
    for module in model.cnn.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_acc = train_correct / max(train_total, 1)
    val_acc = val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch:02d}/20 | "
        f"Loss: {train_loss / max(train_total, 1):.4f} | "
        f"Train: {train_acc:.2%} | Validation: {val_acc:.2%}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

if best_state is not None:
    torch.save(best_state, output)

print("\nExperimental training completed.")
print("Best validation accuracy:", f"{best_val_acc:.2%}")
print("Saved checkpoint:", output)
