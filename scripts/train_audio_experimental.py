from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.audio.model import AudioCNN
from scripts.auto_train_pipeline import AudioDatasetWrapper

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = AudioDatasetWrapper(Path("datasets/audio/train"))
val_ds = AudioDatasetWrapper(Path("datasets/audio/val"))

print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Device:", device)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

model = AudioCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_val_acc = -1
output = Path("models/audio_cnn_experimental.pth")

for epoch in range(1, 31):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total if total else 0

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total else 0

    print(
        f"Epoch {epoch:02d}/30 | "
        f"Loss: {running_loss / max(total, 1):.4f} | "
        f"Train: {train_acc:.2%} | Validation: {val_acc:.2%}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), output)

print("\nTraining completed.")
print("Best validation accuracy:", round(best_val_acc * 100, 2), "%")
print("Experimental checkpoint:", output)
