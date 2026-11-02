import torch
from ml.video.model import get_video_model

model = get_video_model(num_classes=2)

print("VIDEO MODEL INSPECTION")
print("-" * 40)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")

print("\nCNN parameter status:")
for name, param in model.cnn.named_parameters():
    if name.startswith(("layer3", "layer4", "fc")):
        print(name, param.requires_grad)
