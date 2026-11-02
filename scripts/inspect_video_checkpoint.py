import torch
from ml.video.model import get_video_model

model = get_video_model(num_classes=2)
checkpoint = torch.load(
    "models/video_cnn_lstm.pth",
    map_location="cpu",
    weights_only=True
)

print("Checkpoint keys match:", set(model.state_dict()) == set(checkpoint.keys()))

print("Checkpoint tensor shapes match:", all(
    model.state_dict()[k].shape == checkpoint[k].shape
    for k in model.state_dict()
))

print("Checkpoint loaded successfully:", end=" ")
model.load_state_dict(checkpoint)
print("YES")
