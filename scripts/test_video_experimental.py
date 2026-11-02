import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ml.video.model import get_video_model
from scripts.evaluate_video import VideoFolderDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ds = VideoFolderDataset("datasets/video/test", num_frames=16)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

model = get_video_model(num_classes=2).to(device)
model.load_state_dict(torch.load(
    "models/video_cnn_lstm_experimental.pth",
    map_location=device,
    weights_only=True
))
model.eval()

actual = []
predicted = []

with torch.no_grad():
    for frames, labels in test_loader:
        outputs = model(frames.to(device))
        preds = outputs.argmax(dim=1)
        actual.extend(labels.tolist())
        predicted.extend(preds.cpu().tolist())

print("\nEXPERIMENTAL VIDEO MODEL TEST RESULTS")
print("Test samples:", len(actual))
print("Accuracy:", f"{accuracy_score(actual, predicted):.2%}")
print("Confusion matrix:")
print(confusion_matrix(actual, predicted))
print("\nClassification report:")
print(classification_report(
    actual, predicted,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
))
