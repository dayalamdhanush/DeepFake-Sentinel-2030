from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from ml.audio.model import AudioCNN
from scripts.auto_train_pipeline import AudioDatasetWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ds = AudioDatasetWrapper(Path("datasets/audio/test"))
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

model = AudioCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(
    "models/audio_cnn_experimental.pth",
    map_location=device,
    weights_only=True
))
model.eval()

correct = 0
total = 0
actual = []
predicted = []

with torch.no_grad():
    for specs, labels in test_loader:
        specs = specs.to(device)
        labels = labels.to(device)
        outputs = model(specs)
        preds = outputs.argmax(dim=1)

        actual.extend(labels.cpu().tolist())
        predicted.extend(preds.cpu().tolist())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("EXPERIMENTAL AUDIO MODEL TEST RESULTS")
print("Class mapping: REAL=0, FAKE=1")
print("Test samples:", total)
print("Correct predictions:", correct)
print("Test accuracy:", round(100 * correct / total, 2), "%")
print("Actual labels:", Counter(actual))
print("Predicted labels:", Counter(predicted))

for label, name in [(0, "REAL"), (1, "FAKE")]:
    tp = sum(a == label and p == label for a, p in zip(actual, predicted))
    fp = sum(a != label and p == label for a, p in zip(actual, predicted))
    fn = sum(a == label and p != label for a, p in zip(actual, predicted))
    print(f"{name}: TP={tp}, FP={fp}, FN={fn}")
