import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml.image.model import get_image_model


class RealFakeDataset(Dataset):
    # IMPORTANT: Must match the training pipeline:
    # REAL = 0
    # FAKE = 1
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue

            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet-B0 DeepFake image classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/efficientnet_b0_deepfake.pth"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/image"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.model)

    if not checkpoint_path.is_file():
        print("MODEL_NOT_TRAINED")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_image_model(num_classes=2)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model = model.to(device)
    model.eval()

    test_dir = Path(args.data_dir) / "test"

    if not test_dir.is_dir():
        raise FileNotFoundError(
            "Test directory not found."
        )

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = RealFakeDataset(
        test_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"TEST IMAGES: {len(test_dataset)}")
    print("CLASS MAPPING: REAL=0, FAKE=1")

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(
        all_labels,
        all_preds,
        pos_label=1,
        zero_division=0
    )
    rec = recall_score(
        all_labels,
        all_preds,
        pos_label=1,
        zero_division=0
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        pos_label=1,
        zero_division=0
    )
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1]
    ).tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "class_mapping": {
            "real": 0,
            "fake": 1
        }
    }

    os.makedirs("metrics", exist_ok=True)

    out_path = Path("metrics/image_metrics.json")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
