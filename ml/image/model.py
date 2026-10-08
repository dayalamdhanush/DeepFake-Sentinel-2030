import torch
import torchvision

def get_image_model(num_classes: int = 2):
    """Return an EfficientNet‑B0 model ready for binary deep‑fake detection.

    The model is loaded with ImageNet pretrained weights and its final
    classifier is replaced with a linear layer matching ``num_classes``.
    """
    # Load pretrained EfficientNet‑B0
    model = torchvision.models.efficientnet_b0(pretrained=True)
    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model
