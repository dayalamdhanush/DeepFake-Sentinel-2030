import torch
from ml.image.model import get_image_model
import os

def load_image_model(device: torch.device = None):
    """Utility to load the EfficientNet‑B0 model.
    Returns the model instance (already on the requested device) or ``None``
    if the checkpoint file does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/efficientnet_b0_deepfake.pth"))
    if not os.path.exists(model_path):
        return None
    model = get_image_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
