import os
import torch
from ml.image.model import get_image_model
from ml.image.gradcam import generate_gradcam_heatmap
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64

class ImageService:
    """Service for image deepfake inference.
    Loads EfficientNet‑B0 model (if checkpoint exists) and provides prediction
    with confidence, risk level, class probabilities, and Grad‑CAM overlay.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/efficientnet_b0_deepfake.pth'))
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError('MODEL_NOT_TRAINED')
        if self.model is None:
            self.model = get_image_model().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        return self.model

    def _preprocess(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor, img

    def predict(self, image_path: str):
        try:
            model = self._load_model()
        except FileNotFoundError:
            return {"error": "MODEL_NOT_TRAINED"}
        input_tensor, pil_img = self._preprocess(image_path)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        label = 'real' if pred_idx == 0 else 'deepfake'
        risk = self._risk_from_confidence(confidence, label)
        # Grad‑CAM heatmap
        heatmap_img = generate_gradcam_heatmap(model, input_tensor, target_layer="features[7]")
        # Encode original and heatmap as base64 data URLs for frontend
        def img_to_data_url(img: Image.Image):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        original_b64 = img_to_data_url(pil_img)
        heatmap_b64 = img_to_data_url(heatmap_img)
        return {
            "label": label,
            "confidence": confidence,
            "risk": risk,
            "probabilities": {"real": float(probs[0]), "deepfake": float(probs[1])},
            "original_image": original_b64,
            "gradcam": heatmap_b64,
        }

    @staticmethod
    def _risk_from_confidence(confidence: float, label: str) -> str:
        # Use project‑defined thresholds (same for both classes)
        if confidence < 0.4:
            return "low"
        elif confidence < 0.7:
            return "medium"
        else:
            return "high"
