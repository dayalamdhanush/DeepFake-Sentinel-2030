# ml/image/gradcam.py
"""Grad-CAM heatmap generation for EfficientNet-B0.

Hooks into the last convolutional block (features[-1]) to produce
a spatial saliency map showing which regions the model focused on.
Returns a PIL Image with heatmap overlaid on the de-normalized input.
"""

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str = "features",
    class_idx: int = 1,
) -> Image.Image:
    """Generate a Grad-CAM heatmap and overlay it on the input image.

    Args:
        model: EfficientNet-B0 in eval mode, already on correct device.
        input_tensor: Tensor of shape (1, 3, H, W).
        target_layer: Ignored (always hooks last features block of EfficientNet).
        class_idx: Class to compute gradients for (1 = FAKE).

    Returns:
        PIL.Image with heatmap overlay.
    """
    gradients: list = []
    activations: list = []

    def _fwd_hook(module, inp, out):
        activations.append(out.detach())

    def _bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Hook into the last Sequential block of EfficientNet features
    # model.features is a Sequential; last child = features[-1]
    last_block = model.features[-1]
    fwd_h = last_block.register_forward_hook(_fwd_hook)
    bwd_h = last_block.register_full_backward_hook(_bwd_hook)

    try:
        # Enable gradients for the forward pass even in eval mode
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(input_tensor)
            score = outputs[:, class_idx]
            score.backward(retain_graph=False)

        if not gradients or not activations:
            raise RuntimeError("Hooks did not capture data")

        grads = gradients[0]       # (1, C, H', W')
        acts = activations[0]      # (1, C, H', W')

        # Weighted combination
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        _, _, H, W = input_tensor.shape
        cam_resized = cv2.resize(cam_np, (W, H))
        heatmap = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # BGR

        # Denormalize input tensor back to [0,255] RGB for overlay
        inv_norm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        img_t = inv_norm(input_tensor.squeeze(0).cpu())
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        img_bgr = (img_np[..., ::-1] * 255).astype(np.uint8)   # RGB→BGR

        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    except Exception:
        # Fallback: return a blank 224x224 grey image rather than crashing the API
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    finally:
        fwd_h.remove()
        bwd_h.remove()
