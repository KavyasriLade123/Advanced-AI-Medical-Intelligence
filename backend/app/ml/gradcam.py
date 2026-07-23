from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.ml.classifier import ChestXRayClassifier


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN explanations."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def overlay_gradcam(
    original: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    img = np.array(original.convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    return Image.fromarray(overlay)


def explain_prediction(
    classifier: ChestXRayClassifier,
    tensor: torch.Tensor,
    original_image: Image.Image,
    output_path: Path,
    class_idx: int | None = None,
) -> Path:
    cam_engine = GradCAM(classifier.model, classifier.target_layer())
    try:
        cam = cam_engine.generate(tensor.to(classifier.device), class_idx=class_idx)
    finally:
        cam_engine.close()
    overlay = overlay_gradcam(original_image, cam)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, format="PNG")
    return output_path
