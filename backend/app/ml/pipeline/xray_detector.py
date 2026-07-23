"""Trained binary detector: clinical X-ray/CT/MRI vs non-medical images."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.config import MODEL_DIR

logger = logging.getLogger(__name__)

XRAY_GATE_WEIGHTS = MODEL_DIR / "xray_gate_resnet18.pth"
# ImageFolder alphabetical: not_xray=0, xray=1
CLASS_NOT_XRAY = 0
CLASS_XRAY = 1


class XrayDetector:
    """ResNet18 binary gate. Missing weights → inactive (heuristics only)."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if XRAY_GATE_WEIGHTS.exists():
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            state = torch.load(XRAY_GATE_WEIGHTS, map_location=self.device, weights_only=True)
            model.load_state_dict(state, strict=False)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Loaded X-ray gate detector from %s", XRAY_GATE_WEIGHTS)
        else:
            logger.warning("X-ray gate weights missing at %s — heuristics only", XRAY_GATE_WEIGHTS)

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict_proba(self, image: Image.Image) -> float:
        """Return P(is_xray) in [0, 1]. 0.5 if model unavailable."""
        if self.model is None:
            return 0.5
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        return float(probs[CLASS_XRAY].item())


_detector: XrayDetector | None = None


def get_xray_detector() -> XrayDetector:
    global _detector
    if _detector is None:
        _detector = XrayDetector()
    return _detector
