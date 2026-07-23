from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from app.config import WEIGHTS_PATH, get_settings


class ChestXRayClassifier:
    """ResNet18 classifier for NORMAL / PNEUMONIA / BONE_FRACTURE."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self.settings.labels
        self.model = self._build_model()
        self.model_mode = self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.labels))
        return model

    def _load_weights(self) -> str:
        if not WEIGHTS_PATH.exists():
            return "imagenet-demo"
        state = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
        model_state = self.model.state_dict()
        # Allow loading older 2-class checkpoints into backbone only.
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(filtered)
        self.model.load_state_dict(model_state)
        if any(k.startswith("fc.") and k in filtered for k in state):
            return "finetuned"
        return "partial-finetuned"

    def predict(self, tensor: torch.Tensor) -> tuple[str, float, dict[str, float]]:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        values = probs.cpu().tolist()
        mapping = {label: float(values[i]) for i, label in enumerate(self.labels)}
        best_idx = int(probs.argmax().item())
        predicted = self.labels[best_idx]
        confidence = float(values[best_idx])
        return predicted, confidence, mapping

    def target_layer(self) -> nn.Module:
        return self.model.layer4[-1]


_classifier: ChestXRayClassifier | None = None


def get_classifier() -> ChestXRayClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ChestXRayClassifier()
    return _classifier


def reload_classifier() -> ChestXRayClassifier:
    global _classifier
    _classifier = ChestXRayClassifier()
    return _classifier
