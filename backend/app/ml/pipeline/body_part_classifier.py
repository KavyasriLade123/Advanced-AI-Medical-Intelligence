"""Stage 2 — identify X-ray body part."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

from app.config import MODEL_DIR
from app.ml.pipeline.catalog import BODY_PARTS, LABEL_TO_BODY_DISEASE, MSG_UNKNOWN_PART

logger = logging.getLogger(__name__)

BODY_PART_MODEL_CLASSES = [
    "chest",
    "hand",
    "wrist",
    "elbow",
    "shoulder",
    "knee",
    "hip",
    "pelvis",
    "spine",
    "cervical_spine",
    "lumbar_spine",
    "skull",
    "foot",
    "ankle",
    "leg",
    "femur",
    "arm",
    "abdomen",
    "dental",
]


class BodyPartResult:
    __slots__ = ("body_part_id", "display_name", "confidence", "message", "ok")

    def __init__(
        self,
        ok: bool,
        body_part_id: str = "",
        display_name: str = "",
        confidence: float = 0.0,
        message: str = "",
    ) -> None:
        self.ok = ok
        self.body_part_id = body_part_id
        self.display_name = display_name
        self.confidence = confidence
        self.message = message


class BodyPartClassifier:
    """
    Prefers dedicated checkpoint: backend/models/body_part_resnet18.pth
    Otherwise maps unified classifier label → body part.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = list(BODY_PART_MODEL_CLASSES)
        self.weights_path = MODEL_DIR / "body_part_resnet18.pth"
        self.model: nn.Module | None = None
        if self.weights_path.exists():
            self.model = self._build_model()
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Loaded dedicated body-part model from %s", self.weights_path)

    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.labels))
        return model

    @torch.inference_mode()
    def predict_from_tensor(self, tensor: torch.Tensor) -> BodyPartResult:
        if self.model is None:
            return BodyPartResult(False, message=MSG_UNKNOWN_PART)
        logits = self.model(tensor.to(self.device))
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        part_id = self.labels[idx]
        spec = BODY_PARTS.get(part_id)
        if not spec or conf < 0.35:
            return BodyPartResult(False, confidence=conf, message=MSG_UNKNOWN_PART)
        return BodyPartResult(True, part_id, spec.display_name, conf)

    def predict_from_unified_label(self, label: str, confidence: float) -> BodyPartResult:
        mapped = LABEL_TO_BODY_DISEASE.get(label.upper())
        if not mapped:
            return BodyPartResult(False, confidence=confidence, message=MSG_UNKNOWN_PART)
        part_id, _disease = mapped
        spec = BODY_PARTS.get(part_id)
        if not spec:
            return BodyPartResult(False, confidence=confidence, message=MSG_UNKNOWN_PART)
        return BodyPartResult(True, part_id, spec.display_name, confidence)

    def predict_from_unified_probs(
        self,
        predicted: str,
        confidence: float,
        probs: dict[str, float],
    ) -> BodyPartResult:
        """Prefer top label; if unmapped (e.g. UNSUPPORTED), use best anatomy class in probs."""
        direct = self.predict_from_unified_label(predicted, confidence)
        if direct.ok:
            return direct

        ranked = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
        for name, prob in ranked:
            mapped = LABEL_TO_BODY_DISEASE.get(name.upper())
            if not mapped:
                continue
            if float(prob) < 0.12:
                break
            part_id, _disease = mapped
            spec = BODY_PARTS.get(part_id)
            if spec:
                return BodyPartResult(True, part_id, spec.display_name, float(prob))
        return BodyPartResult(False, confidence=confidence, message=MSG_UNKNOWN_PART)
