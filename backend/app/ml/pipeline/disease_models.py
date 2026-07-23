"""Stage 3 — disease prediction for an identified body part."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from app.config import MODEL_DIR
from app.ml.pipeline.catalog import BODY_PARTS, LABEL_TO_BODY_DISEASE, recommendation_for

logger = logging.getLogger(__name__)


class DiseaseResult:
    __slots__ = ("disease", "confidence", "recommendation", "source_label", "probabilities")

    def __init__(
        self,
        disease: str,
        confidence: float,
        recommendation: str,
        source_label: str = "",
        probabilities: dict[str, float] | None = None,
    ) -> None:
        self.disease = disease
        self.confidence = confidence
        self.recommendation = recommendation
        self.source_label = source_label
        self.probabilities = probabilities or {}


class DiseasePredictor:
    """
    Loads optional per-body-part weights from backend/models/disease/{part}.pth.
    Falls back to the unified MedIntel classifier label mapping.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, tuple[nn.Module, list[str]]] = {}

    def _try_load_part_model(self, body_part_id: str) -> tuple[nn.Module, list[str]] | None:
        if body_part_id in self._models:
            return self._models[body_part_id]
        spec = BODY_PARTS.get(body_part_id)
        if not spec or not spec.weights_file:
            return None
        path = MODEL_DIR / spec.weights_file
        if not path.exists():
            return None
        labels = [d.name for d in spec.diseases]
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        self._models[body_part_id] = (model, labels)
        logger.info("Loaded disease model for %s from %s", body_part_id, path)
        return self._models[body_part_id]

    @torch.inference_mode()
    def predict(
        self,
        body_part_id: str,
        tensor: torch.Tensor,
        *,
        unified_label: str,
        unified_confidence: float,
        unified_probs: dict[str, float],
    ) -> DiseaseResult:
        loaded = self._try_load_part_model(body_part_id)
        if loaded is not None:
            model, labels = loaded
            logits = model(tensor.to(self.device))
            probs_t = torch.softmax(logits, dim=1)[0]
            idx = int(probs_t.argmax().item())
            disease = labels[idx]
            conf = float(probs_t[idx].item())
            prob_map = {labels[i]: float(probs_t[i].item()) for i in range(len(labels))}
            return DiseaseResult(
                disease=disease,
                confidence=conf,
                recommendation=recommendation_for(body_part_id, disease),
                source_label=unified_label,
                probabilities=prob_map,
            )

        mapped = LABEL_TO_BODY_DISEASE.get(unified_label.upper())
        if mapped and mapped[0] == body_part_id:
            disease = mapped[1]
        else:
            # Use top disease name for this part as a soft fallback
            spec = BODY_PARTS.get(body_part_id)
            disease = spec.diseases[0].name if spec and spec.diseases else unified_label
        return DiseaseResult(
            disease=disease,
            confidence=unified_confidence,
            recommendation=recommendation_for(body_part_id, disease),
            source_label=unified_label,
            probabilities=unified_probs,
        )
