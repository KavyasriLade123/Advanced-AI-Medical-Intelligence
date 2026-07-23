"""Stage 3 — disease prediction for an identified body part.

Uses optional per-part weights when present; otherwise falls back to the
existing unified MedIntel classifier only for supported families
(brain/skull, chest, bone extremities, abdomen).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

from app.config import MODEL_DIR
from app.ml.pipeline.catalog import BODY_PARTS, LABEL_TO_BODY_DISEASE, recommendation_for

logger = logging.getLogger(__name__)

MSG_DISEASE_UNAVAILABLE = (
    "Body part detected successfully, but disease detection for this body part is currently unavailable."
)

# Body parts the existing unified model can support without a dedicated .pth
BRAIN_FAMILY = {"brain", "skull"}
CHEST_FAMILY = {"chest", "lungs", "heart"}
ORTHO_FAMILY = {
    "bone",
    "shoulder",
    "clavicle",
    "arm",
    "elbow",
    "forearm",
    "wrist",
    "hand",
    "fingers",
    "pelvis",
    "hip",
    "femur",
    "knee",
    "leg",
    "tibia",
    "fibula",
    "ankle",
    "foot",
    "toes",
    "spine",
    "cervical_spine",
    "thoracic_spine",
    "lumbar_spine",
}
ABDOMEN_FAMILY = {"abdomen"}


class DiseaseResult:
    __slots__ = (
        "disease",
        "confidence",
        "recommendation",
        "source_label",
        "probabilities",
        "available",
        "message",
    )

    def __init__(
        self,
        disease: str = "",
        confidence: float = 0.0,
        recommendation: str = "",
        source_label: str = "",
        probabilities: dict[str, float] | None = None,
        available: bool = True,
        message: str = "",
    ) -> None:
        self.disease = disease
        self.confidence = confidence
        self.recommendation = recommendation
        self.source_label = source_label
        self.probabilities = probabilities or {}
        self.available = available
        self.message = message


class DiseasePredictor:
    """Dedicated disease/{part}.pth if present; else unified classifier for known families."""

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

    def _unified_supported(self, body_part_id: str) -> bool:
        return body_part_id in BRAIN_FAMILY | CHEST_FAMILY | ORTHO_FAMILY | ABDOMEN_FAMILY

    def _from_unified(
        self,
        body_part_id: str,
        unified_label: str,
        unified_confidence: float,
        unified_probs: dict[str, float],
    ) -> DiseaseResult | None:
        """Map unified labels into a disease for this body-part family. None = unsupported."""
        label = unified_label.upper()
        mapped = LABEL_TO_BODY_DISEASE.get(label)

        if body_part_id in BRAIN_FAMILY:
            brain_tumor = float(unified_probs.get("BRAIN_TUMOR", 0.0))
            brain_normal = float(unified_probs.get("BRAIN_NORMAL", 0.0))
            if brain_tumor >= brain_normal and brain_tumor >= 0.12:
                return DiseaseResult(
                    "Brain tumor / mass",
                    brain_tumor,
                    recommendation_for(body_part_id, "Brain tumor / mass"),
                    "BRAIN_TUMOR",
                    unified_probs,
                )
            if brain_normal >= 0.12 or (mapped and mapped[0] in BRAIN_FAMILY):
                conf = brain_normal if brain_normal >= 0.12 else unified_confidence
                return DiseaseResult(
                    "Normal",
                    conf,
                    recommendation_for(body_part_id, "Normal"),
                    "BRAIN_NORMAL",
                    unified_probs,
                )
            return None

        if body_part_id in CHEST_FAMILY:
            if mapped and mapped[0] in {"chest", "lungs"}:
                disease = mapped[1]
                if body_part_id == "lungs" and disease == "Breast abnormality":
                    disease = "Pneumonia"
                return DiseaseResult(
                    disease,
                    unified_confidence,
                    recommendation_for(body_part_id, disease),
                    label,
                    unified_probs,
                )
            # Soft pick among chest-related probs
            for name, disease_name in (
                ("PNEUMONIA", "Pneumonia"),
                ("NORMAL", "Normal"),
                ("BREAST_MALIGNANT", "Breast abnormality"),
                ("BREAST_NORMAL", "Normal"),
            ):
                p = float(unified_probs.get(name, 0.0))
                if p >= 0.20:
                    return DiseaseResult(
                        disease_name,
                        p,
                        recommendation_for(body_part_id, disease_name),
                        name,
                        unified_probs,
                    )
            return None

        if body_part_id in ORTHO_FAMILY:
            bone_p = float(unified_probs.get("BONE_FRACTURE", 0.0))
            limb_p = float(unified_probs.get("LOWER_LIMB", 0.0))
            if bone_p >= 0.15 or limb_p >= 0.15 or (mapped and mapped[0] in {"bone", "leg"}):
                conf = max(bone_p, limb_p, unified_confidence if mapped else 0.0)
                return DiseaseResult(
                    "Fracture",
                    conf,
                    recommendation_for(body_part_id, "Fracture"),
                    "BONE_FRACTURE" if bone_p >= limb_p else "LOWER_LIMB",
                    unified_probs,
                )
            # Valid medical bone film but model not sure → Normal rather than inventing disease
            if float(unified_probs.get("NORMAL", 0.0)) >= 0.25:
                return DiseaseResult(
                    "Normal",
                    float(unified_probs["NORMAL"]),
                    recommendation_for(body_part_id, "Normal"),
                    "NORMAL",
                    unified_probs,
                )
            return None

        if body_part_id in ABDOMEN_FAMILY:
            if mapped and mapped[0] == "abdomen":
                return DiseaseResult(
                    mapped[1],
                    unified_confidence,
                    recommendation_for(body_part_id, mapped[1]),
                    label,
                    unified_probs,
                )
            abd = float(unified_probs.get("ABDOMEN", 0.0))
            if abd >= 0.20:
                return DiseaseResult(
                    "Abnormality",
                    abd,
                    recommendation_for(body_part_id, "Abnormality"),
                    "ABDOMEN",
                    unified_probs,
                )
            return None

        return None

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
                available=True,
            )

        if not self._unified_supported(body_part_id):
            return DiseaseResult(available=False, message=MSG_DISEASE_UNAVAILABLE)

        result = self._from_unified(body_part_id, unified_label, unified_confidence, unified_probs)
        if result is None:
            return DiseaseResult(available=False, message=MSG_DISEASE_UNAVAILABLE)
        return result
