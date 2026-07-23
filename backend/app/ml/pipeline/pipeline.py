"""End-to-end medical X-ray analysis pipeline (validate → body part → disease)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.ml.classifier import get_classifier
from app.ml.pipeline.body_part_classifier import BodyPartClassifier
from app.ml.pipeline.catalog import MSG_NOT_XRAY, MSG_UNKNOWN_PART
from app.ml.pipeline.disease_models import DiseasePredictor
from app.ml.pipeline.xray_validator import validate_medical_xray
from app.ml.preprocess import preprocess_image

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    ok: bool
    is_xray: bool = False
    body_part: str = ""
    body_part_id: str = ""
    disease: str = ""
    confidence: float = 0.0
    recommendation: str = ""
    xray_confidence: float = 0.0
    body_part_confidence: float = 0.0
    source_label: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    error: str = ""
    model_mode: str = ""


class MedicalXrayPipeline:
    """Production orchestration for the three-stage X-ray workflow."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.unified = get_classifier()
        self.body_parts = BodyPartClassifier()
        self.diseases = DiseasePredictor()

    def analyze(self, image: Image.Image) -> PipelineResult:
        tensor = preprocess_image(image)
        predicted, confidence, probs = self.unified.predict(tensor)

        # Stage 1 — medical X-ray validation
        min_xray = float(getattr(self.settings, "xray_confidence_threshold", 0.35))
        stage1 = validate_medical_xray(image, probs, min_xray_confidence=min_xray)
        if not stage1.is_xray:
            return PipelineResult(
                ok=False,
                is_xray=False,
                xray_confidence=stage1.confidence,
                error=stage1.message or MSG_NOT_XRAY,
                model_mode=self.unified.model_mode,
                probabilities=probs,
                source_label=predicted,
            )

        # Stage 2 — body part
        if self.body_parts.model is not None:
            part = self.body_parts.predict_from_tensor(tensor)
        else:
            part = self.body_parts.predict_from_unified_label(predicted, confidence)
        if not part.ok:
            return PipelineResult(
                ok=False,
                is_xray=True,
                xray_confidence=stage1.confidence,
                error=part.message or MSG_UNKNOWN_PART,
                model_mode=self.unified.model_mode,
                probabilities=probs,
                source_label=predicted,
            )

        # Stage 3 — disease for that body part
        disease = self.diseases.predict(
            part.body_part_id,
            tensor,
            unified_label=predicted,
            unified_confidence=confidence,
            unified_probs=probs,
        )

        logger.info(
            "Pipeline OK part=%s disease=%s conf=%.3f",
            part.display_name,
            disease.disease,
            disease.confidence,
        )
        return PipelineResult(
            ok=True,
            is_xray=True,
            body_part=part.display_name,
            body_part_id=part.body_part_id,
            disease=disease.disease,
            confidence=disease.confidence,
            recommendation=disease.recommendation,
            xray_confidence=stage1.confidence,
            body_part_confidence=part.confidence,
            source_label=disease.source_label or predicted,
            probabilities=disease.probabilities or probs,
            model_mode=self.unified.model_mode,
        )


_pipeline: MedicalXrayPipeline | None = None


def get_pipeline() -> MedicalXrayPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalXrayPipeline()
    return _pipeline
