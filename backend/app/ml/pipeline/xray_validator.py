"""Stage 1 — validate that the upload is a medical X-ray."""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import _tone_stats, looks_like_photo_or_text
from app.ml.pipeline.catalog import MSG_NOT_XRAY, NON_XRAY_LABELS

logger = logging.getLogger(__name__)


class XrayValidationResult:
    __slots__ = ("is_xray", "confidence", "message")

    def __init__(self, is_xray: bool, confidence: float, message: str = "") -> None:
        self.is_xray = is_xray
        self.confidence = confidence
        self.message = message


def validate_medical_xray(
    image: Image.Image,
    class_probs: dict[str, float] | None = None,
    *,
    min_xray_confidence: float = 0.55,
    min_anatomy_midtone: float = 0.32,
) -> XrayValidationResult:
    """
    Stage 1 gate.

    Rejects nature photos, selfies, documents, screenshots, skin/eye photos, etc.
    `min_xray_confidence` is applied to (1 - P(non-xray labels)).
    Raise toward 0.95 once a dedicated binary X-ray detector is trained.
    """
    if looks_like_photo_or_text(image):
        logger.info("X-ray validation failed: photo/text heuristic")
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    tones = _tone_stats(image)
    if tones["mid"] < min_anatomy_midtone:
        logger.info("X-ray validation failed: insufficient anatomical mid-tones (mid=%.3f)", tones["mid"])
        return XrayValidationResult(False, float(tones["mid"]), MSG_NOT_XRAY)

    if class_probs:
        non_xray_mass = sum(float(class_probs.get(k, 0.0)) for k in NON_XRAY_LABELS)
        xray_confidence = max(0.0, 1.0 - non_xray_mass)
        top_label = max(class_probs.items(), key=lambda kv: kv[1])[0].upper()
        if top_label in NON_XRAY_LABELS:
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        if xray_confidence < min_xray_confidence:
            logger.info(
                "X-ray validation failed: confidence %.3f < %.3f",
                xray_confidence,
                min_xray_confidence,
            )
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        return XrayValidationResult(True, xray_confidence, "")

    # Heuristic-only path (no model probs yet)
    return XrayValidationResult(True, float(tones["mid"]), "")
