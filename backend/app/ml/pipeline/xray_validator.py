"""Stage 1 — validate that the upload is a medical X-ray."""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import XRAY_ANATOMY_LABELS, _tone_stats, looks_like_photo_or_text
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
    min_xray_confidence: float = 0.35,
    min_anatomy_midtone: float = 0.20,
) -> XrayValidationResult:
    """
    Stage 1 gate — prefer accepting real X-rays (including phone photos of films).

    Priority:
      1) Strong anatomy-class prediction + some mid-gray tissue → accept
      2) Clear text/UI / non-clinical photo → reject
      3) Otherwise require mid-tones + non-xray mass below threshold
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])

    top_label = ""
    top_prob = 0.0
    xray_confidence = mid
    if class_probs:
        top_label, top_prob = max(class_probs.items(), key=lambda kv: kv[1])
        top_label = top_label.upper()
        non_xray_mass = sum(float(class_probs.get(k, 0.0)) for k in NON_XRAY_LABELS)
        xray_confidence = max(0.0, 1.0 - non_xray_mass)

        # Strong radiograph prediction overrides mild phone-photo color casts
        if (
            top_label in XRAY_ANATOMY_LABELS
            and top_prob >= 0.30
            and mid >= 0.18
            and top_label not in NON_XRAY_LABELS
        ):
            logger.info(
                "X-ray accepted via anatomy prediction %s (%.3f) mid=%.3f",
                top_label,
                top_prob,
                mid,
            )
            return XrayValidationResult(True, max(xray_confidence, top_prob), "")

    if looks_like_photo_or_text(image):
        logger.info("X-ray validation failed: photo/text heuristic")
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if mid < min_anatomy_midtone:
        logger.info("X-ray validation failed: mid-tones %.3f < %.3f", mid, min_anatomy_midtone)
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
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

    return XrayValidationResult(True, mid, "")
