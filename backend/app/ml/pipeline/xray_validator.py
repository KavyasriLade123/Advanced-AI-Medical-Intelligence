"""Stage 1 — validate medical X-ray / CT / MRI clinical images."""

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
    min_anatomy_midtone: float = 0.16,
) -> XrayValidationResult:
    """
    Accept clinical radiographs and cross-sectional scans (CT/MRI panels).
    Reject selfies, documents, Instagram text, nature photos.
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])

    top_label = ""
    top_prob = 0.0
    xray_confidence = mid
    if class_probs:
        top_label, top_prob = max(class_probs.items(), key=lambda kv: kv[1])
        top_label = top_label.upper()
        non_xray_mass = sum(float(class_probs.get(k, 0.0)) for k in NON_XRAY_LABELS)
        xray_confidence = max(0.0, 1.0 - non_xray_mass)

        # Strong anatomy prediction (chest/brain/bone/etc.)
        if top_label in XRAY_ANATOMY_LABELS and top_prob >= 0.28 and mid >= 0.14:
            logger.info(
                "Clinical image accepted via %s (%.3f) mid=%.3f lut=%s",
                top_label,
                top_prob,
                mid,
                medical_lut,
            )
            return XrayValidationResult(True, max(xray_confidence, top_prob), "")

        # Brain panels often score as BRAIN_* with moderate confidence
        if top_label.startswith("BRAIN") and top_prob >= 0.22 and mid >= 0.12:
            return XrayValidationResult(True, max(xray_confidence, top_prob), "")

    if looks_like_photo_or_text(image):
        logger.info("Clinical validation failed: photo/text heuristic")
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Multi-slice MRI/CT montages can have darker borders → slightly lower mid
    mid_needed = 0.12 if medical_lut else min_anatomy_midtone
    if mid < mid_needed:
        logger.info("Clinical validation failed: mid-tones %.3f < %.3f", mid, mid_needed)
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        if top_label in NON_XRAY_LABELS and top_prob >= 0.45:
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        if xray_confidence < min_xray_confidence and not medical_lut:
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        return XrayValidationResult(True, xray_confidence, "")

    return XrayValidationResult(True, mid, "")
