"""Stage 1 — strict medical X-ray validation only (confidence >= 95%).

Does not change disease prediction. Rejects screenshots, photos, and any
uncertain image. Accepts only when the X-ray detector is highly confident
and the image looks like a radiology film.
"""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import (
    NON_CLINICAL_LABELS,
    _tone_stats,
    looks_like_person_or_color_photo,
    looks_like_photo_or_text,
    looks_like_text_without_anatomy,
    looks_like_ui_screenshot,
)
from app.ml.pipeline.catalog import MSG_NOT_XRAY
from app.ml.pipeline.xray_detector import get_xray_detector

logger = logging.getLogger(__name__)

# User requirement: reject unless X-ray confidence >= 95%
XRAY_MIN_CONFIDENCE = 0.95


class XrayValidationResult:
    __slots__ = ("is_xray", "confidence", "message")

    def __init__(self, is_xray: bool, confidence: float, message: str = "") -> None:
        self.is_xray = is_xray
        self.confidence = confidence
        self.message = message


def _looks_like_radiograph(tones: dict[str, float]) -> bool:
    """
    Radiology film traits: anatomical mid-tones, no skin photo signal.
    Accepts grayscale X-rays and clinical color/PACS-tinted films.
    """
    color = float(tones["color"])
    corr = float(tones.get("corr", 0.0))
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])
    skin = float(tones.get("skin", 0.0))
    sat = float(tones.get("sat", 0.0))
    flat = float(tones.get("flat", 0.0))
    busy = float(tones.get("busy", 0.0))

    if skin >= 0.025:
        return False
    if mid < 0.14:
        return False
    # True grayscale radiographic film
    if color < 10.0 and corr >= 0.97 and mid >= 0.16:
        return True
    # Clinical color / blue PACS radiology display (still a real scan, not a photo)
    if medical_lut and corr >= 0.90 and skin < 0.01 and mid >= 0.16 and flat < 0.35:
        return True
    # Mild clinical tint with textured anatomy (not flat UI chrome)
    if color < 40.0 and corr >= 0.90 and sat < 22.0 and mid >= 0.20 and busy >= 0.25 and flat < 0.30:
        return True
    return False


def validate_medical_xray(
    image: Image.Image,
    class_probs: dict[str, float] | None = None,
    *,
    min_xray_confidence: float = 0.35,
    min_anatomy_midtone: float = 0.16,
) -> XrayValidationResult:
    """
    Extremely strict gate:
      - Reject UI / screenshots / people / documents immediately.
      - Accept only if trained X-ray detector P(xray) >= 0.95
        AND image traits match a radiograph.
      - Any uncertainty → reject (do not guess).
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    color = float(tones["color"])
    corr = float(tones.get("corr", 0.0))
    flat = float(tones.get("flat", 0.0))
    busy = float(tones.get("busy", 0.0))
    skin = float(tones.get("skin", 0.0))
    medical_lut = bool(tones["medical_lut"])

    top_label = ""
    top_prob = 0.0
    skin_class = 0.0
    unsupported = 0.0
    if class_probs:
        top_label, top_prob = max(class_probs.items(), key=lambda kv: float(kv[1]))
        top_label = top_label.upper()
        top_prob = float(top_prob)
        skin_class = float(class_probs.get("SKIN", 0.0))
        unsupported = float(class_probs.get("UNSUPPORTED", 0.0))

    # --- Pre-AI / heuristic rejects (screenshots, photos, text) ---
    if looks_like_ui_screenshot(image):
        logger.info("Rejected UI/screenshot flat=%.3f busy=%.3f", flat, busy)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_person_or_color_photo(image) or skin >= 0.025:
        logger.info("Rejected person/photo skin=%.3f color=%.1f", skin, color)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_text_without_anatomy(image):
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Ordinary photos / text docs — but not clinical PACS-tinted scans
    if looks_like_photo_or_text(image) and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Colored desktop / UI chrome (allow medical LUT radiology films)
    if color >= 10.0 and corr < 0.90 and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Low channel correlation = UI chrome / photos, not a radiograph
    if corr < 0.88 and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Flat colored panels (Project Manager / IDE / dashboards)
    if flat >= 0.32 and color >= 8.0 and corr < 0.95:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if skin_class >= 0.15:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if top_label in NON_CLINICAL_LABELS and top_prob >= 0.35:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if unsupported >= 0.45:
        return XrayValidationResult(False, unsupported, MSG_NOT_XRAY)

    # Must look like a radiograph before trusting the detector
    if not _looks_like_radiograph(tones):
        logger.info(
            "Rejected non-radiograph traits mid=%.3f color=%.1f corr=%.3f lut=%s",
            mid,
            color,
            corr,
            medical_lut,
        )
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # --- Trained detector: require >= 95% ---
    detector = get_xray_detector()
    if not detector.available:
        logger.warning("X-ray gate weights missing — rejecting (strict mode, no guessing)")
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    det_p = detector.predict_proba(image)
    logger.info("Strict X-ray gate P(xray)=%.4f (need >= %.2f)", det_p, XRAY_MIN_CONFIDENCE)

    if det_p < XRAY_MIN_CONFIDENCE:
        return XrayValidationResult(False, det_p, MSG_NOT_XRAY)

    # Final double-check: still not a person/UI after all
    if looks_like_person_or_color_photo(image) or looks_like_ui_screenshot(image):
        return XrayValidationResult(False, det_p, MSG_NOT_XRAY)

    logger.info("Accepted as medical X-ray P=%.4f", det_p)
    return XrayValidationResult(True, det_p, "")
