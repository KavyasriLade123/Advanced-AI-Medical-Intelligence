"""Stage 1 — trained X-ray detector + hard rejects for people/UI/text."""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import (
    BONE_BODY_LABELS,
    MIN_BONE_CONFIDENCE,
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

# Trained detector thresholds (keep person/UI hard rejects above these)
DETECTOR_ACCEPT = 0.55
DETECTOR_REJECT = 0.35


class XrayValidationResult:
    __slots__ = ("is_xray", "confidence", "message")

    def __init__(self, is_xray: bool, confidence: float, message: str = "") -> None:
        self.is_xray = is_xray
        self.confidence = confidence
        self.message = message


def _bone_signal(class_probs: dict[str, float]) -> tuple[str, float, float]:
    best_label = ""
    best_prob = 0.0
    mass = 0.0
    for name, prob in class_probs.items():
        key = name.upper()
        if key not in BONE_BODY_LABELS:
            continue
        p = float(prob)
        mass += p
        if p > best_prob:
            best_label, best_prob = key, p
    return best_label, best_prob, mass


def validate_medical_xray(
    image: Image.Image,
    class_probs: dict[str, float] | None = None,
    *,
    min_xray_confidence: float = 0.35,
    min_anatomy_midtone: float = 0.16,
) -> XrayValidationResult:
    """
    1) Always reject person photos / UI screenshots / text-only (unchanged).
    2) Prefer trained X-ray detector when weights are present.
    3) Fall back to clinical LUT + bone-label heuristics.
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])
    color = float(tones["color"])

    best_label = ""
    best_prob = 0.0
    bone_mass = 0.0
    top_label = ""
    top_prob = 0.0

    if class_probs:
        top_label, top_prob = max(class_probs.items(), key=lambda kv: float(kv[1]))
        top_label = top_label.upper()
        top_prob = float(top_prob)
        best_label, best_prob, bone_mass = _bone_signal(class_probs)

    # --- Hard rejects (do not soften) ---
    if looks_like_ui_screenshot(image) or float(tones.get("flat", 0.0)) >= 0.45:
        logger.info("Rejected UI screenshot flat=%.3f", tones.get("flat", 0.0))
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_person_or_color_photo(image) or float(tones.get("skin", 0.0)) >= 0.012:
        logger.info("Rejected person/color photo skin=%.3f color=%.1f", tones.get("skin", 0.0), color)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_text_without_anatomy(image):
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # --- Trained detector ---
    detector = get_xray_detector()
    det_p = detector.predict_proba(image) if detector.available else None
    if det_p is not None:
        logger.info("X-ray detector P(xray)=%.3f top=%s bone=%.3f", det_p, top_label, best_prob)
        if det_p >= DETECTOR_ACCEPT:
            return XrayValidationResult(True, det_p, "")
        if det_p < DETECTOR_REJECT:
            return XrayValidationResult(False, det_p, MSG_NOT_XRAY)
        # Uncertain band: require clinical look + bone signal

    if looks_like_photo_or_text(image) and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if not medical_lut and (det_p is None or det_p < DETECTOR_ACCEPT):
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if mid < min_anatomy_midtone:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        if best_prob < MIN_BONE_CONFIDENCE and bone_mass < 0.55:
            # Detector uncertain + weak labels
            if det_p is not None and det_p >= 0.45 and medical_lut and mid >= 0.20:
                return XrayValidationResult(True, max(det_p, bone_mass), "")
            return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

        if top_label in NON_CLINICAL_LABELS and best_prob < MIN_BONE_CONFIDENCE:
            return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

        if best_prob >= MIN_BONE_CONFIDENCE and mid >= 0.14:
            return XrayValidationResult(True, max(best_prob, bone_mass), "")

        if bone_mass >= 0.55 and mid >= 0.18 and medical_lut:
            return XrayValidationResult(True, bone_mass, "")

        return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

    if medical_lut and mid >= 0.25 and color < 10.0:
        return XrayValidationResult(True, mid, "")
    return XrayValidationResult(False, mid, MSG_NOT_XRAY)
