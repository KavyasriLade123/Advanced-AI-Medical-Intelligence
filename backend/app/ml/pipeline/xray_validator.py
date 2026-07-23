"""Stage 1 — accept scans with bone/body anatomy; reject text-only uploads."""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import (
    XRAY_ANATOMY_LABELS,
    _tone_stats,
    looks_like_photo_or_text,
    looks_like_text_without_anatomy,
)
from app.ml.pipeline.catalog import MSG_NOT_XRAY, NON_XRAY_LABELS

logger = logging.getLogger(__name__)


class XrayValidationResult:
    __slots__ = ("is_xray", "confidence", "message")

    def __init__(self, is_xray: bool, confidence: float, message: str = "") -> None:
        self.is_xray = is_xray
        self.confidence = confidence
        self.message = message


def _anatomy_signal(class_probs: dict[str, float]) -> tuple[str, float, float]:
    """Return (best_anatomy_label, best_prob, anatomy_mass)."""
    best_label = ""
    best_prob = 0.0
    mass = 0.0
    for name, prob in class_probs.items():
        key = name.upper()
        if key not in XRAY_ANATOMY_LABELS:
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
    Accept when bone structure / body-part anatomy is present in the image.
    Reject text screenshots and non-medical photos → "Please upload medical image."
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])
    text_like = looks_like_photo_or_text(image)

    best_label = ""
    best_prob = 0.0
    anatomy_mass = 0.0
    top_label = ""
    top_prob = 0.0
    xray_confidence = mid

    if class_probs:
        top_label, top_prob = max(class_probs.items(), key=lambda kv: float(kv[1]))
        top_label = top_label.upper()
        top_prob = float(top_prob)
        non_xray_mass = sum(float(class_probs.get(k, 0.0)) for k in NON_XRAY_LABELS)
        xray_confidence = max(0.0, 1.0 - non_xray_mass)
        best_label, best_prob, anatomy_mass = _anatomy_signal(class_probs)

    # Text / UI without bone or body-part tissue → not an X-ray
    if looks_like_text_without_anatomy(image) and anatomy_mass < 0.20:
        logger.info("Rejected text without anatomy (mid=%.3f anatomy=%.3f)", mid, anatomy_mass)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Text / casual photo without enough tissue gray → reject
    # (model can falsely score "brain" on black+white text screenshots)
    if text_like and mid < 0.25:
        logger.info("Rejected text/non-medical (mid=%.3f anatomy=%.3f)", mid, anatomy_mass)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Bone / body-part signal + anatomical mid-tones → accept
    if class_probs and best_prob >= 0.12 and mid >= 0.14:
        logger.info("Anatomy accepted: %s=%.3f mass=%.3f mid=%.3f", best_label, best_prob, anatomy_mass, mid)
        return XrayValidationResult(True, max(xray_confidence, best_prob), "")

    if class_probs and anatomy_mass >= 0.25 and mid >= 0.14:
        return XrayValidationResult(True, max(xray_confidence, anatomy_mass), "")

    mid_needed = 0.12 if medical_lut else min_anatomy_midtone
    if mid < mid_needed:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        if top_label in NON_XRAY_LABELS and top_prob >= 0.40 and anatomy_mass < 0.25:
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        if medical_lut and mid >= 0.14:
            return XrayValidationResult(True, max(xray_confidence, mid), "")
        if xray_confidence < min_xray_confidence:
            return XrayValidationResult(False, xray_confidence, MSG_NOT_XRAY)
        return XrayValidationResult(True, xray_confidence, "")

    if medical_lut and mid >= 0.14:
        return XrayValidationResult(True, mid, "")
    return XrayValidationResult(False, mid, MSG_NOT_XRAY)
