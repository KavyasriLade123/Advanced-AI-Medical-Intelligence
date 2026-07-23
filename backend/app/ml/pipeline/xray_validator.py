"""Stage 1 — only bone / body-part clinical scans count as X-rays."""

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
)
from app.ml.pipeline.catalog import MSG_NOT_XRAY

logger = logging.getLogger(__name__)


class XrayValidationResult:
    __slots__ = ("is_xray", "confidence", "message")

    def __init__(self, is_xray: bool, confidence: float, message: str = "") -> None:
        self.is_xray = is_xray
        self.confidence = confidence
        self.message = message


def _bone_signal(class_probs: dict[str, float]) -> tuple[str, float, float]:
    """Return (best_bone_label, best_prob, bone_mass)."""
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
    Accept only clinical bone/body scans (grayscale X-ray or blue CT/MRI).
    Person photos and weak guesses (e.g. 23% brain tumor) are rejected.
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

    # Hard reject: color camera photos of people / rooms / clothes
    if looks_like_person_or_color_photo(image) or float(tones.get("skin", 0.0)) >= 0.012:
        logger.info(
            "Rejected person/color photo color=%.1f sat=%.1f skin=%.3f lut=%s top=%s conf=%.3f",
            color,
            tones.get("sat", 0.0),
            tones.get("skin", 0.0),
            medical_lut,
            top_label,
            top_prob,
        )
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_text_without_anatomy(image):
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_photo_or_text(image) and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Must be grayscale film or blue PACS panel
    if not medical_lut:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if mid < min_anatomy_midtone:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        # Low-confidence guesses on any image are not trusted as X-rays
        if best_prob < MIN_BONE_CONFIDENCE and bone_mass < 0.55:
            logger.info(
                "Rejected weak bone signal %s=%.3f mass=%.3f (need >=%.2f)",
                best_label,
                best_prob,
                bone_mass,
                MIN_BONE_CONFIDENCE,
            )
            return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

        if top_label in NON_CLINICAL_LABELS and best_prob < MIN_BONE_CONFIDENCE:
            return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

        if best_prob >= MIN_BONE_CONFIDENCE and mid >= 0.14:
            logger.info("Bone/body accepted: %s=%.3f mass=%.3f", best_label, best_prob, bone_mass)
            return XrayValidationResult(True, max(best_prob, bone_mass), "")

        if bone_mass >= 0.55 and mid >= 0.18 and medical_lut:
            return XrayValidationResult(True, bone_mass, "")

        return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

    if medical_lut and mid >= 0.25 and color < 10.0:
        return XrayValidationResult(True, mid, "")
    return XrayValidationResult(False, mid, MSG_NOT_XRAY)
