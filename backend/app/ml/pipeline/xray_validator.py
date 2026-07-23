"""Stage 1 — only bone / body-part clinical scans count as X-rays."""

from __future__ import annotations

import logging

from PIL import Image

from app.ml.image_gate import (
    BONE_BODY_LABELS,
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
    Accept only when bone / body-part structure is visible on a clinical scan
    (grayscale X-ray or blue/cyan CT/MRI panel).

    Person photos, selfies, text screenshots → "Please upload a medical image."
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])

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

    # Person / color camera photo → never an X-ray (even if model says brain tumor)
    if looks_like_person_or_color_photo(image):
        logger.info("Rejected person/color photo (color=%.1f mid=%.3f top=%s)", tones["color"], mid, top_label)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Text / UI without bone tissue
    if looks_like_text_without_anatomy(image):
        logger.info("Rejected text without bones (mid=%.3f)", mid)
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if looks_like_photo_or_text(image) and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    # Must look like a clinical display (gray film or PACS blue LUT)
    if not medical_lut:
        logger.info("Rejected non-clinical display (not grayscale/PACS LUT)")
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if mid < min_anatomy_midtone:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        # Bones / body part on clinical scan → accept (before non-clinical top-label checks)
        if best_prob >= 0.20 and mid >= 0.14:
            logger.info("Bone/body accepted: %s=%.3f mass=%.3f", best_label, best_prob, bone_mass)
            return XrayValidationResult(True, max(best_prob, bone_mass), "")

        if bone_mass >= 0.35 and mid >= 0.14:
            return XrayValidationResult(True, bone_mass, "")

        # Skin / eye / unsupported alone — not a bone X-ray
        if top_label in NON_CLINICAL_LABELS and bone_mass < 0.30:
            return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

        # Clinical LUT with enough mid-tones and some bone signal (MRI/CT panels)
        if medical_lut and mid >= 0.20 and bone_mass >= 0.18:
            return XrayValidationResult(True, max(bone_mass, mid), "")

        return XrayValidationResult(False, best_prob, MSG_NOT_XRAY)

    # Heuristic-only: clinical LUT + tissue mid-tones
    if medical_lut and mid >= 0.22:
        return XrayValidationResult(True, mid, "")
    return XrayValidationResult(False, mid, MSG_NOT_XRAY)
