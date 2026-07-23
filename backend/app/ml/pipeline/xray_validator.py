"""Stage 1 — trained X-ray detector + rejects for people/UI/text."""

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
    Prefer the trained X-ray detector for real films (including phone photos of X-rays).
    Still reject person photos, UI screenshots, and text-only uploads.
    """
    tones = _tone_stats(image)
    mid = float(tones["mid"])
    medical_lut = bool(tones["medical_lut"])
    color = float(tones["color"])
    corr = float(tones.get("corr", 0.0))
    flat = float(tones.get("flat", 0.0))
    busy = float(tones.get("busy", 0.0))
    skin = float(tones.get("skin", 0.0))

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

    detector = get_xray_detector()
    det_p = detector.predict_proba(image) if detector.available else None

    clear_ui = looks_like_ui_screenshot(image) or (
        flat >= 0.55 and busy <= 0.15 and color >= 8.0
    )
    # Grayscale bone films often have flat>=0.45 from black borders — not UI
    if flat >= 0.45 and color < 8.0 and corr >= 0.98:
        clear_ui = looks_like_ui_screenshot(image)
    clear_person = looks_like_person_or_color_photo(image) and not (corr >= 0.985 and color < 32.0)
    clear_text = looks_like_text_without_anatomy(image)

    bones_detected = best_label in BONE_BODY_LABELS and best_prob >= 0.28 and mid >= 0.18

    # Whenever bones / radiographic anatomy are detected → always return output
    # (color-tinted chest films with watermarks still show ribs/clavicles/spine)
    if bones_detected and not clear_ui:
        if clear_person and skin >= 0.15 and corr < 0.94 and (det_p is None or det_p < 0.70):
            # Real selfie with weak X-ray score — keep rejected
            pass
        else:
            logger.info(
                "Bones/anatomy accepted: %s=%.3f det=%s",
                best_label,
                best_prob,
                f"{det_p:.3f}" if det_p is not None else "n/a",
            )
            return XrayValidationResult(True, max(best_prob, det_p or 0.0, bone_mass), "")

    # Trained detector: accept real X-rays / MRI even if phone color cast looks like "skin"
    if det_p is not None and det_p >= DETECTOR_ACCEPT:
        if clear_ui and det_p < 0.90:
            logger.info("Rejected UI despite detector P=%.3f", det_p)
            return XrayValidationResult(False, det_p, MSG_NOT_XRAY)
        # Tinted clinical X-rays can match naive "skin" rules — trust detector + midtones
        if clear_person and skin >= 0.08 and corr < 0.95 and det_p < 0.85 and mid < 0.35:
            logger.info("Rejected person photo despite detector P=%.3f", det_p)
            return XrayValidationResult(False, det_p, MSG_NOT_XRAY)
        if clear_text and mid < 0.22 and det_p < 0.85:
            return XrayValidationResult(False, det_p, MSG_NOT_XRAY)
        logger.info("X-ray accepted by detector P=%.3f corr=%.3f", det_p, corr)
        return XrayValidationResult(True, det_p, "")

    # Hard rejects when detector is missing or unsure
    if clear_ui:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)
    if flat >= 0.50 and busy <= 0.15 and color >= 8.0:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)
    if clear_person or (skin >= 0.05 and corr < 0.98):
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)
    if clear_text and color >= 5.0:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if det_p is not None and det_p < DETECTOR_REJECT:
        return XrayValidationResult(False, det_p, MSG_NOT_XRAY)

    if looks_like_photo_or_text(image) and not medical_lut:
        return XrayValidationResult(False, 0.0, MSG_NOT_XRAY)

    if not medical_lut:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if mid < min_anatomy_midtone:
        return XrayValidationResult(False, mid, MSG_NOT_XRAY)

    if class_probs:
        if best_prob < MIN_BONE_CONFIDENCE and bone_mass < 0.55:
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
