"""Accept real medical scans (X-ray / CT / MRI); reject person photos & text."""

from __future__ import annotations

import numpy as np
from PIL import Image

REJECT_MESSAGE = "Please upload a medical image."

MEDICAL_LABELS = {
    "ABDOMEN",
    "BONE_FRACTURE",
    "BRAIN_NORMAL",
    "BRAIN_TUMOR",
    "BREAST_MALIGNANT",
    "BREAST_NORMAL",
    "EYE_RETINA",
    "LOWER_LIMB",
    "NORMAL",
    "PNEUMONIA",
    "SKIN",
}

# Classes that imply bone / radiographic anatomy (not skin/eye photos)
BONE_BODY_LABELS = {
    "ABDOMEN",
    "BONE_FRACTURE",
    "BRAIN_NORMAL",
    "BRAIN_TUMOR",
    "BREAST_MALIGNANT",
    "BREAST_NORMAL",
    "LOWER_LIMB",
    "NORMAL",
    "PNEUMONIA",
}

XRAY_ANATOMY_LABELS = set(BONE_BODY_LABELS)
ANATOMY_LABELS = set(BONE_BODY_LABELS)

NON_CLINICAL_LABELS = {"UNSUPPORTED", "SKIN", "EYE_RETINA"}


def _rgb_array(image: Image.Image, size: int = 192) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)


def _gray_from_rgb(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _is_medical_display_lut(arr: np.ndarray) -> bool:
    """
    True for grayscale films and blue/cyan PACS CT/MRI panels.
    False for color person photos and casual camera shots.
    """
    gray = _gray_from_rgb(arr)
    mid_frac = float(((gray >= 35) & (gray <= 220)).mean())
    if mid_frac < 0.12:
        return False

    r, g, b = arr[:, :, 0].ravel(), arr[:, :, 1].ravel(), arr[:, :, 2].ravel()
    chroma = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
    mean_chroma = float(chroma.mean())

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
            return 1.0
        return float(np.corrcoef(a, b)[0, 1])

    rg, rb, gb = _corr(r, g), _corr(r, b), _corr(g, b)
    mean_corr = (rg + rb + gb) / 3.0

    # Near-grayscale film
    if mean_chroma < 14.0:
        return True
    # Blue/teal clinical LUT with correlated channels
    if mean_corr >= 0.92 and mean_chroma < 55.0:
        return True
    if mean_corr >= 0.88 and float(b.mean()) >= float(r.mean()) and mean_chroma < 70.0:
        return True
    return False


def _tone_stats(image: Image.Image) -> dict[str, float]:
    arr = _rgb_array(image)
    gray = _gray_from_rgb(arr)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color = float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)
    w, h = image.size
    return {
        "dark": float((gray < 45).mean()),
        "mid": float(((gray >= 35) & (gray <= 220)).mean()),
        "bright": float((gray > 220).mean()),
        "color": color,
        "aspect": float(h) / float(max(w, 1)),
        "medical_lut": 1.0 if _is_medical_display_lut(arr) else 0.0,
    }


def looks_like_person_or_color_photo(image: Image.Image) -> bool:
    """True for selfies / casual color photos (no clinical grayscale or blue LUT)."""
    s = _tone_stats(image)
    if bool(s["medical_lut"]):
        return False
    # Any non-clinical color image is treated as a person/nature photo
    return float(s["color"]) >= 12.0


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """True for text/UI or casual photos — not a clinical bone scan."""
    s = _tone_stats(image)
    dark, mid, bright, color, aspect = s["dark"], s["mid"], s["bright"], s["color"], s["aspect"]
    medical_lut = bool(s["medical_lut"])

    if looks_like_person_or_color_photo(image):
        return True
    if mid < 0.22 and (dark + bright) >= 0.78:
        return True
    if aspect >= 1.65 and mid < 0.28 and not medical_lut:
        return True
    if bright >= 0.55 and mid < 0.40:
        return True
    if not medical_lut and color >= 20.0 and mid < 0.40:
        return True
    if not medical_lut and color >= 40.0:
        return True
    return False


def looks_like_text_without_anatomy(image: Image.Image) -> bool:
    """Text / UI with no bone tissue → not an X-ray."""
    s = _tone_stats(image)
    mid = float(s["mid"])
    dark = float(s["dark"])
    bright = float(s["bright"])
    if mid < 0.20 and (dark + bright) >= 0.75:
        return True
    if bright >= 0.50 and mid < 0.35:
        return True
    return looks_like_photo_or_text(image) and mid < 0.28


def has_anatomical_midtones(image: Image.Image, minimum: float = 0.18) -> bool:
    return _tone_stats(image)["mid"] >= minimum


def is_medical_prediction(
    label: str,
    confidence: float,
    probs: dict[str, float],
    min_confidence: float,
    image: Image.Image | None = None,
) -> bool:
    name = label.upper().strip()
    if name == "UNSUPPORTED" or name not in MEDICAL_LABELS:
        return False
    if confidence < min_confidence:
        return False
    if float(probs.get("UNSUPPORTED", 0.0)) >= 0.45:
        return False
    if image is not None and looks_like_photo_or_text(image):
        return False
    if image is not None and name in ANATOMY_LABELS and not has_anatomical_midtones(image):
        return False
    return True
