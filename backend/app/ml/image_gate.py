"""Accept real medical scans (X-ray / CT / MRI); reject text screenshots & casual photos."""

from __future__ import annotations

import numpy as np
from PIL import Image

REJECT_MESSAGE = "Please upload medical image."

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

# Radiograph / clinical scan classes from the unified model
XRAY_ANATOMY_LABELS = {
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

ANATOMY_LABELS = set(XRAY_ANATOMY_LABELS)


def _rgb_array(image: Image.Image, size: int = 192) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)


def _gray_from_rgb(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _is_medical_display_lut(arr: np.ndarray) -> bool:
    """
    True for grayscale films and common PACS color LUTs (blue/cyan MRI/CT panels).
    False for natural color photos / emoji-heavy social screenshots / black text canvases.
    """
    gray = _gray_from_rgb(arr)
    mid_frac = float(((gray >= 35) & (gray <= 220)).mean())
    # Text on black / empty canvas has almost no tissue mid-tones
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
    if mean_chroma < 14.0:
        return True
    # Blue/teal medical display: higher B but still highly correlated channels
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


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """
    True when the upload looks like text/UI or a casual photo — not a clinical scan.
    Watermarks on real X-ray/CT/MRI are OK when tissue mid-tones remain.
    """
    s = _tone_stats(image)
    dark, mid, bright, color, aspect = s["dark"], s["mid"], s["bright"], s["color"], s["aspect"]
    medical_lut = bool(s["medical_lut"])

    # Black canvas + text/UI (almost no tissue gray) → not an X-ray
    if mid < 0.22 and (dark + bright) >= 0.78:
        return True
    # Tall phone screenshot of text/social without anatomy
    if aspect >= 1.65 and mid < 0.28 and not medical_lut:
        return True
    # Bright document / notes page (text without bones)
    if bright >= 0.55 and mid < 0.40:
        return True
    # Natural color photo (not a medical LUT)
    if not medical_lut and color >= 20.0 and mid < 0.40:
        return True
    if not medical_lut and color >= 40.0:
        return True
    return False


def looks_like_text_without_anatomy(image: Image.Image) -> bool:
    """Text / UI content with no bone or body-part tissue → not an X-ray."""
    s = _tone_stats(image)
    mid = float(s["mid"])
    dark = float(s["dark"])
    bright = float(s["bright"])
    # High-contrast text pages / black canvases lack anatomical mid-gray
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
