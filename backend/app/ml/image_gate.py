"""Accept real medical/X-ray images; reject text screenshots & non-medical photos."""

from __future__ import annotations

import numpy as np
from PIL import Image

REJECT_MESSAGE = "Please upload a valid medical X-ray image."

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

# Radiograph-like classes from the unified model
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


def _tone_stats(image: Image.Image) -> dict[str, float]:
    arr = _rgb_array(image)
    gray = _gray_from_rgb(arr)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color = float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)
    w, h = image.size
    return {
        "dark": float((gray < 45).mean()),
        "mid": float(((gray >= 40) & (gray <= 210)).mean()),
        "bright": float((gray > 210).mean()),
        "color": color,
        "aspect": float(h) / float(max(w, 1)),
    }


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """
    Reject text/UI screenshots and colorful non-clinical photos.
    Do NOT reject phone photos of real X-rays that still have mid-gray anatomy.
    """
    s = _tone_stats(image)
    dark, mid, bright, color, aspect = s["dark"], s["mid"], s["bright"], s["color"], s["aspect"]

    # Black canvas + text/UI (almost no tissue gray)
    if mid < 0.25 and (dark + bright) >= 0.75:
        return True
    # Tall phone screenshot without anatomical mid-tones
    if aspect >= 1.60 and mid < 0.30:
        return True
    # Bright document / notes page
    if bright >= 0.55 and mid < 0.40:
        return True
    # Colorful consumer photo WITHOUT X-ray mid-tones
    if color >= 18.0 and mid < 0.35:
        return True
    # Very colorful non-clinical scene
    if color >= 35.0 and mid < 0.50:
        return True
    return False


def has_anatomical_midtones(image: Image.Image, minimum: float = 0.22) -> bool:
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
