"""Accept trained X-ray/medical classes; reject text & non-medical uploads."""

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


def _gray_array(image: Image.Image, size: int = 192) -> np.ndarray:
    small = image.convert("RGB").resize((size, size))
    arr = np.asarray(small, dtype=np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _color_delta(image: Image.Image) -> float:
    small = image.convert("RGB").resize((128, 128))
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)


def _highpass_energy(gray: np.ndarray) -> float:
    core = gray[1:-1, 1:-1]
    blur = (
        gray[:-2, :-2]
        + gray[:-2, 1:-1]
        + gray[:-2, 2:]
        + gray[1:-1, :-2]
        + gray[1:-1, 1:-1]
        + gray[1:-1, 2:]
        + gray[2:, :-2]
        + gray[2:, 1:-1]
        + gray[2:, 2:]
    ) / 9.0
    return float(np.mean(np.abs(core - blur)))


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """
    Reject clear text documents and colorful photos.
    Real chest X-rays (open samples) stay accepted: dark, near-grayscale, smoother.
    """
    gray = _gray_array(image)
    bright = float((gray > 220).mean())
    hp = _highpass_energy(gray)
    color = _color_delta(image)

    # Paper / text document (very bright page)
    if bright >= 0.50 and hp >= 3.0:
        return True
    if bright >= 0.70:
        return True
    # Colorful everyday photos (X-rays are near grayscale)
    if color >= 25.0:
        return True
    return False


def is_medical_prediction(
    label: str,
    confidence: float,
    probs: dict[str, float],
    min_confidence: float,
    image: Image.Image | None = None,
) -> bool:
    """
    If trained medical/X-ray class with enough confidence → return findings.
    Else → Please upload medical image.
    """
    name = label.upper().strip()
    if name == "UNSUPPORTED" or name not in MEDICAL_LABELS:
        return False
    if confidence < min_confidence:
        return False
    if float(probs.get("UNSUPPORTED", 0.0)) >= 0.45:
        return False
    if image is not None and looks_like_photo_or_text(image):
        return False
    return True
