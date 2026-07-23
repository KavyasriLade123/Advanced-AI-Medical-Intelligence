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
    Reject text documents / UI screenshots.
    Real chest X-rays are smoother (lower high-pass energy) even when dark.
    """
    gray = _gray_array(image)
    bright = float((gray > 220).mean())
    dark = float((gray < 45).mean())
    hp = _highpass_energy(gray)

    # White/light page with text (documents, notes, screenshots of articles)
    if bright >= 0.40 and hp >= 4.5:
        return True
    # Dark-theme text UI (not X-ray — X-rays ~hp 2.5 on samples)
    if dark >= 0.75 and hp >= 3.1:
        return True
    # Extreme glyph detail
    if hp >= 8.0:
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
    Give trained output when the model predicts a medical class confidently.
    Otherwise show: Please upload medical image.
    """
    name = label.upper().strip()
    if name == "UNSUPPORTED" or name not in MEDICAL_LABELS:
        return False
    if confidence < min_confidence:
        return False
    if float(probs.get("UNSUPPORTED", 0.0)) >= 0.45:
        return False
    # Block obvious text documents even if the model is confused
    if image is not None and looks_like_photo_or_text(image):
        return False
    return True
