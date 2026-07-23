"""Accept trained medical classes; reject text/non-medical with a clear message."""

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


def _row_transition_score(gray: np.ndarray) -> float:
    scores: list[float] = []
    for row in gray[::2]:
        thr = float(np.median(row))
        binary = row > thr
        scores.append(float(np.sum(binary[1:] != binary[:-1])))
    return float(np.mean(scores)) if scores else 0.0


def _highpass_energy(gray: np.ndarray) -> float:
    """Fine glyph/detail energy — text screenshots score much higher than X-rays."""
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
    Text/document detector only.
    Real X-rays are smooth (low high-pass energy) even with dark backgrounds.
    """
    gray = _gray_array(image)
    bright = float((gray > 220).mean())
    dark = float((gray < 50).mean())
    transitions = _row_transition_score(gray)
    hp = _highpass_energy(gray)

    # Light documents / notes
    if bright >= 0.35 and hp >= 5.0:
        return True
    # Dark UI / text screenshots (X-rays are dark but smooth → low hp)
    if dark >= 0.70 and hp >= 3.0 and transitions >= 3.0:
        return True
    # Strong fine-detail text
    if hp >= 7.0:
        return True
    if hp >= 4.0 and transitions >= 4.0:
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
    If model predicts a trained medical class with enough confidence → accept.
    Otherwise → reject (Please upload medical image).
    """
    name = label.upper().strip()
    if name == "UNSUPPORTED" or name not in MEDICAL_LABELS:
        return False
    if confidence < min_confidence:
        return False
    if float(probs.get("UNSUPPORTED", 0.0)) >= 0.40:
        return False
    if image is not None and looks_like_photo_or_text(image):
        return False
    return True
