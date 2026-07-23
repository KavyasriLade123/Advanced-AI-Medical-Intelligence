"""Lightweight checks to reject everyday photos / text screenshots."""

from __future__ import annotations

import numpy as np
from PIL import Image

REJECT_MESSAGE = "Please upload medical image."

# Classes that mean a supported medical study (not a random photo/text).
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


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """
    Reject obvious text screenshots / white documents before the model runs.
    Everyday photos are mainly filtered by the UNSUPPORTED class + confidence gate.
    """
    small = image.convert("RGB").resize((160, 160))
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    bright_ratio = float((gray > 230).mean())
    gx = float(np.abs(np.diff(gray, axis=1)).mean())
    gy = float(np.abs(np.diff(gray, axis=0)).mean())
    edge_strength = (gx + gy) / 2.0

    # White page / text screenshot (dense edges on bright background)
    if bright_ratio > 0.50 and edge_strength > 10.0:
        return True
    return False


def is_medical_prediction(label: str, confidence: float, probs: dict[str, float], min_confidence: float) -> bool:
    name = label.upper().strip()
    if name == "UNSUPPORTED" or name not in MEDICAL_LABELS:
        return False
    if confidence < min_confidence:
        return False
    # If the reject class is still strong, treat as non-medical.
    if float(probs.get("UNSUPPORTED", 0.0)) >= 0.30:
        return False
    return True
