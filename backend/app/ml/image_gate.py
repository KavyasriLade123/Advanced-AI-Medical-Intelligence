"""Reject everyday photos / text screenshots before returning clinical findings."""

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

# These studies are almost always near-grayscale clinical frames.
GRAYSCALE_CLINICAL_LABELS = {
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


def _gray_array(image: Image.Image, size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    small = image.convert("RGB").resize((size, size))
    arr = np.asarray(small, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return gray, arr


def _color_delta(arr: np.ndarray) -> float:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)


def _edge_maps(gray: np.ndarray) -> tuple[float, float]:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    strength = float((gx.mean() + gy.mean()) / 2.0)
    ratio = float(((gx > 15.0).mean() + (gy > 15.0).mean()) / 2.0)
    return strength, ratio


def _row_transition_score(gray: np.ndarray) -> float:
    """Character-like ink flips along rows (high for text)."""
    scores: list[float] = []
    for row in gray[::2]:
        thr = float(np.median(row))
        binary = row > thr
        scores.append(float(np.sum(binary[1:] != binary[:-1])))
    return float(np.mean(scores)) if scores else 0.0


def looks_like_photo_or_text(image: Image.Image) -> bool:
    """
    Reject text screenshots / documents / UI captures.
    Tuned so dark or light text pages do not get labeled as brain tumor.
    """
    gray, arr = _gray_array(image)
    color_delta = _color_delta(arr)
    bright_ratio = float((gray > 210).mean())
    dark_ratio = float((gray < 50).mean())
    edge_strength, edge_ratio = _edge_maps(gray)
    transitions = _row_transition_score(gray)

    # White / light document with text
    if bright_ratio > 0.28 and (edge_ratio > 0.045 or transitions > 10):
        return True
    # Dark-theme text / code / notes screenshot
    if dark_ratio > 0.28 and (edge_ratio > 0.045 or transitions > 6):
        return True
    # Dense character-like transitions (any background)
    if transitions > 14 or edge_ratio > 0.10:
        return True
    if edge_strength > 12.0 and transitions > 8:
        return True
    # Near-grayscale but still text-like (common failure mode → BRAIN_TUMOR)
    if color_delta < 10.0 and edge_ratio > 0.05 and (bright_ratio > 0.2 or dark_ratio > 0.2):
        return True
    # Colorful consumer photos
    if color_delta > 26.0 and edge_ratio > 0.04:
        return True
    return False


def looks_like_clinical_grayscale(image: Image.Image) -> bool:
    """Chest/brain/bone studies should be near-grayscale."""
    _, arr = _gray_array(image, size=128)
    return _color_delta(arr) < 16.0


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
    if confidence < max(min_confidence, 0.62):
        return False

    unsupported = float(probs.get("UNSUPPORTED", 0.0))
    if unsupported >= 0.15:
        return False

    ranked = sorted(probs.values(), reverse=True)
    second = ranked[1] if len(ranked) > 1 else 0.0
    if confidence - second < 0.12:
        return False

    # Never accept a clinical grayscale class on a text-like frame.
    if image is not None and looks_like_photo_or_text(image):
        return False

    if image is not None and name in GRAYSCALE_CLINICAL_LABELS:
        if not looks_like_clinical_grayscale(image):
            return False

    return True
