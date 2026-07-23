"""Accept real medical scans (X-ray / CT / MRI); reject person photos & text."""

from __future__ import annotations

import numpy as np
from PIL import Image

REJECT_MESSAGE = "Please upload a valid medical image (X-ray, CT Scan, or MRI)."

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
MIN_BONE_CONFIDENCE = 0.40


def _rgb_array(image: Image.Image, size: int = 192) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)


def _gray_from_rgb(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _channel_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _skin_fraction(arr: np.ndarray) -> float:
    """Approximate share of skin-colored pixels (people photos)."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    skin = (
        (r > 60)
        & (g > 30)
        & (b > 15)
        & (r >= g)
        & (g >= b)
        & ((r - g) >= 10)
        & ((r - g) <= 100)
        & ((g - b) <= 80)
    )
    return float(skin.mean())


def _ui_panel_stats(gray: np.ndarray, block: int = 16) -> tuple[float, float]:
    """
    Flat UI panels (IDE/web screenshots) have many low-variance blocks.
    Real X-ray/MRI tissue has continuous texture → fewer flat blocks.
    Returns (flat_fraction, busy_fraction).
    """
    h, w = gray.shape
    vars_: list[float] = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            vars_.append(float(gray[y : y + block, x : x + block].var()))
    if not vars_:
        return 0.0, 0.0
    v = np.asarray(vars_, dtype=np.float32)
    return float((v < 45.0).mean()), float((v > 500.0).mean())


def looks_like_ui_screenshot(image: Image.Image) -> bool:
    """Dark IDE / website / app screenshots with flat panels — not an X-ray."""
    arr = _rgb_array(image, size=256)
    gray = _gray_from_rgb(arr)
    flat, busy = _ui_panel_stats(gray)
    dark = float((gray < 50).mean())
    mid = float(((gray >= 35) & (gray <= 220)).mean())
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_chroma = float(
        (np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0
    )
    # Extremity bone / chest films: large black borders look "flat" but are grayscale clinical
    if mean_chroma < 8.0 and mid >= 0.18:
        return False

    # Flat chrome + dark theme (VS Code, dashboards, etc.)
    if flat >= 0.45 and busy <= 0.35:
        return True
    if flat >= 0.40 and dark >= 0.55 and busy <= 0.25:
        return True
    # Mostly black canvas with sparse UI/text (not textured MRI/CT tissue)
    if dark >= 0.70 and mid < 0.35 and flat >= 0.30 and busy < 0.25:
        return True
    return False


def _is_medical_display_lut(arr: np.ndarray) -> bool:
    """
    True ONLY for near-grayscale X-ray films or blue/cyan PACS CT/MRI panels.
    Person photos and flat UI screenshots always return False.
    """
    gray = _gray_from_rgb(arr)
    mid_frac = float(((gray >= 35) & (gray <= 220)).mean())
    if mid_frac < 0.12:
        return False
    if _skin_fraction(arr) >= 0.015:
        return False

    r, g, b = arr[:, :, 0].ravel(), arr[:, :, 1].ravel(), arr[:, :, 2].ravel()
    chroma = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
    mean_chroma = float(chroma.mean())

    # Flat UI chrome (colored) — not grayscale bone films with black borders
    flat, busy = _ui_panel_stats(gray if gray.shape[0] >= 128 else _gray_from_rgb(arr))
    if flat >= 0.45 and busy <= 0.35 and mean_chroma >= 8.0:
        return False

    r_m, g_m, b_m = float(r.mean()), float(g.mean()), float(b.mean())

    if mean_chroma < 10.0:
        # Dark IDE themes are near-gray but not X-rays
        dark = float((gray < 45).mean())
        if dark >= 0.55 and mid_frac < 0.45 and flat >= 0.50 and busy <= 0.15:
            return False
        return True

    mean_corr = (_channel_corr(r, g) + _channel_corr(r, b) + _channel_corr(g, b)) / 3.0
    blue_dominant = b_m >= r_m + 15.0 and b_m >= g_m + 6.0
    if blue_dominant and mean_corr >= 0.90 and 10.0 <= mean_chroma < 65.0:
        return True
    return False


def _tone_stats(image: Image.Image) -> dict[str, float]:
    arr = _rgb_array(image)
    gray = _gray_from_rgb(arr)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color = float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)
    sat = float(np.mean(np.abs(r - gray) + np.abs(g - gray) + np.abs(b - gray)) / 3.0)
    flat, busy = _ui_panel_stats(_gray_from_rgb(_rgb_array(image, size=256)))
    corr = (
        _channel_corr(r.ravel(), g.ravel())
        + _channel_corr(r.ravel(), b.ravel())
        + _channel_corr(g.ravel(), b.ravel())
    ) / 3.0
    w, h = image.size
    return {
        "dark": float((gray < 45).mean()),
        "mid": float(((gray >= 35) & (gray <= 220)).mean()),
        "bright": float((gray > 220).mean()),
        "color": color,
        "sat": sat,
        "skin": _skin_fraction(arr),
        "flat": flat,
        "busy": busy,
        "corr": float(corr),
        "aspect": float(h) / float(max(w, 1)),
        "medical_lut": 1.0 if _is_medical_display_lut(arr) else 0.0,
        "r_mean": float(r.mean()),
        "g_mean": float(g.mean()),
        "b_mean": float(b.mean()),
    }


def looks_like_person_or_color_photo(image: Image.Image) -> bool:
    """True for selfies / group photos. Phone photos of X-rays return False."""
    s = _tone_stats(image)
    skin = float(s["skin"])
    color = float(s["color"])
    sat = float(s["sat"])
    medical_lut = bool(s["medical_lut"])
    corr = float(s["corr"])
    blue_gap = float(s["b_mean"]) - float(s["r_mean"])

    # Phone photo of an X-ray: channels stay highly correlated (same anatomy in R/G/B)
    if corr >= 0.985 and color < 32.0:
        return False

    if skin >= 0.012:
        return True

    if color >= 10.0 and blue_gap < 18.0 and corr < 0.985:
        return True
    if sat >= 6.0 and blue_gap < 18.0 and corr < 0.985:
        return True

    if medical_lut and color < 12.0 and skin < 0.01:
        return False
    if medical_lut and blue_gap >= 15.0 and skin < 0.01:
        return False

    if color >= 8.0 or sat >= 6.0:
        return True
    return False


def looks_like_photo_or_text(image: Image.Image) -> bool:
    s = _tone_stats(image)
    dark, mid, bright, color, aspect = s["dark"], s["mid"], s["bright"], s["color"], s["aspect"]
    medical_lut = bool(s["medical_lut"])

    if looks_like_ui_screenshot(image):
        return True
    if looks_like_person_or_color_photo(image):
        return True
    if mid < 0.22 and (dark + bright) >= 0.78:
        return True
    if aspect >= 1.65 and mid < 0.28 and not medical_lut:
        return True
    if bright >= 0.55 and mid < 0.40:
        return True
    if not medical_lut and color >= 8.0:
        return True
    return False


def looks_like_text_without_anatomy(image: Image.Image) -> bool:
    s = _tone_stats(image)
    mid = float(s["mid"])
    dark = float(s["dark"])
    bright = float(s["bright"])
    if looks_like_ui_screenshot(image):
        return True
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
