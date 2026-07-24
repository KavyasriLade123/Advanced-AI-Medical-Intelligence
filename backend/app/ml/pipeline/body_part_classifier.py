"""Stage 2 — body-part classification for validated medical images."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from app.config import MODEL_DIR
from app.ml.pipeline.catalog import (
    BODY_PARTS,
    BODY_PART_MODEL_CLASSES,
    LABEL_TO_BODY_DISEASE,
    MSG_UNKNOWN_PART,
    format_body_part_label,
)

logger = logging.getLogger(__name__)


class BodyPartResult:
    __slots__ = ("body_part_id", "display_name", "confidence", "message", "ok", "modality")

    def __init__(
        self,
        ok: bool,
        body_part_id: str = "",
        display_name: str = "",
        confidence: float = 0.0,
        message: str = "",
        modality: str = "X-ray",
    ) -> None:
        self.ok = ok
        self.body_part_id = body_part_id
        self.display_name = display_name
        self.confidence = confidence
        self.message = message
        self.modality = modality


def infer_modality(image: Image.Image) -> str:
    """Heuristic modality tag for display (X-ray vs CT/MRI panel)."""
    arr = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    chroma = float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)
    if float(b.mean()) >= float(r.mean()) + 15.0 and chroma >= 12.0:
        return "MRI"
    if chroma < 10.0:
        return "X-ray"
    return "X-ray"


def _refine_bone_part(image: Image.Image, fallback_id: str = "bone") -> str:
    """
    Content-aware extremity refine (hand vs knee vs long bone).

    Avoids pure aspect-ratio guessing that flips the same hand between hand/knee.
    """
    arr = np.asarray(image.convert("L").resize((160, 160)), dtype=np.float32)
    h, w = arr.shape
    aspect = float(image.size[1]) / float(max(image.size[0], 1))

    # Edge energy (hands/feet have many small bones → higher high-frequency texture)
    gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
    gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
    edge = float((gx + gy) / 2.0)

    # Bright bone-like pixel fraction
    bright = float((arr > 150).mean())
    mid = float(((arr >= 40) & (arr <= 200)).mean())

    # Local variance: many phalanges → more busy blocks
    block = 16
    vars_: list[float] = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            vars_.append(float(arr[y : y + block, x : x + block].var()))
    busy = float(np.mean([v > 350 for v in vars_])) if vars_ else 0.0

    # Hand / wrist / fingers: high edge + busy texture
    if edge >= 12.0 and busy >= 0.35 and mid >= 0.25:
        if aspect <= 0.85:
            return "hand"
        return "hand" if bright >= 0.08 else "wrist"

    # Foot / ankle: busy but often wider
    if edge >= 11.0 and busy >= 0.30 and aspect <= 0.95 and bright >= 0.05:
        return "foot"

    # Long bone films
    if aspect >= 1.45:
        return "leg" if aspect < 1.9 else "femur"

    # Knee / hip joints: lower edge density, larger soft-tissue region, near-square
    if 0.80 <= aspect <= 1.25 and edge < 12.0 and busy < 0.40:
        return "knee"

    return fallback_id


class BodyPartClassifier:
    """
    Prefers dedicated checkpoint: backend/models/body_part_resnet18.pth
    Otherwise maps unified classifier + geometry heuristics.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = MODEL_DIR / "body_part_resnet18.pth"
        self.labels_meta = MODEL_DIR / "body_part_labels.json"
        self.labels = list(BODY_PART_MODEL_CLASSES)
        if self.labels_meta.exists():
            try:
                import json

                meta = json.loads(self.labels_meta.read_text(encoding="utf-8"))
                classes = meta.get("classes") or []
                if isinstance(classes, list) and classes:
                    self.labels = [str(c) for c in classes]
            except Exception:
                pass
        self.model: nn.Module | None = None
        if self.weights_path.exists():
            self.model = self._build_model()
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Loaded dedicated body-part model from %s (%d classes)", self.weights_path, len(self.labels))

    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.labels))
        return model

    def _pack(self, part_id: str, confidence: float, modality: str) -> BodyPartResult:
        spec = BODY_PARTS.get(part_id)
        if not spec:
            return BodyPartResult(False, confidence=confidence, message=MSG_UNKNOWN_PART)
        label = format_body_part_label(part_id, modality)
        return BodyPartResult(True, part_id, label, confidence, modality=modality)

    @torch.inference_mode()
    def predict_from_tensor(self, tensor: torch.Tensor, image: Image.Image | None = None) -> BodyPartResult:
        if self.model is None:
            return BodyPartResult(False, message=MSG_UNKNOWN_PART)
        logits = self.model(tensor.to(self.device))
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        part_id = self.labels[idx]
        modality = infer_modality(image) if image is not None else "X-ray"
        # Only refine ambiguous generic "bone" — never override confident hand/knee/etc.
        if part_id == "bone" and image is not None:
            part_id = _refine_bone_part(image, "bone")
        elif part_id in {"hand", "knee", "wrist", "foot", "ankle"} and conf < 0.45 and image is not None:
            # Low-confidence extremity: re-check with content cues
            refined = _refine_bone_part(image, part_id)
            if refined in {"hand", "wrist", "foot", "knee", "ankle", "leg", "femur"}:
                part_id = refined
        if conf < 0.28:
            return BodyPartResult(False, confidence=conf, message=MSG_UNKNOWN_PART)
        # Brain MRI panels
        if image is not None and modality == "MRI" and part_id in {"skull", "bone", "chest"}:
            if conf < 0.55:
                return self._pack("brain", max(conf, 0.45), "MRI")
        return self._pack(part_id, conf, modality if part_id != "brain" else "MRI")

    def predict_from_unified_probs(
        self,
        predicted: str,
        confidence: float,
        probs: dict[str, float],
        image: Image.Image | None = None,
    ) -> BodyPartResult:
        modality = infer_modality(image) if image is not None else "X-ray"
        mapped = LABEL_TO_BODY_DISEASE.get(predicted.upper())
        if mapped:
            part_id, _ = mapped
            if part_id == "bone" and image is not None:
                part_id = _refine_bone_part(image, "bone")
            if part_id == "skull" and modality == "MRI":
                part_id = "brain"
            if part_id == "brain":
                modality = "MRI"
            return self._pack(part_id, confidence, modality)

        brain_p = float(probs.get("BRAIN_TUMOR", 0.0)) + float(probs.get("BRAIN_NORMAL", 0.0))
        chest_p = (
            float(probs.get("PNEUMONIA", 0.0))
            + float(probs.get("NORMAL", 0.0))
            + float(probs.get("BREAST_NORMAL", 0.0))
            + float(probs.get("BREAST_MALIGNANT", 0.0))
        )
        bone_p = float(probs.get("BONE_FRACTURE", 0.0)) + float(probs.get("LOWER_LIMB", 0.0))
        abd_p = float(probs.get("ABDOMEN", 0.0))

        # Strongest anatomy group
        scores = {
            "brain": brain_p,
            "chest": chest_p,
            "bone": bone_p,
            "abdomen": abd_p,
            "leg": float(probs.get("LOWER_LIMB", 0.0)),
        }
        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]
        if best_score < 0.10:
            # Still return a part so Stage 3 can run on validated medical images
            best_id, best_score = "chest", max(confidence, 0.2)

        if best_id == "bone" and image is not None:
            best_id = _refine_bone_part(image, "bone")
        if best_id == "brain":
            modality = "MRI"
        elif modality == "MRI" and best_id in {"skull", "chest"} and brain_p >= 0.15:
            best_id, modality = "brain", "MRI"

        return self._pack(best_id, float(best_score), modality)
