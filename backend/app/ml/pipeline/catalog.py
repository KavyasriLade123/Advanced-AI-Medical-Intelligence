"""Expandable catalog: body parts, diseases, modality labels, pipeline messages.

Add new parts/diseases here without changing pipeline orchestration code.
Optional weights: backend/models/body_part_resnet18.pth
                 backend/models/disease/{body_part_id}.pth
"""

from __future__ import annotations

from dataclasses import dataclass


MSG_NOT_MEDICAL = "Please upload a valid medical X-ray image."
# Back-compat aliases used across the codebase
MSG_NOT_XRAY = MSG_NOT_MEDICAL
MSG_UNKNOWN_PART = "Unable to identify the body part. Please upload a clearer medical X-ray."


@dataclass(frozen=True)
class DiseaseSpec:
    name: str
    recommendation: str


@dataclass(frozen=True)
class BodyPartSpec:
    id: str
    display_name: str
    modality_default: str  # "X-ray" | "CT" | "MRI"
    diseases: tuple[DiseaseSpec, ...]
    source_labels: tuple[str, ...] = ()
    weights_file: str | None = None


def _d(name: str, rec: str) -> DiseaseSpec:
    return DiseaseSpec(name, rec)


_ORTHO = "Please consult an orthopedic specialist."
_PULM = "Please consult a pulmonologist."
_NEURO = "Please consult a neurologist or neurosurgeon."
_SPINE = "Please consult an orthopedic spine specialist."
_RADIO = "Please consult a radiologist for further evaluation."


BODY_PARTS: dict[str, BodyPartSpec] = {
    "brain": BodyPartSpec(
        id="brain",
        display_name="Brain",
        modality_default="MRI",
        source_labels=("BRAIN_NORMAL", "BRAIN_TUMOR"),
        diseases=(
            _d("Normal", "No significant intracranial abnormality suggested."),
            _d("Brain tumor / mass", _NEURO),
            _d("Hemorrhage", "Please seek urgent neurological evaluation."),
            _d("Stroke", "Please seek urgent stroke / neurological evaluation."),
        ),
        weights_file="disease/brain.pth",
    ),
    "skull": BodyPartSpec(
        id="skull",
        display_name="Skull",
        modality_default="X-ray",
        source_labels=(),
        diseases=(
            _d("Normal", "No significant skull abnormality suggested."),
            _d("Fracture", _ORTHO),
            _d("Brain tumor / mass", _NEURO),
        ),
        weights_file="disease/skull.pth",
    ),
    "chest": BodyPartSpec(
        id="chest",
        display_name="Chest",
        modality_default="X-ray",
        source_labels=("NORMAL", "PNEUMONIA", "BREAST_NORMAL", "BREAST_MALIGNANT"),
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Pneumonia", _PULM),
            _d("Tuberculosis", _PULM),
            _d("COVID-19", "Please consult a pulmonologist / infectious disease specialist."),
            _d("Lung opacity", _PULM),
            _d("Pleural effusion", _PULM),
            _d("Pneumothorax", "Please seek urgent chest evaluation."),
            _d("Breast abnormality", "Please consult a radiologist or breast specialist."),
        ),
        weights_file="disease/chest.pth",
    ),
    "lungs": BodyPartSpec(
        id="lungs",
        display_name="Lungs",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Pneumonia", _PULM),
            _d("Tuberculosis", _PULM),
            _d("COVID-19", "Please consult a pulmonologist / infectious disease specialist."),
            _d("Lung opacity", _PULM),
        ),
        weights_file="disease/lungs.pth",
    ),
    "heart": BodyPartSpec(
        id="heart",
        display_name="Heart",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant cardiac silhouette abnormality suggested."),
            _d("Cardiomegaly", "Please consult a cardiologist."),
        ),
        weights_file="disease/heart.pth",
    ),
    "cervical_spine": BodyPartSpec(
        id="cervical_spine",
        display_name="Cervical Spine",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Disc degeneration", _SPINE),
            _d("Compression fracture", _SPINE),
            _d("Scoliosis", _SPINE),
        ),
        weights_file="disease/cervical_spine.pth",
    ),
    "thoracic_spine": BodyPartSpec(
        id="thoracic_spine",
        display_name="Thoracic Spine",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Disc degeneration", _SPINE),
            _d("Compression fracture", _SPINE),
            _d("Scoliosis", _SPINE),
        ),
        weights_file="disease/thoracic_spine.pth",
    ),
    "lumbar_spine": BodyPartSpec(
        id="lumbar_spine",
        display_name="Lumbar Spine",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Disc degeneration", _SPINE),
            _d("Compression fracture", _SPINE),
            _d("Scoliosis", _SPINE),
        ),
        weights_file="disease/lumbar_spine.pth",
    ),
    "spine": BodyPartSpec(
        id="spine",
        display_name="Spine",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Scoliosis", _SPINE),
            _d("Disc degeneration", _SPINE),
            _d("Compression fracture", _SPINE),
        ),
        weights_file="disease/spine.pth",
    ),
    "shoulder": BodyPartSpec(
        id="shoulder",
        display_name="Shoulder",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Dislocation", _ORTHO),
        ),
        weights_file="disease/shoulder.pth",
    ),
    "clavicle": BodyPartSpec(
        id="clavicle",
        display_name="Clavicle",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/clavicle.pth",
    ),
    "arm": BodyPartSpec(
        id="arm",
        display_name="Arm",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Hairline fracture", _ORTHO),
        ),
        weights_file="disease/arm.pth",
    ),
    "elbow": BodyPartSpec(
        id="elbow",
        display_name="Elbow",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Dislocation", _ORTHO),
        ),
        weights_file="disease/elbow.pth",
    ),
    "forearm": BodyPartSpec(
        id="forearm",
        display_name="Forearm",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/forearm.pth",
    ),
    "wrist": BodyPartSpec(
        id="wrist",
        display_name="Wrist",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Hairline fracture", _ORTHO),
        ),
        weights_file="disease/wrist.pth",
    ),
    "hand": BodyPartSpec(
        id="hand",
        display_name="Hand",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Arthritis", "Please consult an orthopedist or rheumatologist."),
        ),
        weights_file="disease/hand.pth",
    ),
    "fingers": BodyPartSpec(
        id="fingers",
        display_name="Fingers",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/fingers.pth",
    ),
    "pelvis": BodyPartSpec(
        id="pelvis",
        display_name="Pelvis",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/pelvis.pth",
    ),
    "hip": BodyPartSpec(
        id="hip",
        display_name="Hip",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Arthritis", _ORTHO),
        ),
        weights_file="disease/hip.pth",
    ),
    "femur": BodyPartSpec(
        id="femur",
        display_name="Femur",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/femur.pth",
    ),
    "knee": BodyPartSpec(
        id="knee",
        display_name="Knee",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Osteoarthritis", _ORTHO),
            _d("Fracture", _ORTHO),
            _d("Joint narrowing", _ORTHO),
        ),
        weights_file="disease/knee.pth",
    ),
    "leg": BodyPartSpec(
        id="leg",
        display_name="Leg",
        modality_default="X-ray",
        source_labels=("LOWER_LIMB",),
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/leg.pth",
    ),
    "tibia": BodyPartSpec(
        id="tibia",
        display_name="Tibia",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/tibia.pth",
    ),
    "fibula": BodyPartSpec(
        id="fibula",
        display_name="Fibula",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/fibula.pth",
    ),
    "ankle": BodyPartSpec(
        id="ankle",
        display_name="Ankle",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/ankle.pth",
    ),
    "foot": BodyPartSpec(
        id="foot",
        display_name="Foot",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Arthritis", _ORTHO),
        ),
        weights_file="disease/foot.pth",
    ),
    "toes": BodyPartSpec(
        id="toes",
        display_name="Toes",
        modality_default="X-ray",
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
        ),
        weights_file="disease/toes.pth",
    ),
    "abdomen": BodyPartSpec(
        id="abdomen",
        display_name="Abdomen",
        modality_default="X-ray",
        source_labels=("ABDOMEN",),
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Abnormality", "Please consult a radiologist or gastroenterologist."),
        ),
        weights_file="disease/abdomen.pth",
    ),
    "bone": BodyPartSpec(
        id="bone",
        display_name="Bone",
        modality_default="X-ray",
        source_labels=("BONE_FRACTURE",),
        diseases=(
            _d("Normal", "No significant abnormality detected."),
            _d("Fracture", _ORTHO),
            _d("Hairline fracture", _ORTHO),
            _d("Dislocation", _ORTHO),
            _d("Osteoarthritis", _ORTHO),
            _d("Osteoporosis", _ORTHO),
            _d("Bone lesion", _RADIO),
        ),
        weights_file="disease/bone.pth",
    ),
}


# Unified MedIntel classifier label → (body_part_id, disease_name)
LABEL_TO_BODY_DISEASE: dict[str, tuple[str, str]] = {
    "NORMAL": ("chest", "Normal"),
    "PNEUMONIA": ("chest", "Pneumonia"),
    "BREAST_NORMAL": ("chest", "Normal"),
    "BREAST_MALIGNANT": ("chest", "Breast abnormality"),
    "BRAIN_NORMAL": ("brain", "Normal"),
    "BRAIN_TUMOR": ("brain", "Brain tumor / mass"),
    "BONE_FRACTURE": ("bone", "Fracture"),
    "LOWER_LIMB": ("leg", "Fracture"),
    "ABDOMEN": ("abdomen", "Abnormality"),
}

NON_XRAY_LABELS = {"UNSUPPORTED", "SKIN", "EYE_RETINA"}

# Classes used by optional dedicated body-part ResNet
BODY_PART_MODEL_CLASSES = [
    "chest",
    "brain",
    "skull",
    "hand",
    "wrist",
    "elbow",
    "shoulder",
    "knee",
    "hip",
    "pelvis",
    "spine",
    "cervical_spine",
    "lumbar_spine",
    "foot",
    "ankle",
    "leg",
    "femur",
    "arm",
    "abdomen",
    "bone",
]


def recommendation_for(body_part_id: str, disease: str) -> str:
    part = BODY_PARTS.get(body_part_id)
    if not part:
        return "Please consult a qualified clinician for further evaluation."
    for d in part.diseases:
        if d.name.lower() == disease.lower():
            return d.recommendation
    return f"Please consult a specialist regarding this {part.display_name.lower()} finding."


def body_part_display(body_part_id: str) -> str:
    part = BODY_PARTS.get(body_part_id)
    return part.display_name if part else body_part_id


def format_body_part_label(body_part_id: str, modality: str | None = None) -> str:
    """e.g. 'Chest X-ray' or 'Brain MRI'."""
    part = BODY_PARTS.get(body_part_id)
    if not part:
        return body_part_id
    mod = modality or part.modality_default
    return f"{part.display_name} {mod}"
