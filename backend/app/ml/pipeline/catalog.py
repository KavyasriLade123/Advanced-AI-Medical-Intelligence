"""Body-part and disease configuration for the multi-stage X-ray pipeline.

Add new body parts / diseases here without changing pipeline code.
Optional per-part weights: backend/models/disease/{body_part}.pth
"""

from __future__ import annotations

from dataclasses import dataclass


MSG_NOT_XRAY = "Please upload a valid medical X-ray image."
MSG_UNKNOWN_PART = "Unable to identify the X-ray type. Please upload a clearer medical X-ray."


@dataclass(frozen=True)
class DiseaseSpec:
    name: str
    recommendation: str


@dataclass(frozen=True)
class BodyPartSpec:
    id: str
    display_name: str
    diseases: tuple[DiseaseSpec, ...]
    # Existing unified-model labels that map into this body part
    source_labels: tuple[str, ...] = ()
    weights_file: str | None = None  # optional dedicated disease model


# Canonical supported body parts (expandable).
BODY_PARTS: dict[str, BodyPartSpec] = {
    "chest": BodyPartSpec(
        id="chest",
        display_name="Chest",
        source_labels=("NORMAL", "PNEUMONIA", "BREAST_NORMAL", "BREAST_MALIGNANT"),
        diseases=(
            DiseaseSpec("Normal", "No acute chest finding suggested. Routine follow-up as advised by your clinician."),
            DiseaseSpec("Pneumonia", "Please consult a pulmonologist."),
            DiseaseSpec("Tuberculosis", "Please consult a pulmonologist for further evaluation."),
            DiseaseSpec("COVID", "Please consult a pulmonologist / infectious disease specialist."),
            DiseaseSpec("Breast abnormality", "Please consult a radiologist or breast specialist."),
        ),
        weights_file="disease/chest.pth",
    ),
    "lung": BodyPartSpec(
        id="lung",
        display_name="Lung",
        source_labels=(),
        diseases=(
            DiseaseSpec("Normal", "No acute lung finding suggested. Follow clinician advice."),
            DiseaseSpec("Pneumonia", "Please consult a pulmonologist."),
            DiseaseSpec("COVID", "Please consult a pulmonologist / infectious disease specialist."),
        ),
        weights_file="disease/lung.pth",
    ),
    "skull": BodyPartSpec(
        id="skull",
        display_name="Skull",
        source_labels=("BRAIN_NORMAL", "BRAIN_TUMOR"),
        diseases=(
            DiseaseSpec("Normal", "No acute intracranial finding suggested. Follow neurologist advice if symptomatic."),
            DiseaseSpec("Brain tumor / mass", "Please consult a neurologist or neurosurgeon."),
        ),
        weights_file="disease/skull.pth",
    ),
    "hand": BodyPartSpec(
        id="hand",
        display_name="Hand",
        source_labels=(),
        diseases=(
            DiseaseSpec("Normal", "No acute hand finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
            DiseaseSpec("Arthritis", "Please consult an orthopedist or rheumatologist."),
        ),
        weights_file="disease/hand.pth",
    ),
    "wrist": BodyPartSpec(
        id="wrist",
        display_name="Wrist",
        diseases=(
            DiseaseSpec("Normal", "No acute wrist finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/wrist.pth",
    ),
    "elbow": BodyPartSpec(
        id="elbow",
        display_name="Elbow",
        diseases=(
            DiseaseSpec("Normal", "No acute elbow finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/elbow.pth",
    ),
    "shoulder": BodyPartSpec(
        id="shoulder",
        display_name="Shoulder",
        diseases=(
            DiseaseSpec("Normal", "No acute shoulder finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/shoulder.pth",
    ),
    "knee": BodyPartSpec(
        id="knee",
        display_name="Knee",
        diseases=(
            DiseaseSpec("Normal", "No acute knee finding suggested."),
            DiseaseSpec("Osteoarthritis", "Please consult an orthopedic specialist."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/knee.pth",
    ),
    "hip": BodyPartSpec(
        id="hip",
        display_name="Hip",
        diseases=(
            DiseaseSpec("Normal", "No acute hip finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/hip.pth",
    ),
    "pelvis": BodyPartSpec(
        id="pelvis",
        display_name="Pelvis",
        diseases=(
            DiseaseSpec("Normal", "No acute pelvic finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/pelvis.pth",
    ),
    "spine": BodyPartSpec(
        id="spine",
        display_name="Spine",
        diseases=(
            DiseaseSpec("Normal", "No acute spinal finding suggested."),
            DiseaseSpec("Scoliosis", "Please consult an orthopedic spine specialist."),
            DiseaseSpec("Disc degeneration", "Please consult an orthopedic spine specialist."),
        ),
        weights_file="disease/spine.pth",
    ),
    "cervical_spine": BodyPartSpec(
        id="cervical_spine",
        display_name="Cervical Spine",
        diseases=(
            DiseaseSpec("Normal", "No acute cervical finding suggested."),
            DiseaseSpec("Degenerative change", "Please consult a spine specialist."),
        ),
        weights_file="disease/cervical_spine.pth",
    ),
    "lumbar_spine": BodyPartSpec(
        id="lumbar_spine",
        display_name="Lumbar Spine",
        diseases=(
            DiseaseSpec("Normal", "No acute lumbar finding suggested."),
            DiseaseSpec("Degenerative change", "Please consult a spine specialist."),
        ),
        weights_file="disease/lumbar_spine.pth",
    ),
    "foot": BodyPartSpec(
        id="foot",
        display_name="Foot",
        diseases=(
            DiseaseSpec("Normal", "No acute foot finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/foot.pth",
    ),
    "ankle": BodyPartSpec(
        id="ankle",
        display_name="Ankle",
        diseases=(
            DiseaseSpec("Normal", "No acute ankle finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/ankle.pth",
    ),
    "leg": BodyPartSpec(
        id="leg",
        display_name="Leg",
        source_labels=("LOWER_LIMB",),
        diseases=(
            DiseaseSpec("Normal", "No acute lower-limb finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/leg.pth",
    ),
    "femur": BodyPartSpec(
        id="femur",
        display_name="Femur",
        diseases=(
            DiseaseSpec("Normal", "No acute femur finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/femur.pth",
    ),
    "arm": BodyPartSpec(
        id="arm",
        display_name="Arm",
        diseases=(
            DiseaseSpec("Normal", "No acute arm finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/arm.pth",
    ),
    "abdomen": BodyPartSpec(
        id="abdomen",
        display_name="Abdomen",
        source_labels=("ABDOMEN",),
        diseases=(
            DiseaseSpec("Normal", "No acute abdominal radiographic finding suggested."),
            DiseaseSpec("Abnormality", "Please consult a radiologist or gastroenterologist."),
        ),
        weights_file="disease/abdomen.pth",
    ),
    "dental": BodyPartSpec(
        id="dental",
        display_name="Dental",
        diseases=(
            DiseaseSpec("Normal", "No acute dental radiographic finding suggested."),
            DiseaseSpec("Dental pathology", "Please consult a dentist / oral radiologist."),
        ),
        weights_file="disease/dental.pth",
    ),
    "bone": BodyPartSpec(
        id="bone",
        display_name="Bone",
        source_labels=("BONE_FRACTURE",),
        diseases=(
            DiseaseSpec("Normal", "No acute osseous finding suggested."),
            DiseaseSpec("Fracture", "Please consult an orthopedic specialist."),
        ),
        weights_file="disease/bone.pth",
    ),
}


# Map unified classifier label → (body_part_id, disease_name)
LABEL_TO_BODY_DISEASE: dict[str, tuple[str, str]] = {
    "NORMAL": ("chest", "Normal"),
    "PNEUMONIA": ("chest", "Pneumonia"),
    "BREAST_NORMAL": ("chest", "Normal"),
    "BREAST_MALIGNANT": ("chest", "Breast abnormality"),
    "BRAIN_NORMAL": ("skull", "Normal"),
    "BRAIN_TUMOR": ("skull", "Brain tumor / mass"),
    "BONE_FRACTURE": ("bone", "Fracture"),
    "LOWER_LIMB": ("leg", "Fracture"),
    "ABDOMEN": ("abdomen", "Abnormality"),
}

# Non-X-ray classes from the unified model (reject at Stage 1)
NON_XRAY_LABELS = {"UNSUPPORTED", "SKIN", "EYE_RETINA"}


def recommendation_for(body_part_id: str, disease: str) -> str:
    part = BODY_PARTS.get(body_part_id)
    if not part:
        return "Please consult a qualified clinician for further evaluation."
    for d in part.diseases:
        if d.name.lower() == disease.lower():
            return d.recommendation
    return f"Please consult a specialist regarding this {part.display_name.lower()} X-ray finding."


def body_part_display(body_part_id: str) -> str:
    part = BODY_PARTS.get(body_part_id)
    return part.display_name if part else body_part_id
