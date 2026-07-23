"""Educational disease information for predicted imaging classes."""

from __future__ import annotations

from typing import Any


DISEASE_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "NORMAL": {
        "title": "No pneumonia pattern detected (chest)",
        "body_region": "Chest",
        "summary": (
            "The model did not find dominant radiographic features typically associated "
            "with pneumonia on this chest study."
        ),
        "related_conditions": ["Community-acquired pneumonia", "Viral bronchitis", "Atelectasis"],
        "common_symptoms_to_correlate": [
            "Cough",
            "Fever",
            "Shortness of breath",
            "Chest discomfort",
        ],
        "typical_xray_findings": [
            "Clear lung fields",
            "No focal consolidation highlighted by the model",
            "Normal cardiomediastinal silhouette appearance (model-limited)",
        ],
        "possible_causes_if_symptomatic": [
            "Early infection not yet visible on X-ray",
            "Non-radiographic respiratory illness",
            "Extra-pulmonary causes of symptoms",
        ],
        "recommended_next_steps": [
            "Correlate with clinical exam and vitals",
            "Consider repeat imaging if symptoms progress",
            "Seek clinician review for persistent fever or dyspnea",
        ],
        "urgency": "Routine clinical correlation",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "PNEUMONIA": {
        "title": "Possible pneumonia pattern (chest)",
        "body_region": "Chest",
        "summary": (
            "The deep-learning model detected opacity patterns that can be associated "
            "with pneumonia on chest radiographs."
        ),
        "related_conditions": [
            "Bacterial pneumonia",
            "Viral pneumonia",
            "Aspiration pneumonia",
            "Atypical pneumonia",
        ],
        "common_symptoms_to_correlate": [
            "Fever or chills",
            "Productive or dry cough",
            "Shortness of breath",
            "Pleuritic chest pain",
            "Fatigue",
        ],
        "typical_xray_findings": [
            "Focal or diffuse opacities",
            "Possible consolidation",
            "Airspace disease pattern",
        ],
        "possible_causes_if_symptomatic": [
            "Bacterial infection (e.g., Streptococcus pneumoniae)",
            "Viral infection (e.g., influenza, COVID-19)",
            "Aspiration of oropharyngeal contents",
            "Opportunistic infection in immunocompromised patients",
        ],
        "recommended_next_steps": [
            "Urgent clinician evaluation if oxygen saturation is low or distress is present",
            "Correlate with labs (CBC, CRP) and auscultation",
            "Consider antibiotics only under clinician guidance",
            "Follow-up imaging as clinically indicated",
        ],
        "urgency": "Seek prompt medical review",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "BONE_FRACTURE": {
        "title": "Possible bone fracture pattern (musculoskeletal)",
        "body_region": "Bone / extremity",
        "summary": (
            "The model indicates this image is more consistent with a musculoskeletal "
            "bone X-ray and highlights patterns that can be associated with fracture."
        ),
        "related_conditions": [
            "Acute fracture",
            "Hairline fracture",
            "Displaced fracture",
            "Pathologic fracture",
            "Soft-tissue injury without fracture",
        ],
        "common_symptoms_to_correlate": [
            "Localized pain",
            "Swelling or bruising",
            "Deformity",
            "Limited range of motion",
            "Inability to bear weight (lower limb)",
        ],
        "typical_xray_findings": [
            "Cortical discontinuity",
            "Fracture line or displacement",
            "Possible angulation or joint involvement",
        ],
        "possible_causes_if_symptomatic": [
            "Trauma or fall",
            "Sports injury",
            "Osteoporosis-related fragility fracture",
            "Repetitive stress injury",
        ],
        "recommended_next_steps": [
            "Immobilize the limb and avoid weight-bearing until clinician advice",
            "Orthopedic or emergency evaluation for suspected fracture",
            "Additional views or CT if X-ray is inconclusive",
            "Pain management only as directed by a clinician",
        ],
        "urgency": "Urgent clinical evaluation recommended for suspected fracture",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "BRAIN_NORMAL": {
        "title": "Brain imaging — no tumor pattern highlighted",
        "body_region": "Brain",
        "summary": (
            "The model classified this study as brain imaging and did not highlight "
            "dominant tumor-like patterns."
        ),
        "related_conditions": [
            "Normal brain CT/MRI appearance (model-limited)",
            "Migraine",
            "Early ischemic change not visible yet",
            "Non-tumoral neurological conditions",
        ],
        "common_symptoms_to_correlate": [
            "Headache",
            "Dizziness",
            "Seizure history",
            "Neurological deficit",
            "Trauma history",
        ],
        "typical_xray_findings": [
            "Brain parenchyma without focal mass effect highlighted by the model",
            "No dominant tumor-like lesion pattern detected",
        ],
        "possible_causes_if_symptomatic": [
            "Functional or metabolic causes not visible on this modality",
            "Very small lesions below detection threshold",
            "Vascular events better seen on dedicated stroke protocols",
        ],
        "recommended_next_steps": [
            "Neurology/clinician correlation with symptoms",
            "Consider MRI if clinical concern persists",
            "Urgent care for sudden weakness, speech change, or severe headache",
        ],
        "urgency": "Routine to urgent depending on neurological symptoms",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "BRAIN_TUMOR": {
        "title": "Possible brain tumor pattern",
        "body_region": "Brain",
        "summary": (
            "The deep-learning model detected patterns on brain imaging that can be "
            "associated with intracranial tumor."
        ),
        "related_conditions": [
            "Primary brain tumor",
            "Metastatic brain lesion",
            "Glioma",
            "Meningioma",
            "Abscess or other mass lesion (mimic)",
        ],
        "common_symptoms_to_correlate": [
            "Persistent or progressive headache",
            "Seizures",
            "Nausea/vomiting",
            "Focal weakness or sensory change",
            "Personality or cognitive change",
            "Visual disturbance",
        ],
        "typical_xray_findings": [
            "Focal abnormal density/intensity pattern",
            "Possible mass effect or midline shift cues",
            "Heterogeneous lesion appearance (model-limited)",
        ],
        "possible_causes_if_symptomatic": [
            "Neoplastic growth",
            "Metastasis from systemic cancer",
            "Infectious mass (abscess)",
            "Demarcated cystic or hemorrhagic lesions",
        ],
        "recommended_next_steps": [
            "Urgent clinician/neurosurgery or neurology review",
            "Correlate with contrast MRI and clinical exam",
            "Do not self-interpret severity from AI alone",
            "Seek emergency care for sudden neurological decline",
        ],
        "urgency": "Urgent specialist evaluation recommended",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "ABDOMEN": {
        "title": "Abdominal imaging region detected",
        "body_region": "Abdomen",
        "summary": "The model recognized this as abdominal CT-style imaging (organs such as liver, kidney, spleen, pancreas, bladder).",
        "related_conditions": ["Abdominal pain workup", "Organ-specific pathology", "Trauma evaluation"],
        "common_symptoms_to_correlate": ["Abdominal pain", "Nausea", "Fever", "Jaundice", "Flank pain"],
        "typical_xray_findings": ["Abdominal organ-centered CT slice patterns"],
        "possible_causes_if_symptomatic": ["Infection", "Inflammation", "Mass lesion", "Obstruction"],
        "recommended_next_steps": ["Correlate with clinical exam and labs", "Specialist review as indicated"],
        "urgency": "Clinical correlation required",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "LOWER_LIMB": {
        "title": "Lower-limb imaging region detected",
        "body_region": "Lower limb",
        "summary": "The model recognized lower-limb anatomy (e.g., femur-centered CT/X-ray style imaging).",
        "related_conditions": ["Fracture", "Osteoarthritis", "Soft-tissue injury"],
        "common_symptoms_to_correlate": ["Pain", "Swelling", "Inability to bear weight", "Deformity"],
        "typical_xray_findings": ["Lower extremity bone/soft-tissue patterns"],
        "possible_causes_if_symptomatic": ["Trauma", "Overuse", "Degenerative disease"],
        "recommended_next_steps": ["Orthopedic correlation", "Immobilize if fracture suspected"],
        "urgency": "Urgent if trauma/fracture suspected",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "BREAST_MALIGNANT": {
        "title": "Possible malignant breast imaging pattern",
        "body_region": "Breast",
        "summary": "The model flagged breast imaging patterns that can be associated with malignancy.",
        "related_conditions": ["Breast cancer", "Suspicious breast lesion"],
        "common_symptoms_to_correlate": ["Breast lump", "Skin changes", "Nipple discharge"],
        "typical_xray_findings": ["Suspicious breast lesion patterns (model-limited)"],
        "possible_causes_if_symptomatic": ["Malignant neoplasm", "Complex benign lesion mimicking malignancy"],
        "recommended_next_steps": ["Urgent breast clinic / radiology correlation", "Do not delay specialist review"],
        "urgency": "Urgent specialist evaluation recommended",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "BREAST_NORMAL": {
        "title": "Breast imaging — benign/normal pattern",
        "body_region": "Breast",
        "summary": "The model classified this breast study as normal/benign pattern.",
        "related_conditions": ["Benign breast findings", "Normal screening appearance"],
        "common_symptoms_to_correlate": ["Screening follow-up", "Breast discomfort"],
        "typical_xray_findings": ["No dominant malignant pattern highlighted"],
        "possible_causes_if_symptomatic": ["Benign cysts", "Hormonal changes"],
        "recommended_next_steps": ["Follow local screening guidelines", "Clinician review if symptoms persist"],
        "urgency": "Routine clinical correlation",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "EYE_RETINA": {
        "title": "Retinal / eye imaging detected",
        "body_region": "Eye / Retina",
        "summary": "The model recognized retinal fundus-style imaging.",
        "related_conditions": ["Diabetic retinopathy", "Macular disease", "Hypertensive retinopathy"],
        "common_symptoms_to_correlate": ["Blurred vision", "Floaters", "Vision loss"],
        "typical_xray_findings": ["Retinal fundus patterns (model-limited)"],
        "possible_causes_if_symptomatic": ["Diabetes-related changes", "Vascular retinal disease"],
        "recommended_next_steps": ["Ophthalmology review", "Correlate with diabetes/hypertension history"],
        "urgency": "Prompt eye specialist review if vision changes",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
    "SKIN": {
        "title": "Skin lesion imaging detected",
        "body_region": "Skin",
        "summary": "The model recognized dermatology lesion photography/imaging.",
        "related_conditions": ["Melanoma", "Basal cell carcinoma", "Benign keratosis", "Nevus"],
        "common_symptoms_to_correlate": ["Changing mole", "Itching", "Bleeding lesion", "New skin growth"],
        "typical_xray_findings": ["Dermoscopic/clinical skin lesion patterns"],
        "possible_causes_if_symptomatic": ["Benign lesion", "Premalignant lesion", "Skin cancer"],
        "recommended_next_steps": ["Dermatology evaluation for changing lesions", "ABCDE mole check with clinician"],
        "urgency": "Prompt dermatology review for suspicious lesions",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    },
}


def get_disease_info(predicted_class: str) -> dict[str, Any]:
    key = predicted_class.upper().strip()
    if key in DISEASE_KNOWLEDGE:
        return DISEASE_KNOWLEDGE[key]
    return {
        "title": f"Predicted class: {predicted_class}",
        "body_region": "Unspecified",
        "summary": "No structured disease card is configured for this class.",
        "related_conditions": [],
        "common_symptoms_to_correlate": [],
        "typical_xray_findings": [],
        "possible_causes_if_symptomatic": [],
        "recommended_next_steps": ["Seek clinician review of the imaging study."],
        "urgency": "Clinical correlation required",
        "disclaimer": "Educational decision support only. Not a diagnosis.",
    }
