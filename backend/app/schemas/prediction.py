from datetime import datetime

from pydantic import BaseModel, Field


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class DiseaseInfo(BaseModel):
    title: str
    body_region: str
    summary: str
    related_conditions: list[str]
    common_symptoms_to_correlate: list[str]
    typical_xray_findings: list[str]
    possible_causes_if_symptomatic: list[str]
    recommended_next_steps: list[str]
    urgency: str
    disclaimer: str


class PredictionResponse(BaseModel):
    id: int
    # Legacy field (disease / finding code)
    predicted_class: str
    confidence: float
    probabilities: list[ProbabilityItem]
    disease_info: DiseaseInfo | None = None
    image_url: str
    heatmap_url: str | None = None
    report: str | None = None
    model_mode: str
    created_at: datetime
    # Three-stage X-ray pipeline fields
    is_xray: bool = True
    body_part: str | None = None
    disease: str | None = None
    recommendation: str | None = None
    xray_confidence: float | None = None
    body_part_confidence: float | None = None


class HistoryItem(BaseModel):
    id: int
    original_filename: str
    predicted_class: str
    confidence: float
    model_mode: str
    created_at: datetime
    image_url: str
    heatmap_url: str | None = None
    has_report: bool = False
    body_part: str | None = None
    disease: str | None = None


class HistoryListResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class ReportRequest(BaseModel):
    prediction_id: int = Field(..., ge=1)


class ReportResponse(BaseModel):
    id: int
    report: str
    source: str


class HealthResponse(BaseModel):
    status: str
    app: str
    model_loaded: bool
    model_mode: str
    classes: list[str]
    requirements: dict[str, str] | None = None
    pipeline: dict[str, str] | None = None
