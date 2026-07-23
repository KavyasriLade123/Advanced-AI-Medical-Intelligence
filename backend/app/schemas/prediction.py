from datetime import datetime
from typing import Any

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
    predicted_class: str
    confidence: float
    probabilities: list[ProbabilityItem]
    disease_info: DiseaseInfo | None = None
    image_url: str
    heatmap_url: str | None = None
    report: str | None = None
    model_mode: str
    created_at: datetime


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
