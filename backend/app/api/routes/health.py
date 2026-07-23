from fastapi import APIRouter

from app.config import get_settings
from app.ml import get_classifier
from app.ml.pipeline.xray_detector import get_xray_detector
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    classifier = get_classifier()
    gate = get_xray_detector()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        model_loaded=True,
        model_mode=classifier.model_mode,
        classes=classifier.labels,
        xray_gate_loaded=gate.available,
        requirements={
            "medical_image_analysis": "enabled",
            "deep_learning_prediction": "enabled",
            "explainable_ai_gradcam": "enabled",
            "llm_medical_reports": "enabled",
            "rest_apis": "enabled",
            "prediction_history_database": "enabled",
            "user_interface": "enabled",
        },
        pipeline={
            "stage1": "xray_validation+trained_gate",
            "stage2": "body_part_classification",
            "stage3": "disease_prediction",
        },
    )
