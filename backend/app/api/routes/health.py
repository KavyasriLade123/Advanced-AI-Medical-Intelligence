from fastapi import APIRouter

from app.config import get_settings
from app.ml import get_classifier
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    classifier = get_classifier()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        model_loaded=True,
        model_mode=classifier.model_mode,
        classes=classifier.labels,
        requirements={
            "medical_image_analysis": "enabled",
            "deep_learning_prediction": "enabled",
            "explainable_ai_gradcam": "enabled",
            "llm_medical_reports": "enabled",
            "rest_apis": "enabled",
            "prediction_history_database": "enabled",
            "user_interface": "enabled",
        },
    )
