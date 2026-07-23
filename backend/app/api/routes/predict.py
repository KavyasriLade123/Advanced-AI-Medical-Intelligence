import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import HEATMAP_DIR, UPLOAD_DIR, get_settings
from app.database import get_db
from app.ml import explain_prediction, get_classifier, load_image, preprocess_image
from app.models.db_models import PredictionRecord
from app.schemas.prediction import DiseaseInfo, PredictionResponse, ProbabilityItem
from app.services.disease_info import get_disease_info
from app.services.report import generate_medical_report

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}


def _reject(image_path: Path, message: str) -> None:
    if image_path.exists():
        image_path.unlink(missing_ok=True)
    raise HTTPException(status_code=400, detail=message)


@router.post("", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    generate_report: bool = True,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    settings = get_settings()
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a correct medical image (JPG/PNG).",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded. Please upload a correct medical image.")

    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / stored_name
    image_path.write_bytes(raw)

    try:
        image = load_image(raw)
        tensor = preprocess_image(image)
        classifier = get_classifier()
        predicted, confidence, probs = classifier.predict(tensor)
    except Exception as exc:
        _reject(image_path, f"Could not read image. Please upload a correct medical image. ({exc})")

    # Reject unrelated / out-of-scope predictions
    if predicted.upper() == "UNSUPPORTED" or confidence < settings.min_confidence:
        _reject(
            image_path,
            "This image is not related to the trained medical body-part models. "
            "Please upload a correct medical image "
            "(Brain, Eye/Retina, Breast, Chest X-ray, Abdomen, Skin, Bone fracture, or Lower limb).",
        )

    try:
        class_idx = classifier.labels.index(predicted)
        heatmap_name = f"{Path(stored_name).stem}_gradcam.png"
        heatmap_path = HEATMAP_DIR / heatmap_name
        explain_prediction(classifier, tensor, image, heatmap_path, class_idx=class_idx)
    except Exception as exc:
        _reject(image_path, f"Prediction failed while explaining the image: {exc}")

    report_text = None
    if generate_report:
        report_text, _ = await generate_medical_report(
            predicted_class=predicted,
            confidence=confidence,
            probabilities=probs,
            filename=file.filename or stored_name,
        )

    record = PredictionRecord(
        filename=stored_name,
        original_filename=file.filename or stored_name,
        predicted_class=predicted,
        confidence=confidence,
        class_probabilities=json.dumps(probs),
        heatmap_path=heatmap_name,
        report_text=report_text,
        model_mode=classifier.model_mode,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        id=record.id,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        probabilities=[ProbabilityItem(label=k, probability=v) for k, v in probs.items()],
        disease_info=DiseaseInfo(**get_disease_info(record.predicted_class)),
        image_url=f"/api/media/uploads/{stored_name}",
        heatmap_url=f"/api/media/heatmaps/{heatmap_name}",
        report=report_text,
        model_mode=record.model_mode,
        created_at=record.created_at,
    )
