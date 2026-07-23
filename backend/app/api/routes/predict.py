import gc
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import HEATMAP_DIR, UPLOAD_DIR, get_settings
from app.database import get_db
from app.ml import explain_prediction, get_classifier, load_image
from app.ml.pipeline import get_pipeline
from app.ml.pipeline.catalog import MSG_NOT_XRAY
from app.models.db_models import PredictionRecord
from app.schemas.prediction import DiseaseInfo, PredictionResponse, ProbabilityItem
from app.services.disease_info import get_disease_info
from app.services.report import generate_medical_report

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "application/octet-stream",  # browsers often send this for .webp/.jpg
}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
    """
    Three-stage X-ray workflow:
      1) Validate medical X-ray
      2) Identify body part
      3) Predict disease + recommendation
    """
    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    ctype = (file.content_type or "").lower().strip()
    type_ok = (not ctype) or ctype in ALLOWED_TYPES or suffix in ALLOWED_SUFFIXES
    if not type_ok:
        raise HTTPException(status_code=400, detail=MSG_NOT_XRAY)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=MSG_NOT_XRAY)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large. Please upload a file under 8 MB.")

    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".png"
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / stored_name
    image_path.write_bytes(raw)

    try:
        image = load_image(raw)
        image.thumbnail((1024, 1024))
        result = get_pipeline().analyze(image)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Pipeline failure")
        _reject(image_path, MSG_NOT_XRAY)

    if not result.ok:
        _reject(image_path, result.error or MSG_NOT_XRAY)

    heatmap_name: str | None = None
    try:
        classifier = get_classifier()
        from app.ml.preprocess import preprocess_image

        tensor = preprocess_image(image)
        if result.source_label in classifier.labels:
            class_idx = classifier.labels.index(result.source_label)
        else:
            known = [(k, v) for k, v in result.probabilities.items() if k in classifier.labels]
            class_idx = (
                classifier.labels.index(max(known, key=lambda kv: kv[1])[0]) if known else 0
            )
        heatmap_name = f"{Path(stored_name).stem}_gradcam.png"
        explain_prediction(classifier, tensor, image, HEATMAP_DIR / heatmap_name, class_idx=class_idx)
    except Exception:
        logger.exception("Grad-CAM skipped")
        heatmap_name = None
    finally:
        gc.collect()

    report_text = None
    if generate_report:
        try:
            report_text, _ = await generate_medical_report(
                predicted_class=result.source_label or result.disease,
                confidence=result.confidence,
                probabilities=result.probabilities,
                filename=file.filename or stored_name,
            )
        except Exception:
            report_text = None

    # Prefer structured recommendation in report header
    if result.recommendation:
        header = (
            f"Body Part: {result.body_part}\n"
            f"Prediction: {result.disease}\n"
            f"Confidence: {result.confidence * 100:.1f}%\n"
            f"Recommendation: {result.recommendation}\n"
        )
        report_text = f"{header}\n{report_text}" if report_text else header

    stored_class = result.source_label or result.disease
    record = PredictionRecord(
        filename=stored_name,
        original_filename=file.filename or stored_name,
        predicted_class=stored_class,
        confidence=result.confidence,
        class_probabilities=json.dumps(result.probabilities),
        heatmap_path=heatmap_name,
        report_text=report_text,
        model_mode=result.model_mode,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    disease_card = None
    try:
        disease_card = DiseaseInfo(**get_disease_info(stored_class))
    except Exception:
        disease_card = None

    return PredictionResponse(
        id=record.id,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        probabilities=[ProbabilityItem(label=k, probability=v) for k, v in result.probabilities.items()],
        disease_info=disease_card,
        image_url=f"/api/media/uploads/{stored_name}",
        heatmap_url=f"/api/media/heatmaps/{heatmap_name}" if heatmap_name else None,
        report=report_text,
        model_mode=record.model_mode,
        created_at=record.created_at,
        is_xray=True,
        body_part=result.body_part,
        disease=result.disease,
        recommendation=result.recommendation,
        xray_confidence=result.xray_confidence,
        body_part_confidence=result.body_part_confidence,
    )
