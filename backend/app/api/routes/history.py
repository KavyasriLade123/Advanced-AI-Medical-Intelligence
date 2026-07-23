import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import PredictionRecord
from app.schemas.prediction import (
    DiseaseInfo,
    HistoryItem,
    HistoryListResponse,
    PredictionResponse,
    ProbabilityItem,
    ReportRequest,
    ReportResponse,
)
from app.services.disease_info import get_disease_info
from app.services.report import generate_medical_report

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryListResponse)
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    total = db.scalar(func.count(PredictionRecord.id)) or 0
    rows = (
        db.query(PredictionRecord)
        .order_by(desc(PredictionRecord.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = [
        HistoryItem(
            id=row.id,
            original_filename=row.original_filename,
            predicted_class=row.predicted_class,
            confidence=row.confidence,
            model_mode=row.model_mode,
            created_at=row.created_at,
            image_url=f"/api/media/uploads/{row.filename}",
            heatmap_url=f"/api/media/heatmaps/{row.heatmap_path}" if row.heatmap_path else None,
            has_report=bool(row.report_text),
        )
        for row in rows
    ]
    return HistoryListResponse(total=total, items=items)


@router.get("/history/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)) -> PredictionResponse:
    row = db.get(PredictionRecord, prediction_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    probs = json.loads(row.class_probabilities)
    return PredictionResponse(
        id=row.id,
        predicted_class=row.predicted_class,
        confidence=row.confidence,
        probabilities=[ProbabilityItem(label=k, probability=v) for k, v in probs.items()],
        disease_info=DiseaseInfo(**get_disease_info(row.predicted_class)),
        image_url=f"/api/media/uploads/{row.filename}",
        heatmap_url=f"/api/media/heatmaps/{row.heatmap_path}" if row.heatmap_path else None,
        report=row.report_text,
        model_mode=row.model_mode,
        created_at=row.created_at,
    )


@router.post("/reports", response_model=ReportResponse)
async def regenerate_report(body: ReportRequest, db: Session = Depends(get_db)) -> ReportResponse:
    row = db.get(PredictionRecord, body.prediction_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    probs = json.loads(row.class_probabilities)
    report, source = await generate_medical_report(
        predicted_class=row.predicted_class,
        confidence=row.confidence,
        probabilities=probs,
        filename=row.original_filename,
    )
    row.report_text = report
    db.commit()
    return ReportResponse(id=row.id, report=report, source=source)


@router.delete("/history/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(PredictionRecord, prediction_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    db.delete(row)
    db.commit()
    return {"status": "deleted", "id": prediction_id}
