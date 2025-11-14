"""AI endpoints referencing ml_intelligence_layer specs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.src.ml.inference.service import MLInferenceService

router = APIRouter(prefix="/api/v1/ai", tags=["ai"])
service = MLInferenceService()


@router.get("/regime/current")
async def get_current_regime():
    return await service.predict_current_regime()


@router.get("/geri/forecast")
async def forecast_geri_change(horizon_hours: int = Query(24, enum=[1, 6, 24, 168])):
    return await service.forecast_change(horizon_hours)


@router.get("/anomalies")
async def detect_anomalies(window_hours: int = Query(168, ge=1, le=1000)):
    return await service.detect_anomalies(window_hours)
