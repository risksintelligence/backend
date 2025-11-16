#!/usr/bin/env python3
"""
Analytics API endpoints for historical data, z-scores, and component trends.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from app.core.auth import require_observatory_read
from app.services.geri import compute_geri_score
# from app.services.ingestion import get_observations_by_series  # Not needed
from app.db import SessionLocal
from app.models import ObservationModel
from sqlalchemy import desc, and_, text

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/api/v1/analytics/history")
def get_geri_history_data(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    _auth: dict = Depends(require_observatory_read)
) -> Dict[str, Any]:
    """Get historical GERI scores and component data for charting."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        db = SessionLocal()
        try:
            # Get recent observations for all series
            observations = db.query(ObservationModel).filter(
                and_(
                    ObservationModel.observed_at >= start_date,
                    ObservationModel.observed_at <= end_date
                )
            ).order_by(ObservationModel.observed_at.desc()).all()
            
            # Group by series and date
            series_data = {}
            for obs in observations:
                if obs.series_id not in series_data:
                    series_data[obs.series_id] = []
                series_data[obs.series_id].append({
                    "date": obs.observed_at.isoformat(),
                    "value": float(obs.value),
                    "z_score": float(obs.value * 0.2 - 0.1) if obs.value else 0.0,
                    "contribution": float(obs.value * 0.1)  # Mock contribution calculation
                })
            
            # Calculate historical GERI scores (mock for now)
            geri_history = []
            current_date = start_date
            while current_date <= end_date:
                # Mock GERI calculation based on available data
                total_contribution = sum(
                    series_data.get(series_id, [{}])[-1].get("contribution", 0)
                    for series_id in ["VIX", "YIELD_CURVE", "CREDIT_SPREAD"]
                )
                geri_score = max(0, min(100, 50 + total_contribution * 10))
                
                geri_history.append({
                    "date": current_date.isoformat(),
                    "score": round(geri_score, 1),
                    "band": _get_risk_band(geri_score),
                    "color": _get_risk_color(geri_score)
                })
                current_date += timedelta(days=1)
            
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days
                },
                "geri_history": geri_history[-min(30, len(geri_history)):],  # Last 30 points
                "components": {
                    series_id: data[-min(30, len(data)):] 
                    for series_id, data in series_data.items()
                },
                "metadata": {
                    "total_observations": len(observations),
                    "series_count": len(series_data),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")

@router.get("/api/v1/analytics/components/{component_id}/history")
def get_component_history_data(
    component_id: str,
    days: int = Query(30, ge=1, le=365),
    _auth: dict = Depends(require_observatory_read)
) -> Dict[str, Any]:
    """Get detailed history for a specific component."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        db = SessionLocal()
        try:
            observations = db.query(ObservationModel).filter(
                and_(
                    ObservationModel.series_id == component_id,
                    ObservationModel.observed_at >= start_date,
                    ObservationModel.observed_at <= end_date
                )
            ).order_by(ObservationModel.observed_at.desc()).limit(100).all()
            
            if not observations:
                raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
            
            history = []
            for obs in reversed(observations):  # Chronological order
                z_score = float(obs.value * 0.2 - 0.1) if obs.value else 0.0
                history.append({
                    "date": obs.observed_at.isoformat(),
                    "value": float(obs.value),
                    "z_score": z_score,
                    "percentile": min(100, max(0, 50 + z_score * 15)),
                    "contribution": float(obs.value * 0.1),  # Mock contribution
                    "freshness": "fresh" if obs.fetched_at and (datetime.utcnow() - obs.fetched_at).total_seconds() < 3600 else "stale"
                })
            
            # Calculate statistics
            values = [h["value"] for h in history]
            z_scores = [h["z_score"] for h in history]
            
            stats = {
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "current_z_score": z_scores[-1] if z_scores else 0,
                "volatility": sum((z - sum(z_scores)/len(z_scores))**2 for z in z_scores) / len(z_scores) if len(z_scores) > 1 else 0
            }
            
            return {
                "component_id": component_id,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days
                },
                "history": history,
                "statistics": stats,
                "metadata": {
                    "total_points": len(history),
                    "latest_update": observations[0].fetched_at.isoformat() if observations[0].fetched_at else None,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch component history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch component history")

def _get_risk_band(score: float) -> str:
    """Get risk band name for GERI score."""
    if score < 20:
        return "Minimal"
    elif score < 40:
        return "Low" 
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "High"
    else:
        return "Critical"

def _get_risk_color(score: float) -> str:
    """Get semantic color for GERI score."""
    if score < 20:
        return "#00C853"  # Minimal Risk
    elif score < 40:
        return "#64DD17"  # Low Risk
    elif score < 60:
        return "#FFD600"  # Moderate Risk
    elif score < 80:
        return "#FFAB00"  # High Risk
    else:
        return "#D50000"  # Critical Risk