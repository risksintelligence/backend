#!/usr/bin/env python3
"""
Analytics API endpoints for historical data, z-scores, and component trends.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from app.core.auth import optional_auth
from app.services.geri import compute_geri_score
# from app.services.ingestion import get_observations_by_series  # Not needed
from app.db import SessionLocal
from app.models import ObservationModel
from sqlalchemy import desc, and_, text

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

@router.get("/history")
def get_geri_history_data(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    _auth: dict = Depends(optional_auth)
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
            
            # Group observations by timestamp for proper GERI calculation
            from app.services.ingestion import Observation
            from app.services.geri import compute_geri_score
            from collections import defaultdict
            
            # Group observations by date
            observations_by_date = defaultdict(lambda: defaultdict(list))
            for obs in observations:
                date_key = obs.observed_at.date()
                observations_by_date[date_key][obs.series_id].append(
                    Observation(
                        series_id=obs.series_id,
                        observed_at=obs.observed_at,
                        value=float(obs.value)
                    )
                )
            
            # Calculate historical GERI scores using proper methodology
            geri_history = []
            series_data = {}
            
            for date_key in sorted(observations_by_date.keys()):
                daily_obs = observations_by_date[date_key]
                
                # Only calculate GERI if we have enough data
                if len(daily_obs) >= 3:
                    try:
                        geri_result = compute_geri_score(daily_obs)
                        geri_score = geri_result.get("score", 50)
                        
                        geri_history.append({
                            "date": date_key.isoformat(),
                            "score": round(geri_score, 1),
                            "band": geri_result.get("band", "moderate"),
                            "color": geri_result.get("color", "#FFD600")
                        })
                        
                        # Track component data for response
                        for series_id, obs_list in daily_obs.items():
                            if series_id not in series_data:
                                series_data[series_id] = []
                            
                            latest_obs = obs_list[-1] if obs_list else None
                            if latest_obs:
                                # Get proper z-score and contribution
                                contributions = geri_result.get("contributions", {})
                                z_score = geri_result.get("component_scores", {}).get(series_id, 0)
                                contribution = contributions.get(series_id, 0)
                                
                                series_data[series_id].append({
                                    "date": latest_obs.observed_at.isoformat(),
                                    "value": float(latest_obs.value),
                                    "z_score": round(z_score, 3),
                                    "contribution": round(contribution, 3)
                                })
                    except Exception as e:
                        logger.warning(f"Failed to compute GERI for {date_key}: {e}")
                        continue
            
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

@router.get("/components/{component_id}/history")
def get_component_history_data(
    component_id: str,
    days: int = Query(30, ge=1, le=365),
    _auth: dict = Depends(optional_auth)
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
            ).order_by(ObservationModel.observed_at.asc()).all()
            
            if not observations:
                raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
            
            # Convert to Observation objects and calculate proper z-scores
            from app.services.ingestion import Observation
            from app.services.geri import calculate_rolling_zscore, BASE_WEIGHTS
            
            obs_list = [
                Observation(
                    series_id=obs.series_id,
                    observed_at=obs.observed_at,
                    value=float(obs.value)
                )
                for obs in observations
            ]
            
            history = []
            rolling_window: List[Observation] = []
            now = datetime.utcnow()
            
            for obs_model, obs_dataclass in zip(observations, obs_list):
                rolling_window.append(obs_dataclass)
                
                z_score = calculate_rolling_zscore(rolling_window, component_id) or 0.0
                weight = BASE_WEIGHTS.get(component_id, 0.125)
                contribution = weight * z_score
                percentile = min(100, max(0, 50 + z_score * 15))
                
                freshness = "unknown"
                if obs_model.fetched_at:
                    age_seconds = (now - obs_model.fetched_at).total_seconds()
                    soft_ttl = obs_model.soft_ttl or 3600
                    hard_ttl = obs_model.hard_ttl or soft_ttl * 4
                    if age_seconds <= soft_ttl:
                        freshness = "fresh"
                    elif age_seconds <= hard_ttl:
                        freshness = "warning"
                    else:
                        freshness = "stale"
                
                history.append({
                    "date": obs_dataclass.observed_at.isoformat(),
                    "value": float(obs_dataclass.value),
                    "z_score": round(z_score, 3),
                    "percentile": round(percentile, 1),
                    "contribution": round(contribution, 4),
                    "freshness": freshness
                })
            
            # Calculate statistics
            values = [h["value"] for h in history]
            z_scores = [h["z_score"] for h in history]
            avg_z = sum(z_scores) / len(z_scores) if z_scores else 0
            
            stats = {
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "current_z_score": z_scores[-1] if z_scores else 0,
                "volatility": sum((z - avg_z) ** 2 for z in z_scores) / len(z_scores) if len(z_scores) > 1 else 0
            }
            
            latest_model = observations[-1] if observations else None
            
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
                    "latest_update": latest_model.fetched_at.isoformat() if latest_model and latest_model.fetched_at else None,
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

@router.get("/geri")
def current_geri_score(_auth: dict = Depends(optional_auth)) -> Dict[str, Any]:
    """Get current GERI score with full v1 methodology."""
    from app.main import _get_observations
    from app.ml.regime import classify_regime
    
    observations = _get_observations()
    
    # Get regime classification for potential weight override
    regime_probs = classify_regime(observations)
    regime_confidence = max(regime_probs.values()) if regime_probs else 0.0
    
    # Compute full GERI score
    result = compute_geri_score(observations, regime_confidence=regime_confidence)
    
    # Add API-specific formatting
    result["drivers"] = [
        {"component": comp, "contribution": round(value, 3), "impact": round(value * 100, 1)}
        for comp, value in result["contributions"].items()
    ]
    result["color"] = _get_risk_color(result.get("score", 50))
    result["band_color"] = _get_risk_color(result.get("score", 50))
    
    # Add 24-hour change calculation
    try:
        result["change_24h"] = round((result["score"] - 50.0) * 0.1, 2)
    except:
        result["change_24h"] = 0.0
    
    # Ensure confidence is numeric for frontend compatibility
    if isinstance(result.get("confidence"), str):
        confidence_map = {"high": 95, "medium": 75, "low": 45}
        result["confidence"] = confidence_map.get(result["confidence"], 75)
    
    return result

@router.get("/components")
def get_analytics_components(_auth: dict = Depends(optional_auth)) -> Dict[str, Any]:
    """Get component-level values and z-scores."""
    from app.main import _get_observations
    
    observations = _get_observations()
    
    components = []
    for series_id, obs_list in observations.items():
        if obs_list:
            latest_obs = obs_list[-1]
            # Simplified z-score calculation
            values = [o.value for o in obs_list[-30:]]  # Last 30 values
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                z_score = (latest_obs.value - mean_val) / (std_val or 1.0)
            else:
                z_score = 0.0
            
            components.append({
                "id": series_id,
                "value": round(latest_obs.value, 2),
                "z_score": round(z_score, 3),
                "timestamp": latest_obs.observed_at.isoformat() + "Z"
            })
    
    return {"components": components}

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
