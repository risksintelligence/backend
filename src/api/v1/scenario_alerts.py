"""Alert threshold endpoints for Scenario Studio."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.services.alert_threshold_service import get_alert_threshold_service, AlertThresholdService
from src.services.auth_service import User
from src.api.middleware.auth import require_auth, require_premium

router = APIRouter(prefix="/api/v1/scenario", tags=["scenario-alerts"])

class CreateAlertThresholdRequest(BaseModel):
    name: str
    geri_threshold: Optional[float] = None
    delta_threshold: Optional[float] = None
    conditions: Dict[str, Any] = {}
    notification_channels: List[str] = []

class UpdateAlertThresholdRequest(BaseModel):
    name: Optional[str] = None
    geri_threshold: Optional[float] = None
    delta_threshold: Optional[float] = None
    conditions: Optional[Dict[str, Any]] = None
    notification_channels: Optional[List[str]] = None
    is_active: Optional[bool] = None

class AlertThresholdResponse(BaseModel):
    id: int
    name: str
    geri_threshold: Optional[float]
    delta_threshold: Optional[float]
    conditions: Dict[str, Any]
    notification_channels: List[str]
    is_active: bool
    created_at: str
    updated_at: str

class AlertEventResponse(BaseModel):
    id: int
    threshold_id: int
    trigger_type: str
    trigger_value: float
    scenario_data: Dict[str, Any]
    created_at: str

@router.get("/alert-thresholds")
async def list_alert_thresholds(
    user: User = Depends(require_auth),
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
) -> List[AlertThresholdResponse]:
    """Get user's alert thresholds."""
    thresholds = await threshold_service.get_user_thresholds(user.id)
    
    return [
        AlertThresholdResponse(
            id=threshold.id,
            name=threshold.name,
            geri_threshold=threshold.geri_threshold,
            delta_threshold=threshold.delta_threshold,
            conditions=threshold.conditions,
            notification_channels=threshold.notification_channels,
            is_active=threshold.is_active,
            created_at=threshold.created_at.isoformat(),
            updated_at=threshold.updated_at.isoformat()
        )
        for threshold in thresholds
    ]

@router.post("/alert-thresholds")
async def create_alert_threshold(
    request: CreateAlertThresholdRequest,
    user: User = Depends(require_premium),  # Premium feature
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
) -> AlertThresholdResponse:
    """Create a new alert threshold (premium feature)."""
    try:
        threshold = await threshold_service.create_alert_threshold(
            user_id=user.id,
            name=request.name,
            geri_threshold=request.geri_threshold,
            delta_threshold=request.delta_threshold,
            conditions=request.conditions,
            notification_channels=request.notification_channels
        )
        
        return AlertThresholdResponse(
            id=threshold.id,
            name=threshold.name,
            geri_threshold=threshold.geri_threshold,
            delta_threshold=threshold.delta_threshold,
            conditions=threshold.conditions,
            notification_channels=threshold.notification_channels,
            is_active=threshold.is_active,
            created_at=threshold.created_at.isoformat(),
            updated_at=threshold.updated_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/alert-thresholds/{threshold_id}")
async def update_alert_threshold(
    threshold_id: int,
    request: UpdateAlertThresholdRequest,
    user: User = Depends(require_auth),
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
) -> AlertThresholdResponse:
    """Update an alert threshold."""
    try:
        # Only update fields that are provided
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        
        threshold = await threshold_service.update_alert_threshold(
            threshold_id=threshold_id,
            user_id=user.id,
            **update_data
        )
        
        if not threshold:
            raise HTTPException(status_code=404, detail="Alert threshold not found")
        
        return AlertThresholdResponse(
            id=threshold.id,
            name=threshold.name,
            geri_threshold=threshold.geri_threshold,
            delta_threshold=threshold.delta_threshold,
            conditions=threshold.conditions,
            notification_channels=threshold.notification_channels,
            is_active=threshold.is_active,
            created_at=threshold.created_at.isoformat(),
            updated_at=threshold.updated_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/alert-thresholds/{threshold_id}")
async def delete_alert_threshold(
    threshold_id: int,
    user: User = Depends(require_auth),
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
):
    """Delete an alert threshold."""
    success = await threshold_service.delete_alert_threshold(threshold_id, user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert threshold not found")
    
    return {"success": True, "message": "Alert threshold deleted"}

@router.get("/alert-events")
async def list_alert_events(
    limit: int = 50,
    user: User = Depends(require_auth),
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
) -> List[AlertEventResponse]:
    """Get user's recent alert events."""
    events = await threshold_service.get_user_alert_events(user.id, limit)
    
    return [
        AlertEventResponse(
            id=event.id,
            threshold_id=event.threshold_id,
            trigger_type=event.trigger_type,
            trigger_value=event.trigger_value,
            scenario_data=event.scenario_data,
            created_at=event.created_at.isoformat()
        )
        for event in events
    ]

@router.post("/simulate-with-alerts")
async def simulate_scenario_with_alerts(
    request: dict,
    user: User = Depends(require_auth),
    threshold_service: AlertThresholdService = Depends(get_alert_threshold_service)
):
    """Run scenario simulation and check against alert thresholds."""
    # Import scenario service here to avoid circular import
    from src.services.scenario_service import get_scenario_service
    scenario_service = get_scenario_service()
    
    shocks: List[dict] = request.get("shocks", [])
    horizon = int(request.get("horizon_hours", 24))
    
    # Run the scenario simulation
    result = await scenario_service.simulate(shocks, horizon)
    
    # Prepare scenario data for alert processing
    scenario_data = {
        "baseline": result.baseline,
        "scenario": result.scenario,
        "delta": result.delta,
        "shocks": shocks,
        "horizon_hours": horizon,
        "band": result.band,
        "explanation": result.explanation
    }
    
    # Process alerts for this user
    triggered_alerts = await threshold_service.process_scenario_result(
        user_id=user.id,
        scenario_data=scenario_data
    )
    
    return {
        "baseline": result.baseline,
        "scenario": result.scenario,
        "band": result.band,
        "delta": result.delta,
        "explanation": result.explanation,
        "alerts": {
            "triggered_count": len(triggered_alerts),
            "triggered_alerts": [
                {
                    "id": alert.id,
                    "trigger_type": alert.trigger_type,
                    "trigger_value": alert.trigger_value
                }
                for alert in triggered_alerts
            ]
        }
    }