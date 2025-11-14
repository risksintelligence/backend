"""API endpoints for alert subscription and delivery management."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.scenario_service import get_alert_service, AlertService
from src.services.alert_triggers import AlertType, AlertSeverity
from src.jobs.alert_monitoring import run_alert_monitoring

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class AlertSubscriptionRequest(BaseModel):
    channel: str  # "email" or "webhook"
    address: str  # email address or webhook URL
    conditions: List[Dict[str, Any]]
    description: Optional[str] = None


class AlertConditionRequest(BaseModel):
    alert_type: str  # AlertType enum value
    severity: str    # AlertSeverity enum value
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte", "eq", "ne"
    cooldown_minutes: int = 60


class AlertTestRequest(BaseModel):
    subscription_id: Optional[int] = None
    test_message: str = "Test alert from RiskSX Intelligence System"


@router.post("/subscribe")
async def create_alert_subscription(
    request: AlertSubscriptionRequest,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Create a new alert subscription for email or webhook delivery."""
    
    # Validate channel
    if request.channel not in ["email", "webhook"]:
        raise HTTPException(status_code=400, detail="Channel must be 'email' or 'webhook'")
    
    # Validate email address format if email channel
    if request.channel == "email":
        try:
            # Basic email validation
            if "@" not in request.address or "." not in request.address.split("@")[1]:
                raise ValueError("Invalid email format")
        except:
            raise HTTPException(status_code=400, detail="Invalid email address format")
    
    # Validate webhook URL if webhook channel
    if request.channel == "webhook":
        if not request.address.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Webhook address must be a valid HTTP/HTTPS URL")
    
    # Create subscription
    try:
        subscription = alert_service.subscribe(
            channel=request.channel,
            address=request.address,
            conditions=request.conditions
        )
        
        return {
            "success": True,
            "subscription": subscription,
            "message": f"Alert subscription created for {request.channel}: {request.address}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")


@router.get("/subscriptions")
async def list_alert_subscriptions(
    alert_service: AlertService = Depends(get_alert_service)
):
    """List all active alert subscriptions."""
    try:
        subscriptions = alert_service.list_subscriptions()
        
        return {
            "success": True,
            "subscriptions": subscriptions,
            "count": len(subscriptions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve subscriptions: {str(e)}")


@router.get("/deliveries")
async def get_alert_deliveries(
    limit: int = 50,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get recent alert delivery history."""
    try:
        deliveries = alert_service.deliveries()
        
        # Limit results
        limited_deliveries = deliveries[:limit]
        
        return {
            "success": True,
            "deliveries": limited_deliveries,
            "count": len(limited_deliveries),
            "total_available": len(deliveries)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve deliveries: {str(e)}")


@router.post("/test")
async def send_test_alert(
    request: AlertTestRequest,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Send a test alert to verify delivery configuration."""
    try:
        # Create test alert payload
        test_payload = {
            "alert_type": "test",
            "severity": "low",
            "title": "Test Alert - RiskSX Intelligence System",
            "message": request.test_message,
            "value": 0.0,
            "threshold": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "context": {
                "test": True,
                "sent_at": datetime.utcnow().isoformat()
            },
            "dashboard_url": "https://frontend-1-wvu7.onrender.com/admin"
        }
        
        # Send test alert
        if request.subscription_id:
            # Send to specific subscription (would need additional implementation)
            raise HTTPException(status_code=501, detail="Specific subscription testing not yet implemented")
        else:
            # Send to all subscriptions
            deliveries = await alert_service.deliver_alerts(test_payload)
        
        return {
            "success": True,
            "test_alert_sent": True,
            "deliveries": len(deliveries),
            "message": f"Test alert sent to {len(deliveries)} subscription(s)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test alert: {str(e)}")


@router.post("/monitor/run")
async def trigger_alert_monitoring(
    background_tasks: BackgroundTasks
):
    """Manually trigger an alert monitoring cycle."""
    try:
        # Run monitoring in background
        background_tasks.add_task(run_alert_monitoring)
        
        return {
            "success": True,
            "message": "Alert monitoring cycle triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger monitoring: {str(e)}")


@router.get("/types")
async def get_alert_types():
    """Get available alert types and severities for configuration."""
    return {
        "success": True,
        "alert_types": [
            {
                "name": alert_type.value,
                "description": alert_type.value.replace("_", " ").title()
            }
            for alert_type in AlertType
        ],
        "severities": [
            {
                "name": severity.value,
                "description": severity.value.title()
            }
            for severity in AlertSeverity
        ],
        "comparison_operators": [
            {"name": "gt", "description": "Greater than"},
            {"name": "gte", "description": "Greater than or equal"},
            {"name": "lt", "description": "Less than"},
            {"name": "lte", "description": "Less than or equal"},
            {"name": "eq", "description": "Equal to"},
            {"name": "ne", "description": "Not equal to"}
        ]
    }


@router.get("/conditions/defaults")
async def get_default_alert_conditions():
    """Get default alert conditions for common scenarios."""
    return {
        "success": True,
        "default_conditions": {
            "geri_critical": {
                "alert_type": "geri_threshold",
                "severity": "critical",
                "threshold_value": 10.0,
                "comparison": "lt",
                "description": "Alert when GERI falls below critical level"
            },
            "geri_warning": {
                "alert_type": "geri_threshold",
                "severity": "high",
                "threshold_value": 20.0,
                "comparison": "lt",
                "description": "Alert when GERI falls below warning level"
            },
            "geri_major_change": {
                "alert_type": "geri_change",
                "severity": "high",
                "threshold_value": 15.0,
                "comparison": "gt",
                "description": "Alert on major GERI changes"
            },
            "data_very_stale": {
                "alert_type": "data_stale",
                "severity": "high",
                "threshold_value": 24.0,
                "comparison": "gt",
                "description": "Alert when data is over 24 hours old"
            },
            "ml_anomaly": {
                "alert_type": "ml_anomaly",
                "severity": "medium",
                "threshold_value": -0.2,
                "comparison": "lt",
                "description": "Alert on ML-detected anomalies"
            }
        }
    }


@router.get("/status")
async def get_alert_system_status(
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get alert system status and configuration."""
    try:
        subscriptions = alert_service.list_subscriptions()
        recent_deliveries = alert_service.deliveries()[:10]
        
        # Count by channel
        channel_counts = {}
        for sub in subscriptions:
            channel = sub["channel"]
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        return {
            "success": True,
            "status": "operational",
            "configuration": {
                "total_subscriptions": len(subscriptions),
                "subscriptions_by_channel": channel_counts,
                "recent_deliveries": len(recent_deliveries)
            },
            "last_check": datetime.utcnow().isoformat(),
            "endpoints": {
                "subscribe": "/api/v1/alerts/subscribe",
                "test": "/api/v1/alerts/test",
                "monitor": "/api/v1/alerts/monitor/run"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }
