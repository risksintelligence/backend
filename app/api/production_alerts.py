"""
Production Alerting API Endpoints

Provides endpoints for accessing production health alerts, alert history,
and triggering manual alert checks for system administrators.
"""

from datetime import datetime
from typing import Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import require_system_rate_limit
from app.core.production_alerting import production_alerting, AlertSeverity

router = APIRouter(prefix="/api/v1/alerts", tags=["production-alerts"])

@router.get("/active")
async def get_active_alerts(
    severity: AlertSeverity = Query(None, description="Filter by alert severity"),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get all currently active production alerts."""
    try:
        active_alerts = production_alerting.get_active_alerts()
        
        # Filter by severity if provided
        if severity:
            active_alerts = [
                alert for alert in active_alerts 
                if alert["severity"] == severity.value
            ]
        
        # Calculate summary statistics
        total_alerts = len(active_alerts)
        critical_alerts = len([a for a in active_alerts if a["severity"] == "critical"])
        high_alerts = len([a for a in active_alerts if a["severity"] == "high"])
        medium_alerts = len([a for a in active_alerts if a["severity"] == "medium"])
        low_alerts = len([a for a in active_alerts if a["severity"] == "low"])
        
        return {
            "active_alerts": active_alerts,
            "summary": {
                "total_alerts": total_alerts,
                "critical_alerts": critical_alerts,
                "high_alerts": high_alerts,
                "medium_alerts": medium_alerts,
                "low_alerts": low_alerts,
                "has_critical": critical_alerts > 0,
                "overall_alert_level": _determine_overall_alert_level(active_alerts)
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")

@router.get("/history")
async def get_alert_history(
    limit: int = Query(50, ge=1, le=200, description="Number of recent alerts to return"),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get recent alert history including resolved alerts."""
    try:
        alert_history = production_alerting.get_alert_history(limit=limit)
        
        return {
            "alert_history": alert_history,
            "total_returned": len(alert_history),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {str(e)}")

@router.get("/summary")
async def get_alert_summary(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get comprehensive alert summary and system health overview."""
    try:
        alert_summary = await production_alerting.get_alert_summary()
        active_alerts = production_alerting.get_active_alerts()
        
        # Enhance summary with current alert breakdown
        alert_summary.update({
            "current_alerts": {
                "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
                "high": len([a for a in active_alerts if a["severity"] == "high"]),
                "medium": len([a for a in active_alerts if a["severity"] == "medium"]),
                "low": len([a for a in active_alerts if a["severity"] == "low"])
            },
            "alert_types": _get_alert_type_breakdown(active_alerts),
            "most_affected_services": _get_most_affected_services(active_alerts),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        return alert_summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert summary: {str(e)}")

@router.post("/check")
async def trigger_health_check(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Manually trigger a comprehensive health check and alerting run."""
    try:
        result = await production_alerting.check_system_health_and_alert()
        
        return {
            "check_result": result,
            "message": "Health check completed successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/services/{service_name}")
async def get_service_alerts(
    service_name: str,
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get alerts for a specific service."""
    try:
        active_alerts = production_alerting.get_active_alerts()
        alert_history = production_alerting.get_alert_history()
        
        # Filter alerts for the specified service
        service_active = [
            alert for alert in active_alerts 
            if alert["service_name"] == service_name
        ]
        
        service_history = [
            alert for alert in alert_history 
            if alert["service_name"] == service_name
        ]
        
        return {
            "service_name": service_name,
            "active_alerts": service_active,
            "recent_history": service_history[:20],  # Last 20 alerts
            "summary": {
                "total_active": len(service_active),
                "highest_severity": _get_highest_severity(service_active),
                "last_alert": service_history[0]["timestamp"] if service_history else None
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service alerts: {str(e)}")

@router.get("/health-overview")
async def get_health_overview(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get a comprehensive health overview combining monitoring and alerting data."""
    try:
        # Get alert summary
        alert_summary = await production_alerting.get_alert_summary()
        active_alerts = production_alerting.get_active_alerts()
        
        # Determine overall system health based on alerts
        overall_health = "healthy"
        if any(alert["severity"] == "critical" for alert in active_alerts):
            overall_health = "critical"
        elif any(alert["severity"] == "high" for alert in active_alerts):
            overall_health = "degraded"
        elif len(active_alerts) > 0:
            overall_health = "warning"
        
        return {
            "overall_health": overall_health,
            "alert_summary": alert_summary,
            "active_alerts_count": len(active_alerts),
            "critical_issues": [
                alert for alert in active_alerts 
                if alert["severity"] == "critical"
            ],
            "health_score": _calculate_health_score(active_alerts),
            "recommendations": _get_health_recommendations(active_alerts),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health overview: {str(e)}")

def _determine_overall_alert_level(alerts: List[Dict[str, Any]]) -> str:
    """Determine the overall alert level based on active alerts."""
    if not alerts:
        return "none"
    
    severities = [alert["severity"] for alert in alerts]
    if "critical" in severities:
        return "critical"
    elif "high" in severities:
        return "high"
    elif "medium" in severities:
        return "medium"
    else:
        return "low"

def _get_alert_type_breakdown(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get breakdown of alerts by type."""
    type_counts = {}
    for alert in alerts:
        alert_type = alert.get("alert_type", "unknown")
        type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
    return type_counts

def _get_most_affected_services(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get services with the most alerts."""
    service_counts = {}
    service_severities = {}
    
    for alert in alerts:
        service_name = alert.get("service_name", "unknown")
        severity = alert.get("severity", "low")
        
        service_counts[service_name] = service_counts.get(service_name, 0) + 1
        
        if service_name not in service_severities:
            service_severities[service_name] = severity
        elif _severity_rank(severity) > _severity_rank(service_severities[service_name]):
            service_severities[service_name] = severity
    
    # Sort by alert count and severity
    sorted_services = sorted(
        service_counts.items(), 
        key=lambda x: (x[1], _severity_rank(service_severities[x[0]])), 
        reverse=True
    )
    
    return [
        {
            "service_name": service,
            "alert_count": count,
            "highest_severity": service_severities[service]
        }
        for service, count in sorted_services[:10]  # Top 10
    ]

def _severity_rank(severity: str) -> int:
    """Convert severity to numeric rank for sorting."""
    ranks = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return ranks.get(severity, 0)

def _get_highest_severity(alerts: List[Dict[str, Any]]) -> str:
    """Get the highest severity level from a list of alerts."""
    if not alerts:
        return "none"
    
    severities = [alert["severity"] for alert in alerts]
    if "critical" in severities:
        return "critical"
    elif "high" in severities:
        return "high"
    elif "medium" in severities:
        return "medium"
    else:
        return "low"

def _calculate_health_score(alerts: List[Dict[str, Any]]) -> int:
    """Calculate a health score from 0-100 based on active alerts."""
    if not alerts:
        return 100
    
    # Deduct points based on alert severity
    score = 100
    for alert in alerts:
        severity = alert.get("severity", "low")
        if severity == "critical":
            score -= 25
        elif severity == "high":
            score -= 10
        elif severity == "medium":
            score -= 5
        else:  # low
            score -= 2
    
    return max(0, score)

def _get_health_recommendations(alerts: List[Dict[str, Any]]) -> List[str]:
    """Generate health recommendations based on active alerts."""
    recommendations = []
    
    if not alerts:
        recommendations.append("System is healthy - no active alerts")
        return recommendations
    
    # Critical alerts
    critical_alerts = [a for a in alerts if a["severity"] == "critical"]
    if critical_alerts:
        recommendations.append("⚠️  URGENT: Address critical alerts immediately")
        for alert in critical_alerts[:3]:  # Top 3 critical
            recommendations.append(f"   - {alert['service_name']}: {alert['message']}")
    
    # Service-specific recommendations
    service_counts = {}
    for alert in alerts:
        service_name = alert.get("service_name", "unknown")
        service_counts[service_name] = service_counts.get(service_name, 0) + 1
    
    for service, count in service_counts.items():
        if count >= 3:
            recommendations.append(f"🔧 Service '{service}' has multiple alerts ({count}) - investigate service health")
    
    # General recommendations
    if len(alerts) > 10:
        recommendations.append("📊 High number of active alerts - consider system-wide health check")
    
    return recommendations