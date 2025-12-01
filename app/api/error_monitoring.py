"""
Error Monitoring API Endpoints

Provides endpoints for monitoring, analyzing, and reporting external API errors
and system failures for improved observability and debugging.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.security import require_system_rate_limit
from app.core.error_logging import error_logger, ErrorSeverity, ErrorCategory

router = APIRouter(prefix="/api/v1/errors", tags=["error-monitoring"])

@router.get("/analytics")
async def get_error_analytics(
    service: Optional[str] = Query(None, description="Filter by specific service"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get comprehensive error analytics for external API failures."""
    try:
        analytics = error_logger.get_error_analytics(service=service, hours=hours)
        
        return {
            "error_analytics": analytics,
            "meta": {
                "endpoint": "/errors/analytics",
                "filtered_service": service,
                "analysis_period_hours": hours,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error analytics: {str(e)}")

@router.get("/services/{service}/health-score")
async def get_service_health_score(
    service: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get health score for a specific service based on recent errors."""
    try:
        health_score = error_logger.get_service_health_score(service=service, hours=hours)
        service_stats = error_logger.error_stats.get(service, {})
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "degraded"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            "service_health": {
                "service": service,
                "health_score": round(health_score, 2),
                "health_status": health_status,
                "analysis_period_hours": hours,
                "total_errors": service_stats.get("total_errors", 0),
                "error_rate_per_hour": service_stats.get("error_rate", 0),
                "last_error": service_stats.get("last_error").isoformat() if service_stats.get("last_error") else None,
                "error_categories": dict(service_stats.get("categories", {})),
                "severity_distribution": dict(service_stats.get("severity_counts", {})),
                "recommendations": _generate_health_recommendations(health_score, service_stats)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service health score: {str(e)}")

@router.get("/summary")
async def get_error_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get summary of errors across all services."""
    try:
        analytics = error_logger.get_error_analytics(hours=hours)
        
        # Calculate system-wide health metrics
        total_errors = analytics["overall_stats"]["total_errors"]
        total_services = len(analytics["services"])
        
        # Calculate average health score across all services
        health_scores = []
        service_health_details = {}
        
        for service in analytics["services"].keys():
            score = error_logger.get_service_health_score(service, hours)
            health_scores.append(score)
            service_health_details[service] = {
                "health_score": round(score, 2),
                "error_count": analytics["services"][service]["error_count"],
                "error_rate": analytics["services"][service]["error_rate_per_hour"]
            }
        
        avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 100.0
        
        # Identify problematic services
        problematic_services = [
            service for service, score in zip(analytics["services"].keys(), health_scores)
            if score < 75
        ]
        
        # Top error categories
        top_categories = sorted(
            analytics["overall_stats"]["error_categories"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "error_summary": {
                "period_hours": hours,
                "system_health": {
                    "average_health_score": round(avg_health_score, 2),
                    "total_errors": total_errors,
                    "error_rate_per_hour": round(total_errors / hours, 2),
                    "affected_services": total_services,
                    "problematic_services": problematic_services
                },
                "service_health_details": service_health_details,
                "error_patterns": {
                    "top_categories": top_categories,
                    "severity_distribution": dict(analytics["overall_stats"]["severity_distribution"]),
                    "most_problematic_services": analytics["overall_stats"]["top_error_services"][:3]
                },
                "alerts": _generate_system_alerts(analytics, avg_health_score, problematic_services)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error summary: {str(e)}")

@router.get("/categories")
async def get_error_categories(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get information about error categories and their descriptions."""
    
    category_descriptions = {
        ErrorCategory.NETWORK.value: {
            "description": "Network connectivity issues, DNS failures, SSL/TLS errors",
            "severity": "medium",
            "common_causes": ["Network outages", "DNS issues", "SSL certificate problems"],
            "solutions": ["Check network connectivity", "Verify DNS configuration", "Update SSL certificates"]
        },
        ErrorCategory.AUTHENTICATION.value: {
            "description": "API key, token, or credential failures",
            "severity": "high",
            "common_causes": ["Invalid API keys", "Expired tokens", "Incorrect credentials"],
            "solutions": ["Update API keys", "Refresh tokens", "Check credential configuration"]
        },
        ErrorCategory.RATE_LIMIT.value: {
            "description": "API rate limiting and quota exceeded errors",
            "severity": "medium",
            "common_causes": ["Too many requests", "Quota exceeded", "Burst limit reached"],
            "solutions": ["Implement request throttling", "Upgrade API plan", "Add retry with backoff"]
        },
        ErrorCategory.DATA_FORMAT.value: {
            "description": "Data parsing, format, or structure issues",
            "severity": "medium",
            "common_causes": ["JSON/XML parsing errors", "Schema changes", "Encoding issues"],
            "solutions": ["Update data parsers", "Validate response schemas", "Handle format changes"]
        },
        ErrorCategory.SERVER_ERROR.value: {
            "description": "External service server errors (5xx status codes)",
            "severity": "high",
            "common_causes": ["Service outages", "Server overload", "Internal service errors"],
            "solutions": ["Monitor service status", "Implement circuit breakers", "Add retry logic"]
        },
        ErrorCategory.TIMEOUT.value: {
            "description": "Request timeout and response delay issues",
            "severity": "medium",
            "common_causes": ["Slow service response", "Network latency", "Resource contention"],
            "solutions": ["Increase timeout values", "Optimize requests", "Use async processing"]
        },
        ErrorCategory.CONFIGURATION.value: {
            "description": "Service configuration and setup issues",
            "severity": "high",
            "common_causes": ["Wrong endpoints", "Missing parameters", "Invalid configuration"],
            "solutions": ["Review configuration", "Update endpoints", "Validate parameters"]
        },
        ErrorCategory.UNKNOWN.value: {
            "description": "Unclassified or unknown errors",
            "severity": "low",
            "common_causes": ["Unrecognized errors", "New error types", "Edge cases"],
            "solutions": ["Investigate error details", "Update error classification", "Add monitoring"]
        }
    }
    
    return {
        "error_categories": category_descriptions,
        "severity_levels": {
            ErrorSeverity.LOW.value: "Minor issues that don't significantly impact functionality",
            ErrorSeverity.MEDIUM.value: "Moderate issues that may affect some functionality",
            ErrorSeverity.HIGH.value: "Significant issues that impact core functionality",
            ErrorSeverity.CRITICAL.value: "Severe issues requiring immediate attention"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/recent")
async def get_recent_errors(
    service: Optional[str] = Query(None, description="Filter by specific service"),
    category: Optional[str] = Query(None, description="Filter by error category"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of errors to return"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get recent error records with optional filtering."""
    try:
        recent_errors = []
        
        # Get services to check
        services_to_check = [service] if service else list(error_logger.error_buffer.keys())
        
        for svc in services_to_check:
            if svc not in error_logger.error_buffer:
                continue
            
            for error_record in reversed(list(error_logger.error_buffer[svc])):
                # Apply filters
                if category and error_record.error_category.value != category:
                    continue
                if severity and error_record.severity.value != severity:
                    continue
                
                recent_errors.append({
                    "timestamp": error_record.timestamp.isoformat(),
                    "service": error_record.service,
                    "endpoint": error_record.endpoint,
                    "method": error_record.method,
                    "status_code": error_record.status_code,
                    "error_message": error_record.error_message,
                    "category": error_record.error_category.value,
                    "severity": error_record.severity.value,
                    "response_time_ms": error_record.response_time_ms,
                    "retry_count": error_record.retry_count,
                    "context": error_record.context
                })
                
                if len(recent_errors) >= limit:
                    break
            
            if len(recent_errors) >= limit:
                break
        
        return {
            "recent_errors": {
                "total_returned": len(recent_errors),
                "filters_applied": {
                    "service": service,
                    "category": category,
                    "severity": severity,
                    "limit": limit
                },
                "errors": recent_errors
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent errors: {str(e)}")

@router.post("/clear-old")
async def clear_old_errors(
    days: int = Query(7, ge=1, le=30, description="Age threshold in days"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Clear old error records to prevent memory bloat."""
    try:
        # Count errors before clearing
        total_errors_before = sum(len(buffer) for buffer in error_logger.error_buffer.values())
        
        # Clear old errors
        error_logger.clear_old_errors(days=days)
        
        # Count errors after clearing
        total_errors_after = sum(len(buffer) for buffer in error_logger.error_buffer.values())
        
        return {
            "cleanup_result": {
                "days_threshold": days,
                "errors_before": total_errors_before,
                "errors_after": total_errors_after,
                "errors_removed": total_errors_before - total_errors_after,
                "active_services": len(error_logger.error_buffer),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear old errors: {str(e)}")

def _generate_health_recommendations(health_score: float, service_stats: Dict) -> list:
    """Generate recommendations based on service health score and stats."""
    recommendations = []
    
    if health_score < 25:
        recommendations.append("URGENT: Service is in critical condition - immediate investigation required")
    elif health_score < 50:
        recommendations.append("Service health is poor - review error patterns and fix underlying issues")
    elif health_score < 75:
        recommendations.append("Service health is degraded - monitor closely and address recurring issues")
    
    # Check for specific issues
    categories = service_stats.get("categories", {})
    
    if categories.get("authentication", 0) > 0:
        recommendations.append("Authentication issues detected - verify API credentials")
    
    if categories.get("rate_limit", 0) > 3:
        recommendations.append("Frequent rate limiting - implement request throttling or upgrade API plan")
    
    if categories.get("timeout", 0) > 2:
        recommendations.append("Timeout issues - consider increasing timeout values or optimizing requests")
    
    if categories.get("server_error", 0) > 1:
        recommendations.append("Server errors detected - monitor external service status")
    
    if not recommendations:
        recommendations.append("Service health is good - continue monitoring")
    
    return recommendations

def _generate_system_alerts(analytics: Dict, avg_health_score: float, problematic_services: list) -> list:
    """Generate system-level alerts based on error analytics."""
    alerts = []
    
    total_errors = analytics["overall_stats"]["total_errors"]
    
    if avg_health_score < 50:
        alerts.append({
            "level": "critical",
            "message": f"System-wide health is critical (score: {avg_health_score:.1f})",
            "action": "Immediate investigation required"
        })
    elif avg_health_score < 75:
        alerts.append({
            "level": "warning",
            "message": f"System health is degraded (score: {avg_health_score:.1f})",
            "action": "Monitor and address issues"
        })
    
    if len(problematic_services) >= 3:
        alerts.append({
            "level": "warning",
            "message": f"Multiple services affected: {', '.join(problematic_services)}",
            "action": "Review system-wide issues"
        })
    
    if total_errors > 100:
        alerts.append({
            "level": "warning",
            "message": f"High error volume: {total_errors} errors in analysis period",
            "action": "Investigate error patterns"
        })
    
    # Check for authentication issues across services
    auth_errors = analytics["overall_stats"]["error_categories"].get("authentication", 0)
    if auth_errors > 5:
        alerts.append({
            "level": "critical",
            "message": f"Multiple authentication failures detected ({auth_errors} errors)",
            "action": "Check API credentials across all services"
        })
    
    if not alerts:
        alerts.append({
            "level": "info",
            "message": "System error levels are within normal parameters",
            "action": "Continue monitoring"
        })
    
    return alerts