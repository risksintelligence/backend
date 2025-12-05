"""
Health Monitoring API Endpoints

Provides comprehensive health monitoring endpoints for system observability,
monitoring external API status, and ensuring production readiness.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.core.security import require_system_rate_limit
from app.services.health_monitor import health_monitor, HealthStatus

router = APIRouter(prefix="/api/v1/health", tags=["health-monitoring"])

@router.get("/comprehensive")
async def get_comprehensive_health(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get comprehensive system health status including all services and APIs."""
    try:
        health_summary = await health_monitor.get_system_health_summary()
        return {
            "health_check": health_summary,
            "meta": {
                "endpoint": "/health/comprehensive",
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/external-apis")
async def get_external_api_health(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get health status specifically for external API integrations."""
    try:
        all_health = await health_monitor.check_all_services()
        
        # Filter for external APIs only
        external_apis = {
            name: health for name, health in all_health.items()
            if name.startswith("api_")
        }
        
        # Calculate API-specific metrics
        total_apis = len(external_apis)
        healthy_apis = sum(1 for h in external_apis.values() if h.status == HealthStatus.HEALTHY)
        degraded_apis = sum(1 for h in external_apis.values() if h.status == HealthStatus.DEGRADED)
        unhealthy_apis = sum(1 for h in external_apis.values() if h.status == HealthStatus.UNHEALTHY)
        
        # API response time analysis
        api_response_times = {
            name: health.response_time_ms for name, health in external_apis.items()
            if health.response_time_ms is not None
        }
        
        return {
            "external_api_health": {
                "summary": {
                    "total_apis": total_apis,
                    "healthy": healthy_apis,
                    "degraded": degraded_apis,
                    "unhealthy": unhealthy_apis,
                    "availability_percentage": round((healthy_apis / total_apis) * 100, 1) if total_apis > 0 else 0
                },
                "api_details": {
                    name: {
                        "name": health.name,
                        "status": health.status.value,
                        "response_time_ms": health.response_time_ms,
                        "last_check": health.last_check.isoformat(),
                        "error_message": health.error_message,
                        "has_api_key": health.metadata.get("has_api_key", False) if health.metadata else False,
                        "endpoint": health.metadata.get("endpoint") if health.metadata else None,
                        "service_type": health.metadata.get("service_type") if health.metadata else None
                    }
                    for name, health in external_apis.items()
                },
                "performance_metrics": {
                    "response_times_ms": api_response_times,
                    "fastest_api": min(api_response_times.items(), key=lambda x: x[1])[0] if api_response_times else None,
                    "slowest_api": max(api_response_times.items(), key=lambda x: x[1])[0] if api_response_times else None
                },
                "recommendations": [
                    "Configure missing API keys for better data access",
                    "Monitor rate limits to avoid degraded service",
                    "Consider implementing circuit breakers for unreliable APIs"
                ] if unhealthy_apis > 0 or degraded_apis > 0 else [
                    "All external APIs are functioning normally"
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"External API health check failed: {str(e)}")

@router.get("/internal-services")
async def get_internal_services_health(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get health status for internal services (database, cache, ML models, etc)."""
    try:
        all_health = await health_monitor.check_all_services()
        
        # Filter for internal services only
        internal_services = {
            name: health for name, health in all_health.items()
            if not name.startswith("api_")
        }
        
        # Calculate internal service metrics
        total_services = len(internal_services)
        healthy_services = sum(1 for h in internal_services.values() if h.status == HealthStatus.HEALTHY)
        critical_services = ["database", "redis_cache", "ml_models"]
        critical_health = {
            name: health for name, health in internal_services.items()
            if name in critical_services
        }
        
        return {
            "internal_services_health": {
                "summary": {
                    "total_services": total_services,
                    "healthy": healthy_services,
                    "degraded": sum(1 for h in internal_services.values() if h.status == HealthStatus.DEGRADED),
                    "unhealthy": sum(1 for h in internal_services.values() if h.status == HealthStatus.UNHEALTHY),
                    "reliability_percentage": round((healthy_services / total_services) * 100, 1) if total_services > 0 else 0
                },
                "critical_services": {
                    name: {
                        "name": health.name,
                        "status": health.status.value,
                        "response_time_ms": health.response_time_ms,
                        "last_check": health.last_check.isoformat(),
                        "error_message": health.error_message,
                        "metadata": health.metadata
                    }
                    for name, health in critical_health.items()
                },
                "all_services": {
                    name: {
                        "name": health.name,
                        "status": health.status.value,
                        "response_time_ms": health.response_time_ms,
                        "last_check": health.last_check.isoformat(),
                        "error_message": health.error_message
                    }
                    for name, health in internal_services.items()
                },
                "system_readiness": {
                    "production_ready": all(h.status == HealthStatus.HEALTHY for h in critical_health.values()),
                    "critical_issues": [h.name for h in critical_health.values() if h.status == HealthStatus.UNHEALTHY],
                    "warnings": [h.name for h in critical_health.values() if h.status == HealthStatus.DEGRADED]
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal services health check failed: {str(e)}")

@router.get("/quick")
async def get_quick_health_check(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Quick health check for basic system status."""
    try:
        # Get cached health data if available
        cached_data, _ = health_monitor.cache.get("system_health")
        
        if cached_data:
            # Use cached data for quick response
            services = cached_data
            healthy_count = sum(1 for s in services.values() if s.get("status") == "healthy")
            total_count = len(services)
            overall_healthy = healthy_count == total_count
        else:
            # Perform minimal health check
            minimal_checks = {
                "database": await _quick_database_check(),
                "cache": await _quick_cache_check()
            }
            
            healthy_count = sum(1 for status in minimal_checks.values() if status)
            total_count = len(minimal_checks)
            overall_healthy = healthy_count == total_count
            services = minimal_checks
        
        return {
            "quick_health": {
                "status": "healthy" if overall_healthy else "unhealthy",
                "healthy_services": healthy_count,
                "total_services": total_count,
                "health_percentage": round((healthy_count / total_count) * 100, 1),
                "services": services,
                "using_cache": cached_data is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "quick_health": {
                "status": "error",
                "error": str(e)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

async def _quick_database_check() -> bool:
    """Quick database connectivity check."""
    try:
        from app.db import get_db
        from sqlalchemy import text
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except:
        return False

async def _quick_cache_check() -> bool:
    """Quick cache connectivity check."""
    try:
        test_key = f"quick_health_{datetime.utcnow().timestamp()}"
        health_monitor.cache.set(test_key, {"test": True}, "health_monitor")
        cached_data, _ = health_monitor.cache.get(test_key)
        # Note: not invalidating to avoid method not found error
        return cached_data is not None and cached_data.get("test") is True
    except:
        return False

@router.get("/production-readiness")
async def get_production_readiness(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Comprehensive production readiness assessment."""
    try:
        health_summary = await health_monitor.get_system_health_summary()
        
        # Production readiness criteria
        critical_services = ["database", "redis_cache", "ml_models", "background_refresh"]
        critical_health = {
            name: health for name, health in health_summary["services"].items()
            if any(service in name for service in critical_services)
        }
        
        api_health = {
            name: health for name, health in health_summary["services"].items()
            if name.startswith("api_")
        }
        
        # Calculate readiness scores
        critical_score = sum(1 for h in critical_health.values() if h["status"] == "healthy") / len(critical_health) * 100 if critical_health else 100
        api_score = sum(1 for h in api_health.values() if h["status"] in ["healthy", "degraded"]) / len(api_health) * 100 if api_health else 100
        overall_score = (critical_score * 0.7 + api_score * 0.3)  # Weight critical services higher
        
        # Determine production readiness
        if overall_score >= 95:
            readiness_status = "production_ready"
        elif overall_score >= 80:
            readiness_status = "ready_with_warnings"
        elif overall_score >= 60:
            readiness_status = "needs_attention"
        else:
            readiness_status = "not_ready"
        
        # Generate production checklist
        checklist = []
        
        # Critical services check
        for name, health in critical_health.items():
            if health["status"] == "healthy":
                checklist.append({"item": f"{health['name']}", "status": "✅ Ready", "critical": True})
            else:
                checklist.append({"item": f"{health['name']}", "status": f"❌ {health.get('error_message', 'Needs attention')}", "critical": True})
        
        # API services check
        working_apis = sum(1 for h in api_health.values() if h["status"] in ["healthy", "degraded"])
        total_apis = len(api_health)
        checklist.append({
            "item": "External API Integration", 
            "status": f"{'✅' if working_apis >= total_apis * 0.8 else '⚠️'} {working_apis}/{total_apis} APIs available",
            "critical": False
        })
        
        # Environment configuration check
        env_checks = []
        if hasattr(health_monitor.settings, 'redis_url') and health_monitor.settings.redis_url:
            env_checks.append("✅ Redis configured")
        else:
            env_checks.append("❌ Redis not configured")
        
        if hasattr(health_monitor.settings, 'database_url'):
            env_checks.append("✅ Database configured")
        else:
            env_checks.append("❌ Database not configured")
        
        checklist.append({
            "item": "Environment Configuration",
            "status": ", ".join(env_checks),
            "critical": True
        })
        
        return {
            "production_readiness": {
                "status": readiness_status,
                "overall_score": round(overall_score, 1),
                "critical_services_score": round(critical_score, 1),
                "api_services_score": round(api_score, 1),
                "checklist": checklist,
                "deployment_recommendations": [
                    "Monitor external API rate limits in production",
                    "Set up automated health monitoring alerts",
                    "Configure proper Redis persistence for production",
                    "Implement graceful shutdown procedures",
                    "Set up log aggregation and monitoring"
                ] if readiness_status in ["production_ready", "ready_with_warnings"] else [
                    "Fix critical service issues before deployment",
                    "Verify all environment variables are set",
                    "Test ML model loading and predictions",
                    "Ensure database connectivity is stable"
                ],
                "next_steps": health_summary.get("recommendations", [])
            },
            "detailed_health": health_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Production readiness check failed: {str(e)}")

@router.get("/alerts")
async def get_health_alerts(
    severity: Optional[str] = Query("all", description="Filter by severity: critical, warning, info, all"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get health alerts and warnings that require attention."""
    try:
        health_summary = await health_monitor.get_system_health_summary()
        
        alerts = []
        
        # Critical alerts (unhealthy services)
        for name, health in health_summary["services"].items():
            if health["status"] == "unhealthy":
                alerts.append({
                    "severity": "critical",
                    "service": health["name"],
                    "message": f"{health['name']} is unhealthy: {health.get('error_message', 'Unknown error')}",
                    "timestamp": health["last_check"],
                    "action_required": True
                })
        
        # Warning alerts (degraded services)
        for name, health in health_summary["services"].items():
            if health["status"] == "degraded":
                alerts.append({
                    "severity": "warning",
                    "service": health["name"],
                    "message": f"{health['name']} is degraded: {health.get('error_message', 'Performance issues')}",
                    "timestamp": health["last_check"],
                    "action_required": False
                })
        
        # Info alerts (performance)
        slow_services = [
            health for health in health_summary["services"].values()
            if health.get("response_time_ms") and health["response_time_ms"] > 3000
        ]
        
        for health in slow_services:
            alerts.append({
                "severity": "info",
                "service": health["name"],
                "message": f"{health['name']} has slow response time: {health['response_time_ms']:.1f}ms",
                "timestamp": health["last_check"],
                "action_required": False
            })
        
        # Filter by severity
        if severity != "all":
            alerts = [alert for alert in alerts if alert["severity"] == severity]
        
        # Sort by severity and timestamp
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["timestamp"]))
        
        return {
            "health_alerts": {
                "total_alerts": len(alerts),
                "critical_count": len([a for a in alerts if a["severity"] == "critical"]),
                "warning_count": len([a for a in alerts if a["severity"] == "warning"]),
                "info_count": len([a for a in alerts if a["severity"] == "info"]),
                "alerts": alerts,
                "requires_immediate_action": any(alert["action_required"] for alert in alerts)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health alerts check failed: {str(e)}")
