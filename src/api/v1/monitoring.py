"""API endpoints for monitoring, metrics, and observability."""
from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from typing import Dict, Any

from backend.src.monitoring.observability import get_observability_service, get_prometheus_metrics

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint for scraping."""
    # Collect latest metrics
    observability = get_observability_service()
    
    # Trigger metric collection if needed
    if observability.should_collect_metrics():
        await observability.collect_all_metrics()
    
    # Return Prometheus formatted metrics
    return get_prometheus_metrics()


@router.get("/health")
async def monitoring_health():
    """Health check endpoint for monitoring systems."""
    observability = get_observability_service()
    
    try:
        metrics_summary = await observability.get_metrics_summary()
        
        # Determine overall health
        geri_healthy = metrics_summary.get("geri", {}).get("status") == "healthy"
        system_healthy = metrics_summary.get("system", {}).get("status") == "healthy"
        data_fresh = metrics_summary.get("data_quality", {}).get("freshness_ratio", 0) > 0.8
        
        overall_status = "healthy" if (geri_healthy and system_healthy and data_fresh) else "warning"
        
        return {
            "status": overall_status,
            "timestamp": metrics_summary.get("timestamp"),
            "components": {
                "geri": geri_healthy,
                "system": system_healthy,
                "data_quality": data_fresh,
                "ml": metrics_summary.get("ml", {}).get("status") == "operational"
            },
            "metrics": metrics_summary
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": None,
            "components": {
                "geri": False,
                "system": False,
                "data_quality": False,
                "ml": False
            }
        }


@router.get("/summary")
async def metrics_summary():
    """Get a summary of key system metrics."""
    observability = get_observability_service()
    return await observability.get_metrics_summary()


@router.get("/alerts/status")
async def alert_monitoring_status():
    """Get status of alert monitoring system."""
    try:
        from backend.src.services.scenario_service import get_alert_service
        alert_service = get_alert_service()
        
        subscriptions = alert_service.list_subscriptions()
        recent_deliveries = alert_service.deliveries()
        
        # Calculate delivery stats
        recent_count = len([d for d in recent_deliveries if d.delivered_at])
        
        return {
            "status": "operational",
            "subscriptions": {
                "total": len(subscriptions),
                "active": len([s for s in subscriptions if s.get("status") == "active"])
            },
            "deliveries": {
                "recent_24h": recent_count,
                "last_delivery": recent_deliveries[0].delivered_at if recent_deliveries else None
            },
            "health": "healthy" if len(subscriptions) > 0 else "warning"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "health": "unhealthy"
        }


@router.get("/jobs/status")  
async def background_jobs_status():
    """Get status of background jobs and cron tasks."""
    from backend.src.services.admin_service import get_admin_service
    
    try:
        admin_service = get_admin_service()
        
        # Get recent audit log entries for job monitoring
        recent_logs = await admin_service.get_audit_log(100)
        
        # Filter job-related entries
        job_logs = [log for log in recent_logs if any(keyword in log.get("action", "") for keyword in ["job", "cron", "training", "ingestion", "monitoring"])]
        
        # Categorize by job type
        job_stats = {}
        for log in job_logs[:20]:  # Last 20 job-related events
            action = log.get("action", "unknown")
            if action not in job_stats:
                job_stats[action] = {
                    "count": 0,
                    "last_run": None,
                    "actor": log.get("actor", "unknown")
                }
            job_stats[action]["count"] += 1
            if not job_stats[action]["last_run"]:
                job_stats[action]["last_run"] = log.get("occurred_at")
        
        return {
            "status": "operational",
            "jobs": job_stats,
            "recent_activity": len(job_logs),
            "monitoring": {
                "data_ingestion": "scheduled",
                "geri_computation": "scheduled", 
                "alert_monitoring": "scheduled",
                "ml_training": "scheduled"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "jobs": {}
        }


@router.post("/collect")
async def trigger_metrics_collection():
    """Manually trigger metrics collection."""
    observability = get_observability_service()
    
    try:
        metrics = await observability.collect_all_metrics()
        
        return {
            "success": True,
            "message": "Metrics collection completed",
            "timestamp": metrics.timestamp,
            "geri_value": metrics.geri_value,
            "system_health": {
                "cpu": metrics.cpu_percent,
                "memory": metrics.memory_percent
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Metrics collection failed"
        }