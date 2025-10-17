"""
Monitoring API Routes for RiskX

Provides endpoints for accessing system metrics, health status,
and operational intelligence for production monitoring.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

from src.monitoring.metrics_collector import metrics_collector
from src.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def get_health_status():
    """Get comprehensive system health status"""
    try:
        health_status = await metrics_collector.get_system_health_status()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/metrics/api")
async def get_api_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get API performance metrics for the specified time period"""
    try:
        metrics = await metrics_collector.get_api_performance_summary(hours=hours)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API metrics: {str(e)}")


@router.get("/metrics/system")
async def get_system_metrics():
    """Get current system resource metrics"""
    try:
        system_metrics = await metrics_collector.collect_system_metrics()
        if system_metrics:
            from dataclasses import asdict
            return asdict(system_metrics)
        else:
            raise HTTPException(status_code=500, detail="Failed to collect system metrics")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System metrics error: {str(e)}")


@router.get("/metrics/data-quality")
async def get_data_quality_metrics():
    """Get data quality metrics for all data sources"""
    try:
        quality_report = await metrics_collector.get_data_quality_report()
        return quality_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data quality metrics error: {str(e)}")


@router.get("/status")
async def get_service_status():
    """Get overall service status and uptime information"""
    try:
        # Get basic service information
        uptime_info = {
            "service": "RiskX API",
            "version": "1.0.0",
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "status": "operational"
        }
        
        # Add health metrics
        health_status = await metrics_collector.get_system_health_status()
        uptime_info["health"] = health_status
        
        # Add API performance summary
        api_metrics = await metrics_collector.get_api_performance_summary(hours=1)
        uptime_info["recent_performance"] = api_metrics
        
        return uptime_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/metrics/export")
async def export_metrics():
    """Export current metrics to files (admin endpoint)"""
    try:
        await metrics_collector.export_metrics_to_file()
        return {
            "status": "success",
            "message": "Metrics exported successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics export failed: {str(e)}")


@router.get("/alerts")
async def get_system_alerts():
    """Get current system alerts and warnings"""
    try:
        health_status = await metrics_collector.get_system_health_status()
        
        alerts = []
        
        # Check for system alerts
        if health_status.get("status") in ["warning", "degraded", "critical"]:
            alerts.extend([
                {
                    "type": "system",
                    "severity": health_status["status"],
                    "message": issue,
                    "timestamp": datetime.now().isoformat()
                }
                for issue in health_status.get("issues", [])
            ])
        
        # Check for data quality alerts
        data_quality = await metrics_collector.get_data_quality_report()
        for source_name, source_data in data_quality.get("sources", {}).items():
            if source_data.get("status") == "degraded":
                alerts.append({
                    "type": "data_quality",
                    "severity": "warning",
                    "message": f"Data source {source_name} quality degraded",
                    "source": source_name,
                    "details": source_data,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check for API performance alerts
        api_metrics = await metrics_collector.get_api_performance_summary(hours=1)
        if api_metrics.get("error_rate", 0) > 0.05:  # 5% error rate threshold
            alerts.append({
                "type": "api_performance",
                "severity": "warning",
                "message": f"High API error rate: {api_metrics['error_rate']:.2%}",
                "details": api_metrics,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/metrics/business")
async def get_business_metrics():
    """Get business intelligence and usage metrics"""
    try:
        # Get API usage patterns
        api_summary = await metrics_collector.get_api_performance_summary(hours=24)
        
        # Get data source health
        data_quality = await metrics_collector.get_data_quality_report()
        
        # Calculate business metrics
        active_data_sources = len([
            source for source, data in data_quality.get("sources", {}).items()
            if data.get("status") == "healthy"
        ])
        
        total_data_points = sum([
            source_data.get("data_points", 0) 
            for source_data in data_quality.get("sources", {}).values()
        ])
        
        business_metrics = {
            "daily_api_requests": api_summary.get("total_requests", 0),
            "api_uptime_percent": (1 - api_summary.get("error_rate", 0)) * 100,
            "active_data_sources": active_data_sources,
            "total_data_sources": len(data_quality.get("sources", {})),
            "total_data_points": total_data_points,
            "avg_response_time_ms": api_summary.get("avg_response_time_ms", 0),
            "top_endpoints": api_summary.get("top_endpoints", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return business_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Business metrics error: {str(e)}")