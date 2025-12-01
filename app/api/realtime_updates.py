"""
Real-time Updates API

Provides endpoints to manage real-time data refresh, monitoring, and WebSocket subscriptions
for supply chain cascade updates.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from app.core.security import require_system_rate_limit
from app.services.realtime_refresh import get_refresh_service, RefreshPriority, DataSourceStatus

router = APIRouter(prefix="/api/v1/realtime", tags=["realtime"])


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@router.get("/status")
async def get_refresh_status(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get current status of real-time refresh service."""
    
    try:
        refresh_service = get_refresh_service()
        status = refresh_service.get_refresh_status()
        
        return {
            "status": "success",
            "refresh_service": status,
            "retrieved_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to get refresh status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.post("/start")
async def start_refresh_service(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Start the real-time refresh service."""
    
    try:
        refresh_service = get_refresh_service()
        await refresh_service.start_refresh_service()
        
        logger.info("Real-time refresh service started via API")
        
        return {
            "status": "success",
            "message": "Real-time refresh service started",
            "started_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to start refresh service: {e}")
        raise HTTPException(status_code=500, detail=f"Service start failed: {str(e)}")


@router.post("/stop")
async def stop_refresh_service(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Stop the real-time refresh service."""
    
    try:
        refresh_service = get_refresh_service()
        await refresh_service.stop_refresh_service()
        
        logger.info("Real-time refresh service stopped via API")
        
        return {
            "status": "success",
            "message": "Real-time refresh service stopped",
            "stopped_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop refresh service: {e}")
        raise HTTPException(status_code=500, detail=f"Service stop failed: {str(e)}")


@router.post("/force-refresh/{data_source}")
async def force_refresh_data_source(
    data_source: str,
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Force an immediate refresh of a specific data source."""
    
    try:
        refresh_service = get_refresh_service()
        success = refresh_service.force_refresh(data_source)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Data source '{data_source}' not found")
        
        logger.info(f"Forced refresh requested for {data_source}")
        
        return {
            "status": "success",
            "message": f"Forced refresh scheduled for {data_source}",
            "data_source": data_source,
            "scheduled_at": _now_iso()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to force refresh {data_source}: {e}")
        raise HTTPException(status_code=500, detail=f"Force refresh failed: {str(e)}")


@router.put("/priority/{data_source}")
async def update_refresh_priority(
    data_source: str,
    priority: str = Query(..., description="Priority level: critical, high, medium, low"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Update the refresh priority for a data source."""
    
    try:
        # Validate priority level
        try:
            new_priority = RefreshPriority(priority.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}. Must be one of: critical, high, medium, low")
        
        refresh_service = get_refresh_service()
        success = refresh_service.update_refresh_priority(data_source, new_priority)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Data source '{data_source}' not found")
        
        logger.info(f"Updated refresh priority for {data_source} to {priority}")
        
        return {
            "status": "success",
            "message": f"Priority updated for {data_source}",
            "data_source": data_source,
            "new_priority": priority,
            "updated_at": _now_iso()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update priority for {data_source}: {e}")
        raise HTTPException(status_code=500, detail=f"Priority update failed: {str(e)}")


@router.get("/cached-data/{data_source}")
async def get_cached_data(
    data_source: str,
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get cached data for a specific data source."""
    
    try:
        refresh_service = get_refresh_service()
        cached_data = refresh_service.get_cached_data(data_source)
        
        if cached_data is None:
            raise HTTPException(status_code=404, detail=f"No cached data found for '{data_source}'")
        
        return {
            "status": "success",
            "data_source": data_source,
            "data": cached_data,
            "retrieved_at": _now_iso()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cached data for {data_source}: {e}")
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")


@router.get("/health-summary")
async def get_data_source_health_summary(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get health summary of all data sources."""
    
    try:
        refresh_service = get_refresh_service()
        status = refresh_service.get_refresh_status()
        
        # Aggregate health metrics
        health_summary = {}
        healthy_count = 0
        degraded_count = 0
        failed_count = 0
        
        for job in status["jobs"]:
            data_source = job["data_source"]
            health_status = job["health"]
            
            if health_status == "healthy":
                healthy_count += 1
            elif health_status == "degraded":
                degraded_count += 1
            elif health_status == "failed":
                failed_count += 1
            
            health_summary[data_source] = {
                "status": health_status,
                "last_refresh": job["last_refresh"],
                "refresh_count": job["refresh_count"],
                "error_count": job["error_count"],
                "priority": job["priority"]
            }
        
        # Overall system health
        total_sources = len(health_summary)
        if total_sources == 0:
            overall_health = "unknown"
        elif failed_count > total_sources * 0.5:
            overall_health = "critical"
        elif failed_count + degraded_count > total_sources * 0.3:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        return {
            "status": "success",
            "overall_health": overall_health,
            "health_counts": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "failed": failed_count,
                "total": total_sources
            },
            "data_sources": health_summary,
            "service_running": status["service_status"] == "running",
            "total_subscribers": status["total_subscribers"],
            "generated_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Health summary failed: {str(e)}")


@router.get("/update-history")
async def get_update_history(
    data_source: Optional[str] = Query(None, description="Filter by specific data source"),
    limit: int = Query(20, ge=1, le=100, description="Number of updates to return"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get recent update history for data sources."""
    
    try:
        refresh_service = get_refresh_service()
        status = refresh_service.get_refresh_status()
        
        updates = status["recent_updates"]
        
        # Filter by data source if specified
        if data_source:
            updates = [u for u in updates if u["data_source"] == data_source]
        
        # Apply limit
        updates = updates[-limit:] if len(updates) > limit else updates
        
        # Calculate summary statistics
        total_updates = len(updates)
        changes_detected = len([u for u in updates if u["change_detected"]])
        avg_processing_time = sum(u["processing_time_ms"] for u in updates) / max(1, total_updates)
        
        return {
            "status": "success",
            "updates": updates,
            "summary": {
                "total_updates": total_updates,
                "changes_detected": changes_detected,
                "change_rate": round(changes_detected / max(1, total_updates), 3),
                "avg_processing_time_ms": round(avg_processing_time, 2)
            },
            "filter": {
                "data_source": data_source,
                "limit": limit
            },
            "retrieved_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to get update history: {e}")
        raise HTTPException(status_code=500, detail=f"Update history retrieval failed: {str(e)}")


@router.get("/performance-metrics")
async def get_performance_metrics(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get performance metrics for the refresh service."""
    
    try:
        refresh_service = get_refresh_service()
        status = refresh_service.get_refresh_status()
        
        # Calculate performance metrics
        jobs = status["jobs"]
        recent_updates = status["recent_updates"]
        
        if not jobs:
            return {
                "status": "success",
                "message": "No refresh jobs available",
                "metrics": {}
            }
        
        # Job performance metrics
        total_refreshes = sum(job["refresh_count"] for job in jobs)
        total_errors = sum(job["error_count"] for job in jobs)
        error_rate = total_errors / max(1, total_refreshes + total_errors)
        
        # Priority distribution
        priority_distribution = {}
        for job in jobs:
            priority = job["priority"]
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Processing time metrics
        if recent_updates:
            processing_times = [u["processing_time_ms"] for u in recent_updates]
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
        
        # Data freshness metrics
        current_time = datetime.utcnow()
        freshness_scores = []
        for job in jobs:
            last_refresh = datetime.fromisoformat(job["last_refresh"].replace('Z', ''))
            age_minutes = (current_time - last_refresh).total_seconds() / 60
            
            # Priority-based freshness thresholds
            thresholds = {"critical": 1, "high": 3, "medium": 8, "low": 20}  # minutes
            threshold = thresholds.get(job["priority"], 10)
            
            freshness_score = max(0, 1 - (age_minutes / threshold))
            freshness_scores.append(freshness_score)
        
        avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0
        
        return {
            "status": "success",
            "service_performance": {
                "total_refresh_jobs": len(jobs),
                "total_refreshes_completed": total_refreshes,
                "total_errors": total_errors,
                "error_rate": round(error_rate, 4),
                "service_uptime": status["service_status"] == "running"
            },
            "job_distribution": {
                "by_priority": priority_distribution,
                "by_health": {
                    "healthy": len([j for j in jobs if j["health"] == "healthy"]),
                    "degraded": len([j for j in jobs if j["health"] == "degraded"]), 
                    "failed": len([j for j in jobs if j["health"] == "failed"])
                }
            },
            "processing_performance": {
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "max_processing_time_ms": round(max_processing_time, 2),
                "min_processing_time_ms": round(min_processing_time, 2),
                "recent_updates_analyzed": len(recent_updates)
            },
            "data_freshness": {
                "avg_freshness_score": round(avg_freshness, 3),
                "fresh_data_sources": len([s for s in freshness_scores if s > 0.8]),
                "stale_data_sources": len([s for s in freshness_scores if s < 0.3])
            },
            "subscriber_activity": {
                "total_subscribers": status["total_subscribers"],
                "active_subscriptions": sum(len(job.get("subscribers", [])) for job in jobs if isinstance(job.get("subscribers"), list))
            },
            "generated_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")


@router.post("/subscribe")
async def subscribe_to_data_updates(
    connection_id: str,
    data_sources: List[str],
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Subscribe a connection to real-time data updates."""
    
    try:
        refresh_service = get_refresh_service()
        refresh_service.subscribe_to_updates(connection_id, data_sources)
        
        logger.info(f"Connection {connection_id} subscribed to {len(data_sources)} data sources")
        
        return {
            "status": "success",
            "message": f"Subscribed to {len(data_sources)} data sources",
            "connection_id": connection_id,
            "data_sources": data_sources,
            "subscribed_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to subscribe connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")


@router.post("/unsubscribe/{connection_id}")
async def unsubscribe_from_data_updates(
    connection_id: str,
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Unsubscribe a connection from all data updates."""
    
    try:
        refresh_service = get_refresh_service()
        refresh_service.unsubscribe_from_updates(connection_id)
        
        logger.info(f"Connection {connection_id} unsubscribed from all data sources")
        
        return {
            "status": "success",
            "message": "Unsubscribed from all data sources",
            "connection_id": connection_id,
            "unsubscribed_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to unsubscribe connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unsubscription failed: {str(e)}")


@router.get("/data-sources")
async def list_available_data_sources(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """List all available data sources for real-time updates."""
    
    try:
        refresh_service = get_refresh_service()
        status = refresh_service.get_refresh_status()
        
        data_sources = []
        for job in status["jobs"]:
            data_sources.append({
                "data_source": job["data_source"],
                "job_id": job["job_id"],
                "priority": job["priority"],
                "health": job["health"],
                "refresh_count": job["refresh_count"],
                "last_refresh": job["last_refresh"],
                "description": _get_data_source_description(job["data_source"])
            })
        
        return {
            "status": "success",
            "data_sources": data_sources,
            "total_sources": len(data_sources),
            "generated_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to list data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Data source listing failed: {str(e)}")


def _get_data_source_description(data_source: str) -> str:
    """Get description for a data source."""
    descriptions = {
        "supply_cascade": "Complete supply chain network with nodes, edges, and disruptions",
        "cascade_impacts": "Economic and policy impact analysis from supply chain disruptions",
        "acled": "Real-time geopolitical events and conflicts affecting supply chains",
        "marinetraffic": "Maritime port congestion and shipping disruption data",
        "comtrade": "UN Comtrade bilateral trade statistics and flow data",
        "predictive": "Predictive analytics and disruption forecasting models"
    }
    return descriptions.get(data_source, f"Real-time data updates for {data_source}")


@router.get("/config")
async def get_refresh_configuration(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get current refresh service configuration."""
    
    try:
        refresh_service = get_refresh_service()
        
        # Get configuration details
        config = {
            "refresh_intervals": {
                "critical": "30 seconds",
                "high": "2 minutes", 
                "medium": "5 minutes",
                "low": "15 minutes"
            },
            "backoff_strategy": "Exponential backoff with max 4x multiplier for failed jobs",
            "change_detection": "MD5 hash comparison of JSON-serialized data",
            "concurrent_processing": "Jobs processed in batches by priority level",
            "max_update_history": 100,
            "health_monitoring": "Automatic status tracking with degraded/failed states"
        }
        
        return {
            "status": "success",
            "configuration": config,
            "generated_at": _now_iso()
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")


@router.get("/market-pulse")
async def get_market_pulse() -> Dict[str, Any]:
    """Get real-time market pulse data from live sources"""
    
    try:
        from app.db import SessionLocal
        from app.models import ObservationModel
        from app.services.ingestion import get_latest_observations
        from app.services.geri import compute_geri_score
        
        # Get latest market indicators
        latest_data = get_latest_observations()
        
        # Calculate current GERI score
        if latest_data:
            geri_result = compute_geri_score(latest_data)
            current_geri = geri_result.get("geri_score", 0) if isinstance(geri_result, dict) else geri_result
        else:
            current_geri = 0
        
        # Get data freshness from database
        db = SessionLocal()
        try:
            latest_obs = db.query(ObservationModel).order_by(
                ObservationModel.observed_at.desc()
            ).first()
            
            data_age_minutes = 0
            if latest_obs:
                data_age_minutes = (datetime.utcnow() - latest_obs.observed_at).total_seconds() / 60
                
        finally:
            db.close()
        
        # Determine market status based on GERI and data freshness
        if data_age_minutes > 60:
            market_status = "stale"
            status_message = "Data outdated"
        elif current_geri > 60:
            market_status = "high_risk"
            status_message = "Elevated risk conditions"
        elif current_geri > 40:
            market_status = "moderate_risk" 
            status_message = "Moderate risk levels"
        else:
            market_status = "stable"
            status_message = "Market conditions stable"
        
        return {
            "market_status": market_status,
            "status_message": status_message,
            "current_geri_score": round(float(current_geri), 2),
            "data_freshness": {
                "last_update_minutes_ago": round(data_age_minutes, 1),
                "status": "fresh" if data_age_minutes < 30 else "aging" if data_age_minutes < 120 else "stale"
            },
            "key_indicators": {
                "total_series_monitored": len(latest_data) if latest_data else 0,
                "data_sources": ["FRED", "Alpha Vantage", "EIA", "BLS", "Census", "BEA"],
                "refresh_frequency": "real-time"
            },
            "alerts": [
                {
                    "level": "warning" if current_geri > 50 else "info",
                    "message": "GERI score indicates elevated risk" if current_geri > 50 else "Risk levels normal"
                }
            ] if current_geri > 0 else [],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get market pulse: {e}")
        return {
            "market_status": "unknown",
            "status_message": "Unable to fetch market data",
            "error": str(e),
            "last_updated": datetime.utcnow().isoformat()
        }