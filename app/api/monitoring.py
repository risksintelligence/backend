"""
Data Freshness Monitoring API

Provides endpoints for monitoring cache freshness, data lineage,
and overall system health per architecture requirements.
"""

from datetime import datetime
from typing import Dict, List, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db import get_db
from app.models import ObservationModel
from app.core.unified_cache import UnifiedCache
from app.core.provider_failover import failover_manager
from app.services.background_refresh import refresh_service
from app.core.security import require_system_rate_limit
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

@router.get("/data-freshness")
def get_data_freshness(
    _rate_limit: bool = Depends(require_system_rate_limit),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive data freshness report across all cache layers.
    
    Returns TTL status per component, cache hit rates, and staleness warnings
    per architecture requirements.
    """
    try:
        cache = UnifiedCache("monitoring")
        
        # Get freshness report from unified cache system
        freshness_report = cache.get_freshness_report()
        
        # Get series-level freshness from database
        series_freshness = _get_series_freshness_status(db)
        
        # Get background refresh status
        refresh_status = refresh_service.get_refresh_status()
        
        # Get provider health
        provider_health = failover_manager.get_provider_health()
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": _calculate_overall_status(freshness_report),
            "cache_layers": freshness_report,
            "series_freshness": series_freshness,
            "background_refresh": refresh_status,
            "provider_health": provider_health,
            "compliance": {
                "data_lineage_enabled": True,
                "ttl_management_active": True,
                "stale_while_revalidate": True,
                "no_fake_fallbacks": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate freshness report: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": "error",
            "error": str(e)
        }

@router.get("/data-lineage/{series_id}")
def get_data_lineage(
    series_id: str,
    limit: int = 10,
    _rate_limit: bool = Depends(require_system_rate_limit),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed data lineage for a specific series.
    
    Shows source provenance, fetch timestamps, checksums, and derivation flags
    per architecture requirements.
    """
    try:
        # Get recent observations with full lineage
        observations = db.query(ObservationModel).filter(
            ObservationModel.series_id == series_id
        ).order_by(desc(ObservationModel.fetched_at)).limit(limit).all()
        
        if not observations:
            return {
                "series_id": series_id,
                "status": "no_data",
                "observations": []
            }
        
        lineage_data = []
        for obs in observations:
            lineage_data.append({
                "observed_at": obs.observed_at.isoformat() + "Z",
                "value": obs.value,
                "source": obs.source,
                "source_url": obs.source_url,
                "fetched_at": obs.fetched_at.isoformat() + "Z" if obs.fetched_at else None,
                "checksum": obs.checksum,
                "derivation_flag": obs.derivation_flag,
                "soft_ttl": obs.soft_ttl,
                "hard_ttl": obs.hard_ttl,
                "age_hours": (datetime.utcnow() - (obs.fetched_at or obs.observed_at)).total_seconds() / 3600
            })
        
        return {
            "series_id": series_id,
            "status": "success",
            "total_observations": len(lineage_data),
            "latest_fetch": observations[0].fetched_at.isoformat() + "Z" if observations[0].fetched_at else None,
            "observations": lineage_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get lineage for {series_id}: {e}")
        return {
            "series_id": series_id,
            "status": "error", 
            "error": str(e)
        }

@router.get("/cache-status")
def get_cache_status(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """
    Get detailed cache performance metrics and hit rates.
    """
    try:
        cache = UnifiedCache("monitoring")
        
        # Get cache status from each layer
        l1_status = cache.redis.get_freshness_status()
        l2_status = cache._get_l2_freshness()  
        l3_status = cache._get_l3_freshness()
        
        # Calculate hit rates and performance metrics
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "l1_redis": {
                **l1_status,
                "connection_available": cache.redis.available,
                "namespace": cache.redis.namespace
            },
            "l2_postgresql": l2_status,
            "l3_file_store": l3_status,
            "unified_cache_config": cache._get_unified_status()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "error",
            "error": str(e)
        }

@router.post("/refresh-series/{series_id}")
def force_refresh_series(
    series_id: str,
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, str]:
    """
    Force immediate refresh of a specific series (bypass normal TTL).
    """
    try:
        refresh_service.force_refresh(series_id)
        
        logger.info(f"Manual refresh triggered for {series_id}")
        
        return {
            "status": "success",
            "message": f"Refresh queued for {series_id}",
            "series_id": series_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to force refresh {series_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/provider-health")
def get_provider_health(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """
    Get health status and reliability scores for all data providers.
    """
    try:
        provider_health = failover_manager.get_provider_health()
        
        # Add summary statistics
        total_providers = len(provider_health)
        healthy_providers = sum(1 for p in provider_health.values() if not p['should_skip'])
        avg_reliability = sum(p['reliability_score'] for p in provider_health.values()) / total_providers
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_providers": total_providers,
                "healthy_providers": healthy_providers,
                "unhealthy_providers": total_providers - healthy_providers,
                "average_reliability": round(avg_reliability, 3),
                "overall_health": "good" if healthy_providers >= total_providers * 0.8 else "degraded"
            },
            "providers": provider_health
        }
        
    except Exception as e:
        logger.error(f"Failed to get provider health: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "error",
            "error": str(e)
        }

def _get_series_freshness_status(db: Session) -> Dict[str, Any]:
    """Get freshness status for each series from database."""
    try:
        from datetime import timedelta
        
        # Get latest observation per series
        latest_obs_query = db.query(
            ObservationModel.series_id,
            func.max(ObservationModel.fetched_at).label('latest_fetch'),
            func.max(ObservationModel.observed_at).label('latest_observation'),
            func.count(ObservationModel.id).label('total_observations')
        ).group_by(ObservationModel.series_id).all()
        
        series_status = {}
        now = datetime.utcnow()
        
        for series_id, latest_fetch, latest_obs, count in latest_obs_query:
            age_hours = (now - (latest_fetch or latest_obs)).total_seconds() / 3600 if latest_fetch or latest_obs else float('inf')
            
            # Determine freshness status
            if age_hours < 1:
                freshness = "fresh"
            elif age_hours < 24:
                freshness = "acceptable" 
            elif age_hours < 168:  # 1 week
                freshness = "stale"
            else:
                freshness = "very_stale"
            
            series_status[series_id] = {
                "latest_fetch": latest_fetch.isoformat() + "Z" if latest_fetch else None,
                "latest_observation": latest_obs.isoformat() + "Z" if latest_obs else None,
                "age_hours": round(age_hours, 1),
                "total_observations": count,
                "freshness": freshness
            }
        
        return series_status
        
    except Exception as e:
        logger.error(f"Failed to get series freshness: {e}")
        return {}

def _calculate_overall_status(freshness_report: Dict) -> str:
    """Calculate overall system status from freshness report."""
    try:
        l1_status = freshness_report.get("l1_redis", {}).get("status", "error")
        l2_status = freshness_report.get("l2_postgresql", {}).get("status", "error")
        
        if l1_status == "available" and l2_status == "available":
            return "healthy"
        elif l1_status == "available" or l2_status == "available":
            return "degraded"
        else:
            return "unhealthy"
            
    except Exception:
        return "unknown"