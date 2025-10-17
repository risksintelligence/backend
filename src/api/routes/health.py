"""
Health check endpoints with comprehensive system validation.
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.database import get_db, check_db_connection
from src.cache.cache_manager import CacheManager
from src.data.sources.fred import FREDConnector


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    environment: str
    version: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    overall_status: str
    timestamp: datetime
    environment: str
    version: str
    components: Dict[str, Any]
    performance_metrics: Dict[str, Any]


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Current health status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        environment=settings.environment,
        version="1.0.0",
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check with all system components.
    
    Args:
        db: Database session
        
    Returns:
        DetailedHealthResponse: Detailed system health status
    """
    start_time = datetime.utcnow()
    components = {}
    overall_healthy = True
    
    # Database health check
    try:
        db_healthy = check_db_connection()
        if db_healthy:
            # Test a simple query
            db.execute("SELECT 1")
            components["database"] = {
                "status": "healthy",
                "details": "Connection successful, queries executing",
                "response_time_ms": 0  # Will be updated
            }
        else:
            components["database"] = {
                "status": "unhealthy",
                "details": "Connection failed",
                "response_time_ms": None
            }
            overall_healthy = False
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "details": f"Database error: {str(e)}",
            "response_time_ms": None
        }
        overall_healthy = False
    
    # Cache health check
    try:
        cache_manager = CacheManager()
        cache_health = cache_manager.health_check()
        
        components["cache"] = {
            "status": "healthy" if cache_health["overall_healthy"] else "degraded",
            "details": cache_health,
            "redis_available": cache_health["redis"]["healthy"],
            "postgres_cache_available": cache_health["postgres"]["healthy"],
            "degraded_mode": cache_health["degraded_mode"]
        }
        
        if not cache_health["overall_healthy"]:
            overall_healthy = False
            
    except Exception as e:
        components["cache"] = {
            "status": "unhealthy",
            "details": f"Cache error: {str(e)}"
        }
        overall_healthy = False
    
    # Data sources health check
    try:
        fred_connector = FREDConnector()
        fred_health = fred_connector.health_check()
        
        components["data_sources"] = {
            "fred": {
                "status": "healthy" if fred_health["overall_healthy"] else "degraded",
                "api_available": fred_health["api_available"],
                "cache_available": fred_health["cache_available"],
                "fallback_available": fred_health["fallback_available"],
                "details": fred_health
            }
        }
        
        if not fred_health["overall_healthy"]:
            # Data sources being degraded doesn't make the system unhealthy
            # as long as fallbacks are available
            pass
            
    except Exception as e:
        components["data_sources"] = {
            "fred": {
                "status": "unhealthy",
                "details": f"FRED connector error: {str(e)}"
            }
        }
    
    # Calculate performance metrics
    end_time = datetime.utcnow()
    total_check_time = (end_time - start_time).total_seconds() * 1000
    
    performance_metrics = {
        "total_check_time_ms": round(total_check_time, 2),
        "checks_completed": len(components),
        "timestamp": end_time.isoformat()
    }
    
    # Determine overall status
    if overall_healthy:
        overall_status = "healthy"
    else:
        # Check if we can still operate in degraded mode
        has_database = components.get("database", {}).get("status") == "healthy"
        has_some_cache = (
            components.get("cache", {}).get("redis_available", False) or
            components.get("cache", {}).get("postgres_cache_available", False)
        )
        
        if has_database and has_some_cache:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
    
    return DetailedHealthResponse(
        overall_status=overall_status,
        timestamp=end_time,
        environment=settings.environment,
        version="1.0.0",
        components=components,
        performance_metrics=performance_metrics
    )


@router.get("/health/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check - ensures system is ready to serve requests.
    
    Args:
        db: Database session
        
    Returns:
        dict: Readiness status
    """
    try:
        # Check database connection
        db_healthy = check_db_connection()
        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # Test database query
        db.execute("SELECT 1")
        
        # Check cache availability
        cache_manager = CacheManager()
        cache_health = cache_manager.health_check()
        
        # System is ready if database works and at least one cache layer is available
        ready = db_healthy and cache_health["overall_healthy"]
        
        if not ready:
            raise HTTPException(status_code=503, detail="System not ready")
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow(),
            "database": "healthy",
            "cache": "healthy" if cache_health["overall_healthy"] else "degraded"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - ensures application is alive and responding.
    
    Returns:
        dict: Liveness status
    """
    return {
        "status": "alive", 
        "timestamp": datetime.utcnow(),
        "uptime_check": "passed"
    }


@router.get("/health/cache")
async def cache_health_check():
    """
    Detailed cache health and performance check.
    
    Returns:
        dict: Cache system health and statistics
    """
    try:
        cache_manager = CacheManager()
        
        # Get comprehensive cache statistics
        cache_stats = cache_manager.get_cache_stats()
        cache_health = cache_manager.health_check()
        
        return {
            "status": "healthy" if cache_health["overall_healthy"] else "degraded",
            "timestamp": datetime.utcnow(),
            "health": cache_health,
            "statistics": cache_stats,
            "recommendations": _generate_cache_recommendations(cache_stats, cache_health)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }


@router.get("/health/data-sources")
async def data_sources_health_check():
    """
    Health check for all external data sources.
    
    Returns:
        dict: Data sources health status
    """
    try:
        results = {}
        
        # Check FRED data source
        fred_connector = FREDConnector()
        fred_health = fred_connector.health_check()
        results["fred"] = fred_health
        
        # Calculate overall data sources health
        total_sources = len(results)
        healthy_sources = sum(1 for source in results.values() if source.get("overall_healthy", False))
        
        overall_status = "healthy" if healthy_sources == total_sources else "degraded"
        if healthy_sources == 0:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow(),
            "summary": {
                "total_sources": total_sources,
                "healthy_sources": healthy_sources,
                "degraded_sources": total_sources - healthy_sources,
                "health_percentage": round((healthy_sources / total_sources) * 100, 1)
            },
            "sources": results
        }
        
    except Exception as e:
        return {
            "overall_status": "error",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }


def _generate_cache_recommendations(stats: Dict[str, Any], health: Dict[str, Any]) -> list:
    """Generate cache optimization recommendations."""
    recommendations = []
    
    # Check Redis stats
    redis_stats = stats.get("redis", {})
    if redis_stats.get("status") == "connected":
        keyspace_hits = redis_stats.get("keyspace_hits", 0)
        keyspace_misses = redis_stats.get("keyspace_misses", 0)
        
        if keyspace_hits + keyspace_misses > 0:
            hit_rate = keyspace_hits / (keyspace_hits + keyspace_misses)
            if hit_rate < 0.8:
                recommendations.append("Redis hit rate is below 80%. Consider increasing TTL or cache size.")
    
    # Check PostgreSQL cache stats
    postgres_stats = stats.get("postgres", {})
    if postgres_stats.get("status") == "connected":
        expired_entries = postgres_stats.get("expired_entries", 0)
        total_entries = postgres_stats.get("total_entries", 0)
        
        if total_entries > 0 and expired_entries / total_entries > 0.3:
            recommendations.append("High number of expired entries in PostgreSQL cache. Consider cleanup.")
    
    # Check degraded mode
    if health.get("degraded_mode", False):
        recommendations.append("System running in degraded mode. Redis cache unavailable.")
    
    if not recommendations:
        recommendations.append("Cache system operating optimally.")
    
    return recommendations