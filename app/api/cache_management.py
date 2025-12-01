"""
Cache Management API

Provides endpoints for monitoring, managing, and optimizing the supply chain
caching system with real-time statistics and administrative controls.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.core.security import require_system_rate_limit
from app.core.supply_chain_cache import get_supply_chain_cache, SupplyChainCache
from app.core.unified_cache import UnifiedCache

router = APIRouter(prefix="/api/v1/cache", tags=["cache-management"])


@router.get("/status")
async def get_cache_status(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get comprehensive cache system status and health metrics."""
    try:
        sc_cache = get_supply_chain_cache()
        stats = sc_cache.get_cache_stats()
        
        # Get stale keys statistics
        stale_keys = sc_cache.get_stale_keys()
        stale_by_type = {}
        for key in stale_keys:
            data_type = key.split(":")[0] if ":" in key else "unknown"
            stale_by_type[data_type] = stale_by_type.get(data_type, 0) + 1
        
        return {
            "cache_status": {
                "redis_status": stats.get("status", "unknown"),
                "overall_health": "healthy" if stats.get("status") == "healthy" else "degraded",
                "hit_rate_percentage": round(stats.get("hit_rate", 0), 2),
                "total_keys": stats.get("namespace_stats", {}).get("total_keys", 0),
                "stale_keys_count": len(stale_keys),
                "stale_keys_by_type": stale_by_type,
                "last_check": datetime.utcnow().isoformat()
            },
            "redis_metrics": stats.get("redis_info", {}),
            "namespace_stats": stats.get("namespace_stats", {}),
            "policy_usage": stats.get("namespace_stats", {}).get("policy_usage", {}),
            "recommendations": sc_cache.optimize_cache().get("optimization_suggestions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")


@router.get("/policies")
async def get_cache_policies(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get all cache policies and their configurations."""
    try:
        sc_cache = get_supply_chain_cache()
        policies = {}
        
        for data_type, policy in sc_cache.CACHE_POLICIES.items():
            policies[data_type] = {
                "soft_ttl_seconds": policy.soft_ttl,
                "hard_ttl_seconds": policy.hard_ttl,
                "soft_ttl_human": f"{policy.soft_ttl // 60} minutes" if policy.soft_ttl < 3600 else f"{policy.soft_ttl // 3600} hours",
                "hard_ttl_human": f"{policy.hard_ttl // 60} minutes" if policy.hard_ttl < 3600 else f"{policy.hard_ttl // 3600} hours",
                "compression": policy.compression,
                "encryption": policy.encryption,
                "priority": policy.priority
            }
        
        return {
            "cache_policies": policies,
            "total_policies": len(policies),
            "policy_categories": {
                "real_time": [k for k, v in sc_cache.CACHE_POLICIES.items() if v.soft_ttl <= 600],
                "moderate_frequency": [k for k, v in sc_cache.CACHE_POLICIES.items() if 600 < v.soft_ttl <= 14400],
                "low_frequency": [k for k, v in sc_cache.CACHE_POLICIES.items() if v.soft_ttl > 14400]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache policies: {str(e)}")


@router.get("/stale-keys")
async def get_stale_keys(
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of keys to return"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get list of stale cache keys that need refresh."""
    try:
        sc_cache = get_supply_chain_cache()
        stale_keys = sc_cache.get_stale_keys(data_type)
        
        # Limit results and add metadata
        limited_keys = stale_keys[:limit]
        
        # Group by data type
        by_type = {}
        for key in limited_keys:
            key_type = key.split(":")[0] if ":" in key else "unknown"
            if key_type not in by_type:
                by_type[key_type] = []
            by_type[key_type].append(key)
        
        return {
            "stale_keys": {
                "total_found": len(stale_keys),
                "returned": len(limited_keys),
                "filtered_by_type": data_type,
                "keys": limited_keys,
                "grouped_by_type": by_type,
                "refresh_needed": len(stale_keys) > 0
            },
            "recommendations": [
                "Schedule background refresh for stale keys",
                "Consider increasing soft_ttl for frequently stale data types"
            ] if len(stale_keys) > 10 else [],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stale keys: {str(e)}")


@router.post("/invalidate/{data_type}")
async def invalidate_cache(
    data_type: str = Path(..., description="Data type to invalidate"),
    identifier: str = Query("*", description="Specific identifier or * for all"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Invalidate cache for a specific data type or identifier."""
    try:
        sc_cache = get_supply_chain_cache()
        
        # Validate data type
        if data_type not in sc_cache.CACHE_POLICIES and data_type != "all":
            available_types = list(sc_cache.CACHE_POLICIES.keys()) + ["all"]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data type '{data_type}'. Available types: {available_types}"
            )
        
        if data_type == "all":
            # Invalidate all cache
            from app.core.supply_chain_cache import invalidate_all_cache
            keys_removed = invalidate_all_cache()
            message = "All supply chain cache invalidated"
        else:
            # Invalidate specific data type
            keys_removed = sc_cache.invalidate(data_type, identifier)
            if identifier == "*":
                message = f"All {data_type} cache invalidated"
            else:
                message = f"Cache for {data_type}:{identifier} invalidated"
        
        return {
            "invalidation_result": {
                "data_type": data_type,
                "identifier": identifier,
                "keys_removed": keys_removed,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")


@router.post("/optimize")
async def optimize_cache(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Perform cache optimization and cleanup operations."""
    try:
        sc_cache = get_supply_chain_cache()
        optimization_result = sc_cache.optimize_cache()
        
        return {
            "optimization": optimization_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize cache: {str(e)}")


@router.get("/key/{data_type}/{identifier}")
async def get_cache_key_info(
    data_type: str = Path(..., description="Data type"),
    identifier: str = Path(..., description="Key identifier"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get detailed information about a specific cache key."""
    try:
        sc_cache = get_supply_chain_cache()
        
        # Validate data type
        if data_type not in sc_cache.CACHE_POLICIES:
            raise HTTPException(status_code=404, detail=f"Unknown data type: {data_type}")
        
        # Get cached data and metadata
        data, metadata = sc_cache.get(data_type, identifier)
        
        if data is None:
            return {
                "key_info": {
                    "data_type": data_type,
                    "identifier": identifier,
                    "exists": False,
                    "message": "Key not found in cache"
                }
            }
        
        # Get policy for this data type
        policy = sc_cache.get_policy(data_type)
        
        return {
            "key_info": {
                "data_type": data_type,
                "identifier": identifier,
                "exists": True,
                "policy": {
                    "soft_ttl": policy.soft_ttl,
                    "hard_ttl": policy.hard_ttl,
                    "priority": policy.priority
                },
                "metadata": metadata.__dict__ if metadata else None,
                "data_preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data),
                "data_size_bytes": len(str(data).encode('utf-8')),
                "is_stale": metadata.is_stale_soft if metadata else False,
                "cache_hit": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get key info: {str(e)}")


@router.get("/analytics")
async def get_cache_analytics(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get cache analytics and performance insights."""
    try:
        sc_cache = get_supply_chain_cache()
        stats = sc_cache.get_cache_stats()
        
        # Calculate additional analytics
        hit_rate = stats.get("hit_rate", 0)
        stale_keys_count = len(sc_cache.get_stale_keys())
        total_keys = stats.get("namespace_stats", {}).get("total_keys", 0)
        
        # Performance assessment
        performance_grade = "A" if hit_rate >= 90 else "B" if hit_rate >= 75 else "C" if hit_rate >= 60 else "D"
        staleness_ratio = (stale_keys_count / total_keys * 100) if total_keys > 0 else 0
        
        # Generate insights
        insights = []
        if hit_rate < 70:
            insights.append("Low cache hit rate detected - consider increasing TTL values")
        if staleness_ratio > 20:
            insights.append("High staleness ratio - background refresh may be insufficient")
        if total_keys > 10000:
            insights.append("Large cache size detected - consider implementing cache partitioning")
        if not insights:
            insights.append("Cache performance is optimal")
        
        return {
            "analytics": {
                "performance_grade": performance_grade,
                "hit_rate_percentage": round(hit_rate, 2),
                "staleness_ratio_percentage": round(staleness_ratio, 2),
                "total_keys": total_keys,
                "stale_keys": stale_keys_count,
                "analysis_period_hours": hours,
                "redis_status": stats.get("status", "unknown")
            },
            "insights": insights,
            "recommendations": [
                "Monitor hit rates and adjust TTL policies accordingly",
                "Implement automated cache warming for critical data",
                "Consider Redis clustering for high-availability production deployments"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache analytics: {str(e)}")


@router.get("/health")
async def cache_health_check(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Simple health check for cache system."""
    try:
        sc_cache = get_supply_chain_cache()
        
        # Test basic cache operations
        test_key = "health_check"
        test_data = {"timestamp": datetime.utcnow().isoformat(), "test": True}
        
        # Test write
        sc_cache.set("test_data", test_key, test_data, "health_check")
        
        # Test read
        cached_data, metadata = sc_cache.get("test_data", test_key)
        
        # Verify data
        is_healthy = cached_data and cached_data.get("test") is True
        
        # Clean up test data
        sc_cache.invalidate("test_data", test_key)
        
        return {
            "health": {
                "status": "healthy" if is_healthy else "unhealthy",
                "redis_available": sc_cache.redis.available,
                "read_test": "passed" if cached_data else "failed",
                "write_test": "passed" if is_healthy else "failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        return {
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        }