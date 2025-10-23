from fastapi import APIRouter, Depends
from datetime import datetime
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager

router = APIRouter(prefix="/api/v1/cache", tags=["cache_management"])


@router.get("/metrics")
async def get_cache_metrics(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Get cache performance metrics."""
    
    metrics = cache.get_metrics()
    
    return {
        "status": "success",
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/keys")
async def list_cache_keys(
    pattern: str = "*",
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """List all cache keys matching pattern."""
    
    keys = await cache.keys(pattern)
    
    return {
        "status": "success",
        "pattern": pattern,
        "keys": keys,
        "count": len(keys),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/status")
async def get_cache_status(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Get detailed cache system status."""
    
    # Test all cache tiers
    test_key = "cache:health_test"
    test_data = {
        "test": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Test cache operations
    try:
        await cache.set(test_key, test_data, ttl_seconds=60)
        retrieved_data = await cache.get(test_key)
        await cache.delete(test_key)
        
        cache_operational = retrieved_data is not None
    except Exception as e:
        cache_operational = False
    
    # Get metrics
    metrics = cache.get_metrics()
    
    return {
        "status": "success",
        "cache_operational": cache_operational,
        "tiers": {
            "redis": cache.redis_client is not None,
            "postgres": cache.db_session is not None,
            "file_system": True
        },
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/warm")
async def warm_cache(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Manually trigger cache warming with sample data."""
    
    # Sample data to warm the cache
    warm_data = [
        {
            "key": "risk:overview",
            "data": {
                "overall_score": 75.5,
                "factors": {
                    "economic": 78.2,
                    "market": 72.1,
                    "geopolitical": 68.9,
                    "technical": 81.3
                },
                "trend": "stable",
                "confidence": 0.87,
                "last_updated": datetime.utcnow().isoformat()
            }
        },
        {
            "key": "fred:GDP",
            "data": {
                "value": 27000000,
                "units": "millions_of_dollars",
                "frequency": "quarterly",
                "last_updated": datetime.utcnow().isoformat()
            }
        },
        {
            "key": "fred:UNRATE",
            "data": {
                "value": 3.7,
                "units": "percent",
                "frequency": "monthly",
                "last_updated": datetime.utcnow().isoformat()
            }
        },
        {
            "key": "market:VIX",
            "data": {
                "value": 18.5,
                "units": "volatility_index",
                "frequency": "realtime",
                "last_updated": datetime.utcnow().isoformat()
            }
        }
    ]
    
    warmed_keys = []
    
    for item in warm_data:
        try:
            await cache.set(item["key"], item["data"], ttl_seconds=3600)
            warmed_keys.append(item["key"])
        except Exception as e:
            # Continue warming other keys even if one fails
            pass
    
    return {
        "status": "success",
        "message": "Cache warming completed",
        "warmed_keys": warmed_keys,
        "count": len(warmed_keys),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.delete("/clear")
async def clear_cache(
    pattern: str = "*",
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Clear cache keys matching pattern."""
    
    keys_to_delete = await cache.keys(pattern)
    deleted_count = 0
    
    for key in keys_to_delete:
        try:
            await cache.delete(key)
            deleted_count += 1
        except Exception:
            # Continue deleting other keys
            pass
    
    return {
        "status": "success",
        "message": f"Cleared {deleted_count} cache keys",
        "pattern": pattern,
        "deleted_count": deleted_count,
        "timestamp": datetime.utcnow().isoformat()
    }