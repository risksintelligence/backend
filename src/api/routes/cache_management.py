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
    """Manually trigger cache warming with real data from external APIs."""
    
    # Load real data from external sources to warm cache
    from src.data.sources import fred, bea, bls
    
    warmed_keys = []
    failed_keys = []
    
    try:
        # Get real GDP data
        gdp_data = await fred.get_gdp()
        if gdp_data:
            await cache.set("fred:GDP", gdp_data, ttl_seconds=3600)
            warmed_keys.append("fred:GDP")
        else:
            failed_keys.append("fred:GDP")
        
        # Get real unemployment data
        unemployment_data = await fred.get_unemployment_rate()
        if unemployment_data:
            await cache.set("fred:UNRATE", unemployment_data, ttl_seconds=3600)
            warmed_keys.append("fred:UNRATE")
        else:
            failed_keys.append("fred:UNRATE")
        
        # Get real inflation data
        inflation_data = await fred.get_inflation_rate()
        if inflation_data:
            await cache.set("fred:CPIAUCSL", inflation_data, ttl_seconds=3600)
            warmed_keys.append("fred:CPIAUCSL")
        else:
            failed_keys.append("fred:CPIAUCSL")
        
        # Generate risk overview from real data only if components available
        if len(warmed_keys) >= 2:
            from src.ml.serving.model_server import ModelServer
            model_server = ModelServer()
            try:
                risk_overview = await model_server.get_comprehensive_risk_assessment()
                if risk_overview:
                    await cache.set("risk:overview", risk_overview, ttl_seconds=300)
                    warmed_keys.append("risk:overview")
                else:
                    failed_keys.append("risk:overview")
            except Exception as e:
                logger.error(f"Failed to generate risk overview: {e}")
                failed_keys.append("risk:overview")
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        return {
            "status": "error",
            "message": f"Cache warming failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "success" if warmed_keys else "partial",
        "message": f"Cache warmed with {len(warmed_keys)} real data keys",
        "warmed_keys": warmed_keys,
        "failed_keys": failed_keys,
        "note": "Only real external API data used for cache warming",
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