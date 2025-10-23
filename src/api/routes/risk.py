from fastapi import APIRouter, Depends
from datetime import datetime
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/overview")
async def get_risk_overview(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get risk overview - INSTANT response from cache.
    Zero external API calls.
    """
    
    # Try to get from cache (< 10ms)
    cache_key = "risk:overview"
    cached_data = await cache.get(cache_key, max_age_seconds=300)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # If cache completely empty (first request ever)
    # Return loading state
    return {
        "status": "loading",
        "message": "Data is being prepared. Please try again in a moment.",
        "retry_after_seconds": 5,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/factors")
async def get_risk_factors(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get risk factors - INSTANT from cache.
    """
    
    cache_key = "risk:factors"
    data = await cache.get(cache_key, max_age_seconds=600)
    
    if data:
        return {
            "status": "success",
            "data": data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "loading",
        "message": "Risk factors are being prepared.",
        "retry_after_seconds": 5,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/score/realtime")
async def get_realtime_risk_score(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get real-time risk score - served from cache updated every 5 min.
    """
    
    cache_key = "risk:overview"
    data = await cache.get(cache_key, max_age_seconds=300)
    
    if data:
        return {
            "status": "success",
            "score": data.get("overall_score"),
            "components": data.get("factors"),
            "confidence": data.get("confidence"),
            "trend": data.get("trend"),
            "last_updated": data.get("last_updated"),
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Fallback to last known good data
    fallback_data = await cache.get(cache_key, max_age_seconds=None)
    if fallback_data:
        return {
            "status": "success",
            "score": fallback_data.get("overall_score"),
            "components": fallback_data.get("factors"),
            "confidence": fallback_data.get("confidence"),
            "trend": fallback_data.get("trend"),
            "last_updated": fallback_data.get("last_updated"),
            "source": "cache_fallback",
            "note": "Serving last known good data",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "unavailable",
        "message": "Risk score temporarily unavailable",
        "timestamp": datetime.utcnow().isoformat()
    }