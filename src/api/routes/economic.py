from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager

router = APIRouter(prefix="/api/v1/economic", tags=["economic"])


@router.get("/indicators")
async def get_economic_indicators(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get all economic indicators - INSTANT response from cache.
    """
    
    # Get multiple indicators from cache
    indicators = {}
    
    indicator_keys = [
        ("fred:GDP", "gdp"),
        ("fred:UNRATE", "unemployment"),
        ("fred:CPIAUCSL", "inflation"),
        ("fred:FEDFUNDS", "fed_funds_rate")
    ]
    
    for cache_key, indicator_name in indicator_keys:
        data = await cache.get(cache_key, max_age_seconds=900)  # 15 min max age
        if data:
            indicators[indicator_name] = data
    
    if indicators:
        return {
            "status": "success",
            "data": indicators,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "loading",
        "message": "Economic indicators are being prepared.",
        "retry_after_seconds": 10,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/indicators/{series_id}")
async def get_economic_indicator(
    series_id: str,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get specific economic indicator - INSTANT from cache.
    """
    
    # Map common series IDs
    series_mapping = {
        "gdp": "fred:GDP",
        "unemployment": "fred:UNRATE", 
        "inflation": "fred:CPIAUCSL",
        "fed_rate": "fred:FEDFUNDS"
    }
    
    cache_key = series_mapping.get(series_id, f"fred:{series_id.upper()}")
    data = await cache.get(cache_key, max_age_seconds=600)
    
    if data:
        return {
            "status": "success",
            "series_id": series_id,
            "data": data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "not_found",
        "message": f"Indicator {series_id} not available or still loading",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/market")
async def get_market_indicators(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get market indicators - INSTANT response from cache.
    """
    
    indicators = {}
    
    market_keys = [
        ("market:VIX", "volatility_index"),
        ("market:SP500", "sp500_index")
    ]
    
    for cache_key, indicator_name in market_keys:
        data = await cache.get(cache_key, max_age_seconds=600)  # 10 min max age
        if data:
            indicators[indicator_name] = data
    
    if indicators:
        return {
            "status": "success",
            "data": indicators,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "loading",
        "message": "Market indicators are being prepared.",
        "retry_after_seconds": 10,
        "timestamp": datetime.utcnow().isoformat()
    }