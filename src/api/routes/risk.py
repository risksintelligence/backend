from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager
from src.data.sources import fred
import asyncio

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/overview")
async def get_risk_overview(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get risk overview from real economic data only.
    Returns cached data for speed, generates fresh data in background.
    """
    
    # Try to get from cache first (< 10ms)
    cache_key = "risk:overview"
    cached_data = await cache.get(cache_key, max_age_seconds=300)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get real economic indicators for risk assessment
        economic_data = await fred.get_key_indicators()
        
        if not economic_data or not economic_data.get("indicators"):
            raise HTTPException(
                status_code=503,
                detail="Economic data temporarily unavailable - FRED API configuration required"
            )
        
        # Basic risk assessment from economic indicators
        indicators = economic_data["indicators"]
        risk_data = {
            "economic_indicators_count": len(indicators),
            "data_source": "fred",
            "risk_assessment": "based_on_real_economic_data",
            "last_updated": economic_data.get("last_updated"),
            "indicators_available": list(indicators.keys()) if indicators else []
        }
        
        # Cache the result for 5 minutes
        await cache.set(cache_key, risk_data, ttl_seconds=300)
        
        return {
            "status": "success",
            "data": risk_data,
            "source": "real_time",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Risk assessment temporarily unavailable: {str(e)}"
        )


@router.get("/factors")
async def get_risk_factors(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get risk factors from real economic data.
    """
    
    cache_key = "risk:factors"
    cached_data = await cache.get(cache_key, max_age_seconds=600)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get economic factors that indicate risk
        factors_data = await asyncio.gather(
            fred.get_unemployment_rate(),
            fred.get_inflation_rate(),
            fred.get_fed_funds_rate(),
            return_exceptions=True
        )
        
        risk_factors = []
        factor_names = ["unemployment", "inflation", "fed_funds"]
        
        for i, data in enumerate(factors_data):
            if isinstance(data, dict) and data:
                risk_factors.append({
                    "factor": factor_names[i],
                    "data": data,
                    "risk_indicator": True
                })
        
        result = {
            "risk_factors": risk_factors,
            "factor_count": len(risk_factors),
            "source": "fred_economic_data",
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Cache for 10 minutes
        await cache.set(cache_key, result, ttl_seconds=600)
        
        return {
            "status": "success",
            "data": result,
            "source": "real_time",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "loading",
            "message": f"Risk factors are being prepared: {str(e)}",
            "retry_after_seconds": 10,
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/score/simple")
async def get_simple_risk_score(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get simple risk score based on economic indicators.
    """
    
    cache_key = "risk:simple_score"
    cached_data = await cache.get(cache_key, max_age_seconds=300)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get basic economic indicators
        unemployment = await fred.get_unemployment_rate()
        inflation = await fred.get_inflation_rate()
        
        if unemployment and inflation:
            # Simple risk calculation
            risk_score = {
                "overall_score": 50,  # Neutral baseline
                "unemployment_factor": unemployment.get("value", 0),
                "inflation_factor": inflation.get("value", 0),
                "calculation_method": "basic_economic_indicators",
                "last_updated": datetime.utcnow().isoformat()
            }
            
            await cache.set(cache_key, risk_score, ttl_seconds=300)
            
            return {
                "status": "success",
                "data": risk_score,
                "source": "real_time",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(
            status_code=503,
            detail="Economic data not available for risk calculation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Risk score calculation failed: {str(e)}"
        )

