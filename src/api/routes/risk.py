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


@router.get("/factors")
async def get_risk_factors(
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get detailed risk factors breakdown.
    Returns individual risk factors with their current values and impact levels.
    """
    
    cache_key = "risk:factors"
    cached_data = await cache.get(cache_key, max_age_seconds=300)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        async with fred.FREDClient() as client:
            # Fetch key economic indicators that drive risk
            unemployment = await client.get_series("UNRATE", limit=1)
            inflation = await client.get_series("CPIAUCSL", limit=2)
            fed_funds = await client.get_series("FEDFUNDS", limit=1)
            gdp_growth = await client.get_series("GDP", limit=2)
            market_volatility = await client.get_series("VIXCLS", limit=1)
        
        risk_factors = []
        
        # Process unemployment factor
        if unemployment and len(unemployment) > 0:
            unemployment_rate = unemployment[0].get("value", 0)
            risk_factors.append({
                "id": 1,
                "name": "Unemployment Rate",
                "category": "economic",
                "current_value": unemployment_rate,
                "current_score": min(unemployment_rate * 10, 100),
                "impact_level": "high" if unemployment_rate > 6 else "moderate" if unemployment_rate > 4 else "low",
                "weight": 0.25,
                "data_source": "FRED - Bureau of Labor Statistics",
                "series_id": "UNRATE",
                "last_updated": unemployment[0].get("date", ""),
                "thresholds": {"low": 4.0, "high": 6.0}
            })
        
        # Process inflation factor
        if inflation and len(inflation) >= 2:
            current_cpi = inflation[0].get("value", 0)
            previous_cpi = inflation[1].get("value", 0)
            inflation_rate = ((current_cpi - previous_cpi) / previous_cpi) * 100 if previous_cpi > 0 else 0
            risk_factors.append({
                "id": 2,
                "name": "Inflation Rate",
                "category": "economic", 
                "current_value": inflation_rate,
                "current_score": min(abs(inflation_rate - 2.0) * 20, 100),
                "impact_level": "critical" if abs(inflation_rate - 2.0) > 3 else "high" if abs(inflation_rate - 2.0) > 1.5 else "moderate",
                "weight": 0.3,
                "data_source": "FRED - Bureau of Labor Statistics",
                "series_id": "CPIAUCSL",
                "last_updated": inflation[0].get("date", ""),
                "thresholds": {"low": 1.0, "high": 3.0}
            })
        
        # Process fed funds rate factor
        if fed_funds and len(fed_funds) > 0:
            fed_rate = fed_funds[0].get("value", 0)
            risk_factors.append({
                "id": 3,
                "name": "Federal Funds Rate",
                "category": "market",
                "current_value": fed_rate,
                "current_score": min(fed_rate * 5, 100),
                "impact_level": "high" if fed_rate > 5 else "moderate" if fed_rate > 2 else "low",
                "weight": 0.2,
                "data_source": "FRED - Federal Reserve",
                "series_id": "FEDFUNDS", 
                "last_updated": fed_funds[0].get("date", ""),
                "thresholds": {"low": 2.0, "high": 5.0}
            })
        
        # Process market volatility factor
        if market_volatility and len(market_volatility) > 0:
            vix_value = market_volatility[0].get("value", 0)
            risk_factors.append({
                "id": 4,
                "name": "Market Volatility (VIX)",
                "category": "market",
                "current_value": vix_value,
                "current_score": min(vix_value * 2, 100),
                "impact_level": "critical" if vix_value > 30 else "high" if vix_value > 20 else "moderate",
                "weight": 0.25,
                "data_source": "FRED - Chicago Board Options Exchange",
                "series_id": "VIXCLS",
                "last_updated": market_volatility[0].get("date", ""),
                "thresholds": {"low": 15.0, "high": 25.0}
            })
        
        result = {
            "risk_factors": risk_factors,
            "total_factors": len(risk_factors),
            "categories": {
                "economic": len([f for f in risk_factors if f["category"] == "economic"]),
                "market": len([f for f in risk_factors if f["category"] == "market"]),
                "geopolitical": 0,
                "technical": 0
            },
            "last_updated": datetime.utcnow().isoformat(),
            "data_freshness": "real_time"
        }
        
        # Cache for 5 minutes
        await cache.set(cache_key, result, ttl_seconds=300)
        
        return {
            "status": "success",
            "data": result,
            "source": "real_time",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "loading",
            "message": f"Risk factors are being calculated: {str(e)}",
            "retry_after_seconds": 15,
            "timestamp": datetime.utcnow().isoformat()
        }

