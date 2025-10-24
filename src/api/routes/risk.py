from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from typing import Dict, Any, Optional
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager
from src.ml.serving.model_server import get_model_server, ModelServer

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/overview")
async def get_risk_overview(
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Get comprehensive risk overview from all financial models.
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
        # Generate fresh comprehensive risk assessment
        assessment = await model_server.get_comprehensive_risk_assessment()
        
        # Cache the result for 5 minutes
        await cache.set(cache_key, assessment, ttl_seconds=300)
        
        return {
            "status": "success",
            "data": assessment,
            "source": "real_time",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # If models fail, return cached data if available (even if old)
        fallback_data = await cache.get(cache_key, max_age_seconds=3600)  # 1 hour fallback
        
        if fallback_data:
            return {
                "status": "success",
                "data": fallback_data,
                "source": "cache_fallback",
                "warning": "Using cached data due to model unavailability",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Ultimate fallback
        return {
            "status": "service_initializing",
            "message": "Risk assessment models are initializing. Please try again in a moment.",
            "retry_after_seconds": 10,
            "error": str(e),
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


@router.get("/predictions/recession")
async def get_recession_prediction(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get recession probability prediction from financial model"""
    try:
        prediction = await model_server.predict_recession_probability()
        return {
            "status": "success",
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recession prediction failed: {str(e)}")


@router.get("/predictions/supply-chain")
async def get_supply_chain_prediction(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get supply chain risk prediction from financial model"""
    try:
        prediction = await model_server.predict_supply_chain_risk()
        return {
            "status": "success",
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supply chain prediction failed: {str(e)}")


@router.get("/predictions/market-volatility")
async def get_market_volatility_prediction(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get market volatility prediction from financial model"""
    try:
        prediction = await model_server.predict_market_volatility()
        return {
            "status": "success",
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market volatility prediction failed: {str(e)}")


@router.get("/predictions/geopolitical")
async def get_geopolitical_prediction(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get geopolitical risk prediction from financial model"""
    try:
        prediction = await model_server.predict_geopolitical_risk()
        return {
            "status": "success",
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geopolitical prediction failed: {str(e)}")


@router.get("/models/status")
async def get_models_status(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get status of all financial models"""
    try:
        status = model_server.get_model_status()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model status check failed: {str(e)}")


@router.post("/models/train")
async def trigger_model_training():
    """Trigger training of all financial models"""
    try:
        from src.ml.training.model_trainer import ModelTrainingPipeline
        
        pipeline = ModelTrainingPipeline()
        results = await pipeline.train_all_models()
        
        return {
            "status": "success",
            "data": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")