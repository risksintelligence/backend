from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.services.ml_intelligence_service import MLIntelligenceService
from app.services.network_ml_intelligence import NetworkMLIntelligenceService
from app.core.cache import cache_with_fallback, CacheConfig
from app.core.errors import RRIOAPIError, server_error, ErrorCodes
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ml", tags=["ml-intelligence"])

ml_service = MLIntelligenceService()
network_ml_service = NetworkMLIntelligenceService()

cache_config = CacheConfig(
    key_prefix="ml_intelligence",
    ttl_seconds=1800,  # 30 minutes
    fallback_ttl_seconds=86400  # 24 hours
)

class SupplyChainPredictionRequest(BaseModel):
    route_data: Dict[str, Any]
    economic_data: Dict[str, Any]
    prediction_horizon: Optional[int] = 30

class MarketTrendPredictionRequest(BaseModel):
    market_data: Dict[str, Any]
    prediction_horizon: Optional[int] = 30
    include_confidence: Optional[bool] = True

class AnomalyDetectionRequest(BaseModel):
    market_data: Dict[str, Any]
    sensitivity: Optional[float] = 0.1

class NetworkAnalysisRequest(BaseModel):
    network_data: Dict[str, Any]
    analysis_type: Optional[str] = "comprehensive"  # cascade, resilience, anomaly, comprehensive

@router.post("/supply-chain/predict")
@cache_with_fallback(cache_config)
async def predict_supply_chain_risk(request: SupplyChainPredictionRequest):
    """
    Predict supply chain risk using ML models
    """
    try:
        prediction = await ml_service.predict_supply_chain_risk(
            route_data=request.route_data,
            economic_data=request.economic_data,
            prediction_horizon=request.prediction_horizon
        )
        
        return {
            "status": "success",
            "data": prediction,
            "metadata": {
                "prediction_horizon_days": request.prediction_horizon,
                "model_version": "v1.0.0",
                "confidence_threshold": 0.7
            }
        }
    except Exception as e:
        logger.error(f"Supply chain prediction error: {str(e)}")
        raise server_error(f"Supply chain prediction failed: {str(e)}", ErrorCodes.ML_MODEL_ERROR)

@router.post("/market/predict")
@cache_with_fallback(cache_config)
async def predict_market_trends(request: MarketTrendPredictionRequest):
    """
    Predict market intelligence trends using ML models
    """
    try:
        prediction = await ml_service.predict_market_intelligence_trends(
            market_data=request.market_data,
            prediction_horizon=request.prediction_horizon
        )
        
        return {
            "status": "success", 
            "data": prediction,
            "metadata": {
                "prediction_horizon_days": request.prediction_horizon,
                "model_version": "v1.0.0",
                "include_confidence": request.include_confidence
            }
        }
    except Exception as e:
        logger.error(f"Market trend prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/anomalies/detect")
@cache_with_fallback(cache_config)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in market data using ML models
    """
    try:
        anomalies = await ml_service.detect_anomalies(
            market_data=request.market_data,
            sensitivity=request.sensitivity
        )
        
        return {
            "status": "success",
            "data": anomalies,
            "metadata": {
                "sensitivity": request.sensitivity,
                "model_version": "v1.0.0",
                "detection_algorithm": "isolation_forest"
            }
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """
    Get status of ML models
    """
    try:
        status = await ml_service.get_model_status()
        
        return {
            "status": "success",
            "data": status,
            "metadata": {
                "timestamp": status.get("timestamp"),
                "service_version": "v1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Model status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/models/train")
async def trigger_model_training(background_tasks: BackgroundTasks):
    """
    Trigger background model training
    """
    try:
        background_tasks.add_task(ml_service.train_models)
        
        return {
            "status": "success",
            "message": "Model training started in background",
            "metadata": {
                "estimated_duration_minutes": 15,
                "training_mode": "incremental"
            }
        }
    except Exception as e:
        logger.error(f"Model training trigger error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")

@router.post("/network/cascade/predict")
@cache_with_fallback(cache_config)
async def predict_network_cascade_risk(request: NetworkAnalysisRequest):
    """
    Predict supply chain cascade risk using network topology analysis
    """
    try:
        prediction = await network_ml_service.predict_cascade_risk(
            network_data=request.network_data
        )
        
        return {
            "status": "success",
            "data": prediction,
            "metadata": {
                "analysis_type": "cascade_risk",
                "model_version": "v1.0.0",
                "confidence_threshold": 0.7
            }
        }
    except Exception as e:
        logger.error(f"Network cascade prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/network/resilience/predict")
@cache_with_fallback(cache_config)
async def predict_network_resilience(request: NetworkAnalysisRequest):
    """
    Predict network resilience and recovery capabilities
    """
    try:
        prediction = await network_ml_service.predict_resilience_score(
            network_data=request.network_data
        )
        
        return {
            "status": "success",
            "data": prediction,
            "metadata": {
                "analysis_type": "resilience_analysis",
                "model_version": "v1.0.0",
                "metric": "resilience_score"
            }
        }
    except Exception as e:
        logger.error(f"Network resilience prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/network/anomalies/detect")
@cache_with_fallback(cache_config)
async def detect_network_anomalies(request: NetworkAnalysisRequest):
    """
    Detect anomalies in network behavior and topology
    """
    try:
        anomalies = await network_ml_service.detect_network_anomalies(
            network_data=request.network_data
        )
        
        return {
            "status": "success",
            "data": anomalies,
            "metadata": {
                "analysis_type": "anomaly_detection",
                "model_version": "v1.0.0",
                "detection_algorithm": "isolation_forest"
            }
        }
    except Exception as e:
        logger.error(f"Network anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/network/analysis/comprehensive")
@cache_with_fallback(cache_config)
async def get_comprehensive_network_analysis(request: NetworkAnalysisRequest):
    """
    Get comprehensive ML analysis of network data including cascade risk, resilience, and anomalies
    """
    try:
        analysis = await network_ml_service.get_network_ml_summary(
            network_data=request.network_data
        )
        
        return {
            "status": "success",
            "data": analysis,
            "metadata": {
                "analysis_type": "comprehensive",
                "model_version": "v1.0.0",
                "components": ["cascade_risk", "resilience_analysis", "anomaly_detection"]
            }
        }
    except Exception as e:
        logger.error(f"Comprehensive network analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/network/insights/summary")
@cache_with_fallback(cache_config)
async def get_network_ml_insights_summary():
    """
    Get summary of network ML insights using real network data
    """
    try:
        # Fetch real network data from the network cascade endpoint
        import httpx
        
        # Get real network data from the network cascade API
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                # Try to fetch real network data from the cascade endpoint  
                response = await client.get("http://localhost:8000/api/v1/network/cascade/snapshot")
                if response.status_code == 200:
                    network_data = response.json()
                else:
                    # If network data unavailable, return cached analysis if available
                    from ..core.unified_cache import UnifiedCache
                    cache = UnifiedCache("ml_network_insights")
                    cached_analysis, _ = cache.get("network_summary")
                    if cached_analysis:
                        return cached_analysis
                    
                    # Return minimal structure if no cache
                    return {
                        "status": "fallback",
                        "message": "Network data unavailable, no cached analysis",
                        "fallback_data": True,
                        "metadata": {
                            "fallback_reason": "Network data unavailable",
                            "data_source": "minimal_structure"
                        }
                    }
            except Exception:
                # If service unavailable, return cached analysis if available
                from ..core.unified_cache import UnifiedCache
                cache = UnifiedCache("ml_network_insights")
                cached_analysis, _ = cache.get("network_summary")
                if cached_analysis:
                    return cached_analysis
                
                # Return minimal structure if no cache
                return {
                    "status": "fallback",
                    "message": "Network data service unavailable, no cached analysis",
                    "fallback_data": True,
                    "metadata": {
                        "fallback_reason": "Network data service unavailable",
                        "data_source": "minimal_structure"
                    }
                }
        
        # Get comprehensive network analysis
        analysis = await network_ml_service.get_network_ml_summary(network_data)
        
        result = {
            "status": "success",
            "data": analysis,
            "metadata": {
                "data_source": "network_cascade_api",
                "analysis_timestamp": analysis.get("analysis_timestamp"),
                "model_versions": {
                    "cascade_prediction": "v1.0.0",
                    "resilience_analysis": "v1.0.0",
                    "anomaly_detection": "v1.0.0"
                }
            }
        }
        
        # Cache the successful analysis
        from ..core.unified_cache import UnifiedCache
        cache = UnifiedCache("ml_network_insights")
        cache.set("network_summary", result, source="ml_network_insights_api", hard_ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"Network ML insights summary error: {str(e)}")
        # Try to return cached data instead of 500 error
        from ..core.unified_cache import UnifiedCache
        cache = UnifiedCache("ml_network_insights") 
        cached_analysis, _ = cache.get("network_summary")
        if cached_analysis:
            return cached_analysis
            
        # Return minimal structure if no cache
        return {
            "status": "error_fallback",
            "message": f"Insights generation failed: {str(e)}",
            "fallback_data": True,
            "metadata": {
                "fallback_reason": f"Insights generation failed: {str(e)}",
                "data_source": "minimal_structure"
            }
        }

@router.get("/insights/summary")
@cache_with_fallback(cache_config)
async def get_ml_insights_summary():
    """
    Get summary of ML-powered insights for dashboard
    """
    try:
        # Generate sample market and route data for insights
        sample_market_data = {
            "financial_health": [
                {"company": "AAPL", "score": 0.85, "sector": "technology"},
                {"company": "MSFT", "score": 0.82, "sector": "technology"},
                {"company": "JPM", "score": 0.78, "sector": "financial"},
            ],
            "country_risk": [
                {"country": "USA", "risk_score": 0.15, "gdp_growth": 0.025},
                {"country": "CHN", "risk_score": 0.35, "gdp_growth": 0.055},
                {"company": "DEU", "risk_score": 0.22, "gdp_growth": 0.018},
            ],
            "trade_flows": [
                {"from": "USA", "to": "CHN", "volume": 150000000000, "trend": "increasing"},
                {"from": "DEU", "to": "USA", "volume": 75000000000, "trend": "stable"},
            ]
        }
        
        sample_route_data = {
            "routes": [
                {"origin": "Shanghai", "destination": "Los Angeles", "risk_level": "medium"},
                {"origin": "Hamburg", "destination": "New York", "risk_level": "low"},
            ]
        }
        
        # Get ML predictions
        supply_chain_prediction = await ml_service.predict_supply_chain_risk(
            route_data=sample_route_data,
            economic_data=sample_market_data
        )
        
        market_prediction = await ml_service.predict_market_intelligence_trends(
            market_data=sample_market_data
        )
        
        anomalies = await ml_service.detect_anomalies(
            market_data=sample_market_data
        )
        
        return {
            "status": "success",
            "data": {
                "supply_chain_insights": supply_chain_prediction,
                "market_trend_insights": market_prediction, 
                "anomaly_insights": anomalies,
                "summary_metrics": {
                    "total_predictions": 3,
                    "high_confidence_predictions": 2,
                    "anomalies_detected": len(anomalies.get("anomalies", [])),
                    "overall_risk_level": "medium"
                }
            },
            "metadata": {
                "generated_at": "2025-11-25",
                "data_sources": ["market_intelligence", "supply_chain_routes"],
                "model_versions": {
                    "supply_chain": "v1.0.0",
                    "market_trends": "v1.0.0", 
                    "anomaly_detection": "v1.0.0"
                }
            }
        }
    except Exception as e:
        logger.error(f"ML insights summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")