"""
Model Serving API for RiskX ML Models
Production-ready model serving with real-time predictions
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from src.ml.models.recession_predictor import RecessionPredictor, EconomicIndicators
from src.ml.models.supply_chain_risk_model import SupplyChainRiskModel, SupplyChainMetrics
from src.ml.models.market_volatility_model import MarketVolatilityModel, MarketIndicators
from src.ml.models.geopolitical_risk_model import GeopoliticalRiskModel, GeopoliticalIndicators
from src.ml.explainability.shap_analyzer import ShapAnalyzer
from src.data.sources.fred import FREDClient
from src.data.sources.bea import BEAClient
from src.data.sources.bls import BLSClient
from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelServer:
    """Production model serving with real-time predictions and explanations"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.shap_analyzer = ShapAnalyzer()
        
        # Data clients for real-time features
        self.fred_client = FREDClient()
        self.bea_client = BEAClient()
        self.bls_client = BLSClient()
        
        self.is_initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all models and load them into memory"""
        logger.info("Initializing RiskX Model Server...")
        
        try:
            # Load all trained models
            await self._load_models()
            
            # Initialize SHAP analyzer
            await self._initialize_shap_analyzer()
            
            self.is_initialized = True
            
            logger.info("Model server initialization completed")
            
            return {
                "status": "initialized",
                "models_loaded": list(self.models.keys()),
                "model_metadata": self.model_metadata,
                "shap_enabled": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model server initialization failed: {str(e)}")
            raise
    
    async def _load_models(self) -> None:
        """Load all trained models"""
        try:
            # Load recession prediction model
            self.models['recession_predictor'] = RecessionPredictor()
            self.models['recession_predictor'].load_models()
            
            # Load supply chain risk model  
            self.models['supply_chain_risk'] = SupplyChainRiskModel()
            self.models['supply_chain_risk'].load_models()
            
            # Load market volatility model
            self.models['market_volatility'] = MarketVolatilityModel()
            self.models['market_volatility'].load_models()
            
            # Load geopolitical risk model
            self.models['geopolitical_risk'] = GeopoliticalRiskModel()
            self.models['geopolitical_risk'].load_models()
            
            # Store model metadata
            for model_name, model in self.models.items():
                self.model_metadata[model_name] = {
                    "version": getattr(model, 'version', 'unknown'),
                    "last_trained": getattr(model, 'last_trained', None),
                    "model_type": type(model).__name__,
                    "status": "loaded"
                }
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Try to load models individually and report which ones fail
            for model_name in ['recession_predictor', 'supply_chain_risk', 'market_volatility', 'geopolitical_risk']:
                try:
                    if model_name == 'recession_predictor':
                        model = RecessionPredictor()
                        model.load_models()
                        self.models[model_name] = model
                        self.model_metadata[model_name] = {"status": "loaded"}
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {str(model_error)}")
                    self.model_metadata[model_name] = {"status": "failed", "error": str(model_error)}
    
    async def _initialize_shap_analyzer(self) -> None:
        """Initialize SHAP analyzer for model explainability"""
        try:
            # Initialize SHAP analyzer with loaded models
            for model_name, model in self.models.items():
                if hasattr(model, 'models') and model.models:
                    # Use the first model in ensemble for SHAP analysis
                    first_model_name = list(model.models.keys())[0]
                    first_model = model.models[first_model_name]
                    await self.shap_analyzer.add_model(model_name, first_model)
            
            logger.info("SHAP analyzer initialized")
            
        except Exception as e:
            logger.warning(f"SHAP analyzer initialization failed: {str(e)}")
    
    async def predict_recession_probability(self, 
                                           economic_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get recession probability prediction with real-time data"""
        if not self.is_initialized:
            await self.initialize()
        
        model = self.models.get('recession_predictor')
        if not model:
            raise ValueError("Recession prediction model not available")
        
        try:
            # Get real-time economic data if not provided
            if economic_data is None:
                economic_data = await self._fetch_real_time_economic_data()
            
            # Create economic indicators object
            indicators = EconomicIndicators(
                yield_curve_10y2y=economic_data.get('yield_curve_10y2y'),
                yield_curve_10y3m=economic_data.get('yield_curve_10y3m'),
                unemployment_rate=economic_data.get('unemployment_rate'),
                gdp_growth_rate=economic_data.get('gdp_growth_rate'),
                sp500_volatility=economic_data.get('sp500_volatility'),
                manufacturing_pmi=economic_data.get('manufacturing_pmi'),
                consumer_confidence=economic_data.get('consumer_confidence'),
                industrial_production_change=economic_data.get('industrial_production_change')
            )
            
            # Get prediction
            prediction = await model.predict_recession_probability(indicators)
            
            # Get SHAP explanation if available
            explanation = None
            try:
                explanation = await self.shap_analyzer.explain_prediction(
                    'recession_predictor', 
                    [list(economic_data.values())]
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
            
            return {
                "model": "recession_predictor",
                "prediction": {
                    "probability": prediction.probability,
                    "confidence": prediction.confidence,
                    "risk_level": prediction.risk_level,
                    "time_horizon": prediction.time_horizon
                },
                "key_indicators": prediction.key_indicators,
                "contributing_factors": prediction.contributing_factors,
                "explanation": explanation,
                "data_timestamp": datetime.utcnow().isoformat(),
                "model_version": prediction.model_version
            }
            
        except Exception as e:
            logger.error(f"Recession prediction failed: {str(e)}")
            raise
    
    async def predict_supply_chain_risk(self, 
                                       supply_chain_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get supply chain risk prediction"""
        if not self.is_initialized:
            await self.initialize()
        
        model = self.models.get('supply_chain_risk')
        if not model:
            raise ValueError("Supply chain risk model not available")
        
        try:
            # Get real-time supply chain data if not provided
            if supply_chain_data is None:
                supply_chain_data = await self._fetch_real_time_supply_chain_data()
            
            # Create supply chain metrics object
            metrics = SupplyChainMetrics(
                supplier_concentration=supply_chain_data.get('supplier_concentration', 0.0),
                geographic_concentration=supply_chain_data.get('geographic_concentration', 0.0),
                single_source_dependencies=supply_chain_data.get('single_source_dependencies', 0),
                capacity_utilization=supply_chain_data.get('capacity_utilization', 0.0),
                lead_time_variability=supply_chain_data.get('lead_time_variability', 0.0)
            )
            
            # Get prediction
            prediction = await model.predict_supply_chain_risk(metrics)
            
            return {
                "model": "supply_chain_risk",
                "prediction": {
                    "risk_score": prediction.risk_score,
                    "risk_level": prediction.risk_level,
                    "confidence": prediction.confidence
                },
                "risk_factors": prediction.risk_factors,
                "mitigation_recommendations": prediction.mitigation_recommendations,
                "data_timestamp": datetime.utcnow().isoformat(),
                "model_version": prediction.model_version
            }
            
        except Exception as e:
            logger.error(f"Supply chain risk prediction failed: {str(e)}")
            raise
    
    async def predict_market_volatility(self, 
                                       market_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get market volatility prediction"""
        if not self.is_initialized:
            await self.initialize()
        
        model = self.models.get('market_volatility')
        if not model:
            raise ValueError("Market volatility model not available")
        
        try:
            # Get real-time market data if not provided
            if market_data is None:
                market_data = await self._fetch_real_time_market_data()
            
            # Create market data object
            data = MarketData(
                current_volatility=market_data.get('current_volatility', 0.0),
                price_change=market_data.get('price_change', 0.0),
                volume_change=market_data.get('volume_change', 0.0),
                interest_rate=market_data.get('interest_rate', 0.0),
                economic_uncertainty=market_data.get('economic_uncertainty', 0.0)
            )
            
            # Get prediction
            prediction = await model.predict_volatility(data)
            
            return {
                "model": "market_volatility",
                "prediction": {
                    "volatility_forecast": prediction.volatility_forecast,
                    "volatility_regime": prediction.volatility_regime,
                    "confidence": prediction.confidence
                },
                "regime_probabilities": prediction.regime_probabilities,
                "risk_metrics": prediction.risk_metrics,
                "data_timestamp": datetime.utcnow().isoformat(),
                "model_version": prediction.model_version
            }
            
        except Exception as e:
            logger.error(f"Market volatility prediction failed: {str(e)}")
            raise
    
    async def predict_geopolitical_risk(self, 
                                       geopolitical_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get geopolitical risk prediction"""
        if not self.is_initialized:
            await self.initialize()
        
        model = self.models.get('geopolitical_risk')
        if not model:
            raise ValueError("Geopolitical risk model not available")
        
        try:
            # Get real-time geopolitical data if not provided
            if geopolitical_data is None:
                geopolitical_data = await self._fetch_real_time_geopolitical_data()
            
            # Create geopolitical indicators object
            indicators = GeopoliticalIndicators(
                political_stability=geopolitical_data.get('political_stability', 0.0),
                trade_tensions=geopolitical_data.get('trade_tensions', 0.0),
                military_conflicts=geopolitical_data.get('military_conflicts', 0.0),
                diplomatic_relations=geopolitical_data.get('diplomatic_relations', 0.0),
                economic_sanctions=geopolitical_data.get('economic_sanctions', 0.0)
            )
            
            # Get prediction
            prediction = await model.predict_geopolitical_risk(indicators)
            
            return {
                "model": "geopolitical_risk", 
                "prediction": {
                    "overall_risk_score": prediction.overall_risk_score,
                    "risk_category": prediction.risk_category,
                    "confidence": prediction.confidence
                },
                "regional_risks": prediction.regional_risks,
                "risk_drivers": prediction.risk_drivers,
                "data_timestamp": datetime.utcnow().isoformat(),
                "model_version": prediction.model_version
            }
            
        except Exception as e:
            logger.error(f"Geopolitical risk prediction failed: {str(e)}")
            raise
    
    async def get_comprehensive_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment from all models"""
        logger.info("Generating comprehensive risk assessment...")
        
        try:
            # Get predictions from all models in parallel
            predictions = await asyncio.gather(
                self.predict_recession_probability(),
                self.predict_supply_chain_risk(),
                self.predict_market_volatility(),
                self.predict_geopolitical_risk(),
                return_exceptions=True
            )
            
            model_names = ['recession', 'supply_chain', 'market_volatility', 'geopolitical']
            results = {}
            
            # Process results
            for i, (model_name, prediction) in enumerate(zip(model_names, predictions)):
                if isinstance(prediction, Exception):
                    logger.warning(f"{model_name} prediction failed: {str(prediction)}")
                    results[model_name] = {"status": "failed", "error": str(prediction)}
                else:
                    results[model_name] = prediction
            
            # Calculate overall risk score
            risk_scores = []
            for model_name, result in results.items():
                if result.get("status") != "failed":
                    pred = result.get("prediction", {})
                    if "probability" in pred:
                        risk_scores.append(pred["probability"])
                    elif "risk_score" in pred:
                        risk_scores.append(pred["risk_score"])
                    elif "overall_risk_score" in pred:
                        risk_scores.append(pred["overall_risk_score"])
                    elif "volatility_forecast" in pred:
                        # Normalize volatility to 0-1 scale
                        risk_scores.append(min(1.0, pred["volatility_forecast"] / 50.0))
            
            overall_risk = np.mean(risk_scores) if risk_scores else 0.0
            
            return {
                "comprehensive_assessment": {
                    "overall_risk_score": overall_risk,
                    "risk_level": self._categorize_risk_level(overall_risk),
                    "assessment_timestamp": datetime.utcnow().isoformat(),
                    "models_used": len([r for r in results.values() if r.get("status") != "failed"]),
                    "total_models": len(model_names)
                },
                "individual_predictions": results,
                "risk_summary": {
                    "highest_risk_area": max(results.keys(), key=lambda k: self._extract_risk_score(results[k])) if risk_scores else None,
                    "average_confidence": np.mean([self._extract_confidence(r) for r in results.values() if r.get("status") != "failed"]) if results else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive risk assessment failed: {str(e)}")
            raise
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on score"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _extract_risk_score(self, result: Dict[str, Any]) -> float:
        """Extract risk score from model result"""
        if result.get("status") == "failed":
            return 0.0
        
        pred = result.get("prediction", {})
        if "probability" in pred:
            return pred["probability"]
        elif "risk_score" in pred:
            return pred["risk_score"]
        elif "overall_risk_score" in pred:
            return pred["overall_risk_score"]
        elif "volatility_forecast" in pred:
            return min(1.0, pred["volatility_forecast"] / 50.0)
        return 0.0
    
    def _extract_confidence(self, result: Dict[str, Any]) -> float:
        """Extract confidence from model result"""
        if result.get("status") == "failed":
            return 0.0
        
        pred = result.get("prediction", {})
        return pred.get("confidence", 0.0)
    
    async def _fetch_real_time_economic_data(self) -> Dict[str, float]:
        """Fetch real-time economic data from APIs - real data only"""
        try:
            # Get latest economic indicators from FRED
            series_data = await self.fred_client.get_multiple_series(
                series_ids=['DGS10', 'DGS2', 'DGS3MO', 'UNRATE', 'VIXCLS', 'INDPRO', 'NAPM', 'UMCSENT'],
                limit=1
            )
            
            if not series_data:
                raise ValueError("No real economic data available from FRED API")
            
            # Calculate derived indicators
            data = {}
            
            # Yield curve calculations require real data
            if 'DGS10' in series_data and 'DGS2' in series_data:
                data['yield_curve_10y2y'] = series_data['DGS10'][-1] - series_data['DGS2'][-1]
            else:
                raise ValueError("Yield curve data not available from FRED")
                
            if 'DGS10' in series_data and 'DGS3MO' in series_data:
                data['yield_curve_10y3m'] = series_data['DGS10'][-1] - series_data['DGS3MO'][-1]
            else:
                raise ValueError("3-month yield curve data not available from FRED")
            
            # All other data must come from real APIs
            if 'UNRATE' not in series_data:
                raise ValueError("Unemployment rate not available from FRED")
            if 'VIXCLS' not in series_data:
                raise ValueError("VIX volatility data not available from FRED")
            if 'INDPRO' not in series_data:
                raise ValueError("Industrial production data not available from FRED")
            if 'NAPM' not in series_data:
                raise ValueError("Manufacturing PMI data not available from FRED")
            if 'UMCSENT' not in series_data:
                raise ValueError("Consumer confidence data not available from FRED")
            
            # Get additional data from BEA for GDP growth
            bea_data = await self.bea_client.get_gdp_data()
            if not bea_data or 'growth_rate' not in bea_data:
                raise ValueError("GDP growth rate not available from BEA API")
            
            data.update({
                'unemployment_rate': series_data['UNRATE'][-1],
                'sp500_volatility': series_data['VIXCLS'][-1],
                'gdp_growth_rate': bea_data['growth_rate'],
                'manufacturing_pmi': series_data['NAPM'][-1],
                'consumer_confidence': series_data['UMCSENT'][-1],
                'industrial_production_change': series_data['INDPRO'][-1]
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch real-time economic data: {str(e)}")
            raise ValueError(f"Real economic data not available - synthetic data not allowed: {str(e)}")
    
    async def _fetch_real_time_supply_chain_data(self) -> Dict[str, Any]:
        """Fetch real-time supply chain data from APIs - real data only"""
        try:
            # Must implement real supply chain data APIs
            # This would integrate with actual supply chain monitoring services
            raise ValueError("Real supply chain APIs not yet implemented - synthetic data not allowed")
        except Exception as e:
            logger.error(f"Supply chain data not available: {str(e)}")
            raise ValueError("Real supply chain data required - synthetic data not allowed")
    
    async def _fetch_real_time_market_data(self) -> Dict[str, float]:
        """Fetch real-time market data from APIs - real data only"""
        try:
            # Must implement real market data APIs
            # This would integrate with actual market data providers
            raise ValueError("Real market data APIs not yet implemented - synthetic data not allowed")
        except Exception as e:
            logger.error(f"Market data not available: {str(e)}")
            raise ValueError("Real market data required - synthetic data not allowed")
    
    async def _fetch_real_time_geopolitical_data(self) -> Dict[str, float]:
        """Fetch real-time geopolitical data from APIs - real data only"""
        try:
            # Must implement real geopolitical risk APIs
            # This would integrate with actual geopolitical risk data providers
            raise ValueError("Real geopolitical APIs not yet implemented - synthetic data not allowed")
        except Exception as e:
            logger.error(f"Geopolitical data not available: {str(e)}")
            raise ValueError("Real geopolitical data required - synthetic data not allowed")
    
    def get_model(self, model_id: str):
        """Get a specific model by ID"""
        return self.models.get(model_id)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            "server_status": "initialized" if self.is_initialized else "not_initialized",
            "models_loaded": len(self.models),
            "model_metadata": self.model_metadata,
            "available_predictions": [
                "recession_probability",
                "supply_chain_risk",
                "market_volatility", 
                "geopolitical_risk",
                "comprehensive_assessment"
            ],
            "shap_enabled": hasattr(self, 'shap_analyzer'),
            "last_check": datetime.utcnow().isoformat()
        }


# Global model server instance
model_server = ModelServer()


async def get_model_server() -> ModelServer:
    """Get initialized model server instance"""
    if not model_server.is_initialized:
        await model_server.initialize()
    return model_server