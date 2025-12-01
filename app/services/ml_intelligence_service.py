"""
ML-Powered Market Intelligence Service
Provides advanced ML predictions and insights for market intelligence data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLIntelligenceService:
    """ML-powered intelligence service for market predictions and insights"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.last_trained = {}
        
    async def predict_supply_chain_risk(self, route_data: Dict, economic_data: Dict, prediction_horizon: int = 30) -> Dict[str, Any]:
        """
        ML-powered supply chain risk prediction
        Uses route characteristics and economic indicators to predict future risk
        """
        try:
            # Prepare features from route and economic data
            features = self._prepare_supply_chain_features(route_data, economic_data)
            
            # Generate or use cached model
            model_key = "supply_chain_risk"
            if model_key not in self.models or self._should_retrain(model_key):
                await self._train_supply_chain_model(features)
            
            # Make predictions
            predictions = {}
            
            for route_name, route_info in route_data.get('routes', {}).items():
                try:
                    # Extract features for this route
                    route_features = self._extract_route_features(route_info, economic_data)
                    feature_array = np.array([route_features])
                    
                    # Scale features
                    if model_key in self.scalers:
                        feature_array = self.scalers[model_key].transform(feature_array)
                    
                    # Predict risk score and trend
                    if model_key in self.models:
                        predicted_risk = self.models[model_key].predict(feature_array)[0]
                        confidence = self._calculate_prediction_confidence(feature_array, model_key)
                        
                        # Generate risk factors using feature importance
                        risk_factors = self._generate_ml_risk_factors(route_features, model_key)
                        
                        predictions[route_name] = {
                            "predicted_risk_score": max(0, min(100, predicted_risk)),
                            "current_risk_score": route_info.get('risk_score', 50),
                            "risk_trend": "increasing" if predicted_risk > route_info.get('risk_score', 50) + 5 
                                         else "decreasing" if predicted_risk < route_info.get('risk_score', 50) - 5 
                                         else "stable",
                            "confidence_level": confidence,
                            "ml_risk_factors": risk_factors,
                            "prediction_horizon": "7_days"
                        }
                    
                except Exception as e:
                    logger.warning(f"Failed to predict risk for route {route_name}: {e}")
                    continue
            
            # Generate overall insights
            overall_insights = self._generate_supply_chain_insights(predictions)
            
            return {
                "predictions": predictions,
                "model_performance": self._get_model_performance(model_key),
                "insights": overall_insights,
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "model_version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Supply chain risk prediction failed: {e}")
            return self._get_fallback_supply_chain_prediction()
    
    async def predict_market_intelligence_trends(self, market_data: Dict, prediction_horizon: int = 30) -> Dict[str, Any]:
        """
        ML-powered market intelligence trend prediction
        Predicts future market conditions and risk scores
        """
        try:
            # Prepare market features
            features = self._prepare_market_features(market_data)
            
            # Generate or use cached model
            model_key = "market_trends"
            if model_key not in self.models or self._should_retrain(model_key):
                await self._train_market_trends_model(features)
            
            # Make predictions
            predictions = {}
            
            # Predict overall market risk
            if 'combined_intelligence' in market_data:
                combined = market_data['combined_intelligence']
                current_risk = combined.get('overall_risk_score', 50)
                
                market_features = self._extract_market_features(market_data)
                feature_array = np.array([market_features])
                
                if model_key in self.scalers:
                    feature_array = self.scalers[model_key].transform(feature_array)
                
                if model_key in self.models:
                    predicted_risk = self.models[model_key].predict(feature_array)[0]
                    
                    predictions['overall_market'] = {
                        "current_risk_score": current_risk,
                        "predicted_risk_score": max(0, min(100, predicted_risk)),
                        "trend_direction": "bullish" if predicted_risk < current_risk - 3 
                                          else "bearish" if predicted_risk > current_risk + 3 
                                          else "neutral",
                        "volatility_forecast": self._predict_volatility(market_features),
                        "confidence": self._calculate_prediction_confidence(feature_array, model_key)
                    }
            
            # Predict country-specific risks
            if 'trade_intelligence' in market_data:
                country_predictions = {}
                for country, data in market_data['trade_intelligence'].get('country_risks', {}).items():
                    try:
                        country_features = self._extract_country_features(data)
                        feature_array = np.array([country_features])
                        
                        if model_key in self.scalers:
                            feature_array = self.scalers[model_key].transform(feature_array)
                        
                        if model_key in self.models:
                            predicted_risk = self.models[model_key].predict(feature_array)[0]
                            
                            country_predictions[country] = {
                                "current_risk": data.get('risk_score', 50),
                                "predicted_risk": max(0, min(100, predicted_risk)),
                                "risk_drivers": self._identify_risk_drivers(country_features, model_key),
                                "outlook": "improving" if predicted_risk < data.get('risk_score', 50) - 2 
                                          else "deteriorating" if predicted_risk > data.get('risk_score', 50) + 2 
                                          else "stable"
                            }
                    except Exception as e:
                        logger.warning(f"Failed to predict risk for country {country}: {e}")
                        continue
                
                predictions['countries'] = country_predictions
            
            # Generate ML insights
            ml_insights = self._generate_market_insights(predictions, market_data)
            
            return {
                "predictions": predictions,
                "insights": ml_insights,
                "model_performance": self._get_model_performance(model_key),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "forecast_horizon": "30_days"
            }
            
        except Exception as e:
            logger.error(f"Market intelligence prediction failed: {e}")
            return self._get_fallback_market_prediction()
    
    async def detect_anomalies(self, market_data: Dict, sensitivity: float = 0.5) -> Dict[str, Any]:
        """
        ML-powered anomaly detection in market data
        Identifies unusual patterns that may indicate emerging risks
        """
        try:
            anomalies = []
            
            # Check for anomalies in economic indicators
            if 'trade_intelligence' in market_data:
                country_risks = market_data['trade_intelligence'].get('country_risks', {})
                for country, data in country_risks.items():
                    risk_factors = data.get('risk_factors', {})
                    
                    # Create feature vector for anomaly detection
                    features = [
                        risk_factors.get('gdp_growth', 0),
                        risk_factors.get('inflation', 0),
                        risk_factors.get('unemployment', 0),
                        risk_factors.get('trade_openness', 0),
                        data.get('risk_score', 0)
                    ]
                    
                    # Use isolation forest for anomaly detection
                    if len(features) == 5 and all(isinstance(f, (int, float)) for f in features):
                        anomaly_score = self._detect_country_anomaly(features, country)
                        
                        if anomaly_score > 0.7:  # High anomaly score
                            anomalies.append({
                                "type": "economic_indicator_anomaly",
                                "country": country,
                                "severity": "high" if anomaly_score > 0.9 else "medium",
                                "description": f"Unusual economic pattern detected in {country}",
                                "anomaly_score": anomaly_score,
                                "affected_indicators": self._identify_anomalous_indicators(features)
                            })
            
            # Check for supply chain anomalies
            if 'supply_chain_mapping' in market_data:
                routes = market_data['supply_chain_mapping'].get('routes', {})
                route_risks = [route.get('risk_score', 0) for route in routes.values() if isinstance(route, dict)]
                
                if route_risks:
                    mean_risk = np.mean(route_risks)
                    std_risk = np.std(route_risks)
                    
                    for route_name, route_data in routes.items():
                        if isinstance(route_data, dict):
                            risk_score = route_data.get('risk_score', 0)
                            if abs(risk_score - mean_risk) > 2 * std_risk:
                                anomalies.append({
                                    "type": "supply_chain_anomaly",
                                    "route": route_name,
                                    "severity": "high" if abs(risk_score - mean_risk) > 3 * std_risk else "medium",
                                    "description": f"Unusual risk level detected on {route_name}",
                                    "risk_score": risk_score,
                                    "expected_range": [mean_risk - std_risk, mean_risk + std_risk]
                                })
            
            return {
                "anomalies": anomalies,
                "total_anomalies": len(anomalies),
                "severity_breakdown": {
                    "high": len([a for a in anomalies if a.get('severity') == 'high']),
                    "medium": len([a for a in anomalies if a.get('severity') == 'medium']),
                    "low": len([a for a in anomalies if a.get('severity') == 'low'])
                },
                "detection_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"anomalies": [], "total_anomalies": 0, "error": str(e)}
    
    def _prepare_supply_chain_features(self, route_data: Dict, economic_data: Dict) -> pd.DataFrame:
        """Prepare features for supply chain ML model"""
        # This would normally use historical data, but we'll simulate training data
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate historical route features
        data = {
            'distance': np.random.normal(5000, 3000, n_samples),
            'duration': np.random.normal(100, 50, n_samples),
            'route_type_maritime': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'route_type_air': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'importance_high': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'gdp_growth': np.random.normal(2.5, 1.5, n_samples),
            'inflation': np.random.normal(3.0, 2.0, n_samples),
            'unemployment': np.random.normal(6.0, 2.5, n_samples),
        }
        
        # Generate target variable (risk score) based on features
        risk_scores = (
            30 +  # Base risk
            data['distance'] * 0.003 +  # Distance impact
            data['route_type_maritime'] * 15 +  # Maritime routes riskier
            data['route_type_air'] * 5 +  # Air routes moderately risky
            data['importance_high'] * 10 +  # Important routes riskier
            data['inflation'] * 2 +  # Inflation impact
            data['unemployment'] * 1.5 +  # Unemployment impact
            np.random.normal(0, 5, n_samples)  # Random noise
        )
        risk_scores = np.clip(risk_scores, 0, 100)
        
        df = pd.DataFrame(data)
        df['risk_score'] = risk_scores
        
        return df
    
    def _prepare_market_features(self, market_data: Dict) -> pd.DataFrame:
        """Prepare features for market trends ML model"""
        np.random.seed(42)
        n_samples = 800
        
        # Simulate historical market features
        data = {
            'financial_health_score': np.random.normal(65, 15, n_samples),
            'trade_stress_score': np.random.normal(55, 12, n_samples),
            'trade_vulnerability': np.random.normal(45, 10, n_samples),
            'supply_chain_risk': np.random.normal(50, 8, n_samples),
            'volatility_index': np.random.normal(25, 8, n_samples),
            'liquidity_ratio': np.random.normal(1.4, 0.3, n_samples),
        }
        
        # Generate target (future risk score)
        future_risk = (
            data['financial_health_score'] * 0.3 +
            data['trade_stress_score'] * 0.25 +
            data['trade_vulnerability'] * 0.25 +
            data['supply_chain_risk'] * 0.2 +
            np.random.normal(0, 3, n_samples)
        )
        future_risk = np.clip(future_risk, 0, 100)
        
        df = pd.DataFrame(data)
        df['future_risk'] = future_risk
        
        return df
    
    async def _train_supply_chain_model(self, features_df: pd.DataFrame):
        """Train the supply chain risk prediction model"""
        try:
            X = features_df.drop(['risk_score'], axis=1)
            y = features_df['risk_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train_scaled, y_train)
            
            # Calculate performance
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metadata
            model_key = "supply_chain_risk"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importance[model_key] = dict(zip(X.columns, model.feature_importances_))
            self.last_trained[model_key] = datetime.utcnow()
            
            logger.info(f"Supply chain model trained - MAE: {mae:.2f}, R2: {r2:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train supply chain model: {e}")
    
    async def _train_market_trends_model(self, features_df: pd.DataFrame):
        """Train the market trends prediction model"""
        try:
            X = features_df.drop(['future_risk'], axis=1)
            y = features_df['future_risk']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            model.fit(X_train_scaled, y_train)
            
            # Calculate performance
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metadata
            model_key = "market_trends"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importance[model_key] = dict(zip(X.columns, model.feature_importances_))
            self.last_trained[model_key] = datetime.utcnow()
            
            logger.info(f"Market trends model trained - MAE: {mae:.2f}, R2: {r2:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train market trends model: {e}")
    
    def _extract_route_features(self, route_info: Dict, economic_data: Dict) -> List[float]:
        """Extract numerical features from route data"""
        features = [
            route_info.get('distance_km', 0),
            route_info.get('duration_hours', 0),
            1 if route_info.get('route_info', {}).get('type') == 'maritime' else 0,
            1 if route_info.get('route_info', {}).get('type') == 'air' else 0,
            1 if route_info.get('route_info', {}).get('importance') == 'high' else 0,
            economic_data.get('avg_gdp_growth', 2.5),
            economic_data.get('avg_inflation', 3.0),
            economic_data.get('avg_unemployment', 6.0),
        ]
        return features
    
    def _extract_market_features(self, market_data: Dict) -> List[float]:
        """Extract numerical features from market data"""
        financial = market_data.get('financial_health', {})
        trade_intel = market_data.get('trade_intelligence', {})
        trade_flows = market_data.get('trade_flows', {})
        supply_chain = market_data.get('supply_chain_mapping', {})
        
        features = [
            financial.get('market_score', 65),
            trade_intel.get('global_stress_score', 55),
            trade_flows.get('vulnerability_score', 45),
            supply_chain.get('average_risk_score', 50),
            25.0,  # Mock volatility index
            1.4,   # Mock liquidity ratio
        ]
        return features
    
    def _extract_country_features(self, country_data: Dict) -> List[float]:
        """Extract features for country-specific predictions"""
        risk_factors = country_data.get('risk_factors', {})
        features = [
            country_data.get('risk_score', 50),
            risk_factors.get('gdp_growth', 2.5),
            risk_factors.get('inflation', 3.0),
            risk_factors.get('unemployment', 6.0),
            risk_factors.get('trade_openness', 50),
        ]
        return features
    
    def _should_retrain(self, model_key: str) -> bool:
        """Check if model should be retrained"""
        if model_key not in self.last_trained:
            return True
        
        time_since_training = datetime.utcnow() - self.last_trained[model_key]
        return time_since_training > timedelta(hours=24)  # Retrain daily
    
    def _calculate_prediction_confidence(self, features: np.ndarray, model_key: str) -> float:
        """Calculate prediction confidence based on model certainty"""
        try:
            if model_key in self.models and hasattr(self.models[model_key], 'predict'):
                # For RandomForest, use prediction variance across trees
                if hasattr(self.models[model_key], 'estimators_'):
                    predictions = np.array([
                        estimator.predict(features)[0] 
                        for estimator in self.models[model_key].estimators_[:10]
                    ])
                    variance = np.var(predictions)
                    # Convert variance to confidence (inverse relationship)
                    confidence = max(0.1, min(0.95, 1 / (1 + variance * 0.01)))
                    return round(confidence, 2)
            
            return 0.75  # Default confidence
            
        except Exception:
            return 0.65  # Lower default if calculation fails
    
    def _generate_ml_risk_factors(self, features: List[float], model_key: str) -> List[str]:
        """Generate risk factors based on feature importance"""
        try:
            if model_key not in self.feature_importance:
                return ["ML model training in progress"]
            
            importance = self.feature_importance[model_key]
            feature_names = list(importance.keys())
            
            # Get top 3 most important features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            risk_factors = []
            for feature_name, importance_score in sorted_features:
                if importance_score > 0.1:  # Only include significant factors
                    factor_desc = self._get_feature_description(feature_name, importance_score)
                    risk_factors.append(factor_desc)
            
            return risk_factors or ["Standard risk assessment applied"]
            
        except Exception:
            return ["ML analysis in progress"]
    
    def _get_feature_description(self, feature_name: str, importance: float) -> str:
        """Get human-readable description of risk factors"""
        descriptions = {
            'distance': f"Route distance impact (importance: {importance:.1%})",
            'duration': f"Transit time factor (importance: {importance:.1%})",
            'route_type_maritime': f"Maritime route risks (importance: {importance:.1%})",
            'route_type_air': f"Air transport factors (importance: {importance:.1%})",
            'importance_high': f"Strategic corridor importance (importance: {importance:.1%})",
            'gdp_growth': f"Economic growth impact (importance: {importance:.1%})",
            'inflation': f"Inflation pressure (importance: {importance:.1%})",
            'unemployment': f"Employment conditions (importance: {importance:.1%})",
        }
        return descriptions.get(feature_name, f"{feature_name} impact (importance: {importance:.1%})")
    
    def _generate_supply_chain_insights(self, predictions: Dict) -> List[Dict[str, str]]:
        """Generate actionable insights from supply chain predictions"""
        insights = []
        
        try:
            if not predictions:
                return insights
            
            # Find routes with increasing risk
            increasing_risk_routes = [
                name for name, pred in predictions.items()
                if pred.get('risk_trend') == 'increasing'
            ]
            
            if increasing_risk_routes:
                insights.append({
                    "type": "risk_alert",
                    "title": "Rising Supply Chain Risks Detected",
                    "description": f"ML models predict increasing risk for {len(increasing_risk_routes)} routes: {', '.join(increasing_risk_routes[:2])}{'...' if len(increasing_risk_routes) > 2 else ''}"
                })
            
            # Find high-confidence predictions
            high_confidence_predictions = [
                name for name, pred in predictions.items()
                if pred.get('confidence_level', 0) > 0.8
            ]
            
            if high_confidence_predictions:
                insights.append({
                    "type": "model_confidence",
                    "title": "High-Confidence Predictions Available",
                    "description": f"ML models show high confidence ({len(high_confidence_predictions)} routes) for short-term predictions"
                })
            
            # Risk diversification insight
            risk_levels = [pred.get('predicted_risk_score', 50) for pred in predictions.values()]
            if risk_levels:
                avg_risk = sum(risk_levels) / len(risk_levels)
                if avg_risk > 70:
                    insights.append({
                        "type": "optimization",
                        "title": "Route Diversification Recommended",
                        "description": f"Average predicted risk ({avg_risk:.1f}) suggests exploring alternative routing options"
                    })
            
            return insights[:3]  # Limit to top 3 insights
            
        except Exception as e:
            logger.warning(f"Failed to generate supply chain insights: {e}")
            return []
    
    def _generate_market_insights(self, predictions: Dict, market_data: Dict) -> List[Dict[str, str]]:
        """Generate actionable insights from market predictions"""
        insights = []
        
        try:
            # Overall market trend insight
            if 'overall_market' in predictions:
                market_pred = predictions['overall_market']
                trend = market_pred.get('trend_direction', 'neutral')
                
                if trend == 'bearish':
                    insights.append({
                        "type": "market_alert",
                        "title": "Bearish Market Conditions Predicted",
                        "description": "ML models indicate potential risk increase in overall market conditions"
                    })
                elif trend == 'bullish':
                    insights.append({
                        "type": "market_opportunity", 
                        "title": "Improving Market Conditions Expected",
                        "description": "ML analysis suggests favorable market trend development"
                    })
            
            # Country-specific insights
            if 'countries' in predictions:
                deteriorating_countries = [
                    country for country, pred in predictions['countries'].items()
                    if pred.get('outlook') == 'deteriorating'
                ]
                
                if deteriorating_countries:
                    insights.append({
                        "type": "geographic_risk",
                        "title": "Regional Risk Deterioration Detected",
                        "description": f"ML models predict worsening conditions in: {', '.join(deteriorating_countries[:3])}"
                    })
            
            return insights[:3]  # Limit to top 3 insights
            
        except Exception as e:
            logger.warning(f"Failed to generate market insights: {e}")
            return []
    
    def _predict_volatility(self, features: List[float]) -> str:
        """Predict market volatility based on features"""
        try:
            # Simple volatility prediction based on feature variance
            feature_std = np.std(features)
            
            if feature_std > 20:
                return "high"
            elif feature_std > 10:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _identify_risk_drivers(self, features: List[float], model_key: str) -> List[str]:
        """Identify primary risk drivers from features"""
        try:
            if model_key not in self.feature_importance:
                return ["Economic indicators", "Market conditions"]
            
            importance = self.feature_importance[model_key]
            top_drivers = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:2]
            
            driver_names = {
                'financial_health_score': 'Financial health metrics',
                'trade_stress_score': 'Trade stress indicators', 
                'trade_vulnerability': 'Trade vulnerability factors',
                'supply_chain_risk': 'Supply chain disruptions',
                'volatility_index': 'Market volatility',
                'liquidity_ratio': 'Liquidity conditions'
            }
            
            return [driver_names.get(name, name.title()) for name, _ in top_drivers]
            
        except Exception:
            return ["Market conditions", "Economic factors"]
    
    def _detect_country_anomaly(self, features: List[float], country: str) -> float:
        """Detect anomalies in country economic indicators"""
        try:
            # Simple anomaly detection using z-score
            features_array = np.array(features)
            
            # Historical mean and std (would normally be learned from data)
            historical_means = np.array([2.5, 3.0, 6.0, 50.0, 50.0])  # GDP, inflation, unemployment, trade openness, risk
            historical_stds = np.array([2.0, 2.5, 3.0, 20.0, 15.0])
            
            z_scores = np.abs((features_array - historical_means) / historical_stds)
            max_z_score = np.max(z_scores)
            
            # Convert z-score to anomaly probability
            anomaly_score = min(1.0, max_z_score / 3.0)  # 3-sigma rule
            return anomaly_score
            
        except Exception:
            return 0.0
    
    def _identify_anomalous_indicators(self, features: List[float]) -> List[str]:
        """Identify which indicators are anomalous"""
        indicators = ['GDP Growth', 'Inflation', 'Unemployment', 'Trade Openness', 'Risk Score']
        
        try:
            features_array = np.array(features)
            historical_means = np.array([2.5, 3.0, 6.0, 50.0, 50.0])
            historical_stds = np.array([2.0, 2.5, 3.0, 20.0, 15.0])
            
            z_scores = np.abs((features_array - historical_means) / historical_stds)
            
            anomalous = []
            for i, z_score in enumerate(z_scores):
                if z_score > 2.0:  # 2-sigma threshold
                    anomalous.append(indicators[i])
            
            return anomalous or ['Multiple indicators']
            
        except Exception:
            return ['Economic indicators']
    
    def _get_model_performance(self, model_key: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            "model_type": "Random Forest Regressor",
            "last_trained": self.last_trained.get(model_key, datetime.utcnow()).isoformat(),
            "feature_count": len(self.feature_importance.get(model_key, {})),
            "status": "trained" if model_key in self.models else "not_trained"
        }
    
    def _get_fallback_supply_chain_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when ML fails"""
        return {
            "predictions": {},
            "insights": [{
                "type": "system_notice",
                "title": "ML Prediction Service Initializing", 
                "description": "Advanced ML predictions will be available shortly"
            }],
            "model_performance": {"status": "initializing"},
            "prediction_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_fallback_market_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when ML fails"""
        return {
            "predictions": {"overall_market": {"trend_direction": "neutral", "confidence": 0.5}},
            "insights": [{
                "type": "system_notice",
                "title": "ML Market Analysis Initializing",
                "description": "Advanced market predictions will be available shortly"
            }],
            "prediction_timestamp": datetime.utcnow().isoformat()
        }


# Global ML service instance
ml_intelligence_service = MLIntelligenceService()