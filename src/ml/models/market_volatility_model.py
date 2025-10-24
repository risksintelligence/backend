"""
Financial Market Volatility Model
Advanced ML model for predicting and analyzing financial market volatility
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VolatilityPrediction:
    """Market volatility prediction result"""
    predicted_volatility: float
    current_volatility: float
    volatility_trend: str  # "increasing", "decreasing", "stable"
    risk_level: str
    confidence: float
    contributing_factors: List[Dict[str, Any]]
    volatility_regime: str  # "low", "medium", "high", "extreme"
    stress_indicators: Dict[str, float]
    time_horizon: str
    prediction_date: str
    model_version: str


@dataclass
class MarketIndicators:
    """Market indicators for volatility modeling"""
    # Price-based indicators
    current_price: Optional[float] = None
    price_returns_1d: Optional[float] = None
    price_returns_5d: Optional[float] = None
    price_returns_20d: Optional[float] = None
    
    # Volatility measures
    realized_volatility_5d: Optional[float] = None
    realized_volatility_20d: Optional[float] = None
    garch_volatility: Optional[float] = None
    
    # Volume and liquidity
    volume_ratio: Optional[float] = None  # Current vs average volume
    bid_ask_spread: Optional[float] = None
    market_depth: Optional[float] = None
    
    # Options market
    vix_level: Optional[float] = None
    put_call_ratio: Optional[float] = None
    options_skew: Optional[float] = None
    
    # Cross-market indicators
    bond_yield_10y: Optional[float] = None
    bond_yield_volatility: Optional[float] = None
    fx_volatility: Optional[float] = None
    commodity_volatility: Optional[float] = None
    
    # Economic uncertainty
    economic_policy_uncertainty: Optional[float] = None
    geopolitical_risk_index: Optional[float] = None
    credit_spreads: Optional[float] = None
    
    # Market microstructure
    intraday_range: Optional[float] = None
    overnight_gap: Optional[float] = None
    number_of_trades: Optional[float] = None


class MarketVolatilityModel:
    """
    Advanced ML model for financial market volatility prediction
    Uses ensemble methods and market microstructure indicators
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.volatility_regimes = {}
        self.model_path = model_path or "models/market_volatility"
        self.version = "1.0.0"
        
        # Volatility regime thresholds (annualized)
        self.regime_thresholds = {
            "low": 0.15,      # < 15% annualized
            "medium": 0.25,   # 15-25% annualized
            "high": 0.40,     # 25-40% annualized
            "extreme": 1.0    # > 40% annualized
        }
        
        # Feature importance weights based on volatility research
        self.feature_weights = {
            "realized_volatility_20d": 0.20,
            "vix_level": 0.18,
            "price_returns_1d": 0.12,
            "volume_ratio": 0.10,
            "put_call_ratio": 0.08,
            "credit_spreads": 0.08,
            "bond_yield_volatility": 0.07,
            "options_skew": 0.06,
            "intraday_range": 0.06,
            "geopolitical_risk": 0.05
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of models for volatility prediction"""
        
        # Model 1: Random Forest (captures non-linear patterns)
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Model 2: Gradient Boosting (sequential learning)
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        # Model 3: Neural Network (captures complex patterns)
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Scalers for each model
        self.scalers['random_forest'] = RobustScaler()
        self.scalers['gradient_boosting'] = StandardScaler()
        self.scalers['neural_network'] = StandardScaler()
    
    def _generate_volatility_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on market volatility patterns
        In production, this would use real market data
        """
        np.random.seed(42)
        n_samples = 2000
        
        features = []
        target_volatilities = []
        
        for i in range(n_samples):
            # Simulate different market regimes
            regime = np.random.choice(['low_vol', 'medium_vol', 'high_vol', 'crisis'], 
                                    p=[0.40, 0.35, 0.20, 0.05])
            
            if regime == 'low_vol':
                # Low volatility regime (normal markets)
                base_vol = np.random.uniform(0.08, 0.15)
                vix = np.random.normal(15, 3)
                returns_1d = np.random.normal(0, 0.008)
                volume_ratio = np.random.normal(1.0, 0.2)
                put_call_ratio = np.random.normal(0.8, 0.1)
                credit_spreads = np.random.normal(100, 20)
                
            elif regime == 'medium_vol':
                # Medium volatility regime
                base_vol = np.random.uniform(0.15, 0.25)
                vix = np.random.normal(20, 4)
                returns_1d = np.random.normal(0, 0.015)
                volume_ratio = np.random.normal(1.2, 0.3)
                put_call_ratio = np.random.normal(0.9, 0.15)
                credit_spreads = np.random.normal(150, 30)
                
            elif regime == 'high_vol':
                # High volatility regime
                base_vol = np.random.uniform(0.25, 0.40)
                vix = np.random.normal(30, 5)
                returns_1d = np.random.normal(0, 0.025)
                volume_ratio = np.random.normal(1.5, 0.4)
                put_call_ratio = np.random.normal(1.1, 0.2)
                credit_spreads = np.random.normal(200, 40)
                
            else:  # crisis
                # Crisis regime (extreme volatility)
                base_vol = np.random.uniform(0.40, 0.80)
                vix = np.random.normal(45, 8)
                returns_1d = np.random.normal(0, 0.040)
                volume_ratio = np.random.normal(2.0, 0.5)
                put_call_ratio = np.random.normal(1.3, 0.25)
                credit_spreads = np.random.normal(300, 60)
            
            # Additional features
            realized_vol_5d = base_vol + np.random.normal(0, 0.02)
            realized_vol_20d = base_vol + np.random.normal(0, 0.015)
            price_returns_5d = returns_1d * 5 + np.random.normal(0, 0.01)
            price_returns_20d = returns_1d * 20 + np.random.normal(0, 0.02)
            
            # Market microstructure
            bid_ask_spread = np.random.exponential(0.001) if regime != 'crisis' else np.random.exponential(0.003)
            intraday_range = base_vol * np.random.uniform(0.8, 1.2)
            overnight_gap = np.random.normal(0, base_vol * 0.3)
            
            # Cross-market indicators
            bond_yield_10y = np.random.normal(3.0, 0.5)
            bond_yield_vol = base_vol * np.random.uniform(0.5, 1.5)
            fx_volatility = base_vol * np.random.uniform(0.7, 1.3)
            
            # Options market
            options_skew = np.random.normal(-0.1, 0.05) if regime != 'crisis' else np.random.normal(-0.2, 0.08)
            
            # Economic uncertainty
            epu_index = np.random.normal(100, 20) if regime == 'low_vol' else np.random.normal(150, 30)
            geopolitical_risk = np.random.beta(2, 5) if regime != 'crisis' else np.random.beta(5, 2)
            
            sample_features = [
                returns_1d, price_returns_5d, price_returns_20d,
                realized_vol_5d, realized_vol_20d, vix,
                volume_ratio, bid_ask_spread, put_call_ratio,
                bond_yield_10y, bond_yield_vol, fx_volatility,
                options_skew, intraday_range, overnight_gap,
                epu_index, geopolitical_risk, credit_spreads
            ]
            
            # Target volatility (next period)
            # Add some persistence and mean reversion
            next_vol = base_vol * 0.8 + realized_vol_20d * 0.2 + np.random.normal(0, 0.01)
            next_vol = max(0.05, min(1.0, next_vol))  # Clamp between 5% and 100%
            
            features.append(sample_features)
            target_volatilities.append(next_vol)
        
        return np.array(features), np.array(target_volatilities)
    
    def train_models(self) -> Dict[str, float]:
        """Train all models in the ensemble"""
        
        # Get training data
        X, y = self._generate_volatility_training_data()
        
        # Feature names
        feature_names = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'realized_vol_5d', 'realized_vol_20d', 'vix',
            'volume_ratio', 'bid_ask_spread', 'put_call_ratio',
            'bond_yield_10y', 'bond_yield_vol', 'fx_volatility',
            'options_skew', 'intraday_range', 'overnight_gap',
            'epu_index', 'geopolitical_risk', 'credit_spreads'
        ]
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} volatility model...")
            
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error'
            )
            
            # Train final model
            model.fit(X_scaled, y)
            
            # Predictions for evaluation
            y_pred = model.predict(X_scaled)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = importance
            
            results[model_name] = {
                'cv_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std': cv_scores.std(),
                'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'train_r2': r2_score(y, y_pred),
                'train_mae': mean_absolute_error(y, y_pred)
            }
            
            logger.info(f"{model_name} - CV RMSE: {np.sqrt(-cv_scores.mean()):.4f}")
        
        return results
    
    def _prepare_features(self, indicators: MarketIndicators) -> np.ndarray:
        """Prepare features from market indicators"""
        
        features = [
            indicators.price_returns_1d or 0.0,
            indicators.price_returns_5d or 0.0,
            indicators.price_returns_20d or 0.0,
            indicators.realized_volatility_5d or 0.15,
            indicators.realized_volatility_20d or 0.15,
            indicators.vix_level or 20.0,
            indicators.volume_ratio or 1.0,
            indicators.bid_ask_spread or 0.001,
            indicators.put_call_ratio or 0.8,
            indicators.bond_yield_10y or 3.0,
            indicators.bond_yield_volatility or 0.05,
            indicators.fx_volatility or 0.10,
            indicators.options_skew or -0.1,
            indicators.intraday_range or 0.02,
            indicators.overnight_gap or 0.0,
            indicators.economic_policy_uncertainty or 100.0,
            indicators.geopolitical_risk_index or 0.3,
            indicators.credit_spreads or 150.0
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_volatility(
        self, 
        indicators: MarketIndicators,
        time_horizon: str = "1_day"
    ) -> VolatilityPrediction:
        """Predict market volatility using ensemble of models"""
        
        # Prepare features
        X = self._prepare_features(indicators)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            
            # Get prediction
            volatility_pred = model.predict(X_scaled)[0]
            predictions[model_name] = max(0.01, min(1.0, volatility_pred))  # Clamp values
        
        # Ensemble prediction (weighted average)
        model_weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'neural_network': 0.2}
        ensemble_volatility = sum(
            predictions[model] * weight 
            for model, weight in model_weights.items()
        )
        
        # Current volatility (from indicators)
        current_vol = indicators.realized_volatility_20d or 0.15
        
        # Determine volatility trend
        vol_change = ensemble_volatility - current_vol
        if vol_change > 0.02:
            volatility_trend = "increasing"
        elif vol_change < -0.02:
            volatility_trend = "decreasing"
        else:
            volatility_trend = "stable"
        
        # Determine volatility regime
        volatility_regime = self._classify_volatility_regime(ensemble_volatility)
        
        # Determine risk level
        if ensemble_volatility >= 0.40:
            risk_level = "Critical"
        elif ensemble_volatility >= 0.25:
            risk_level = "High"
        elif ensemble_volatility >= 0.15:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate confidence (based on model agreement)
        pred_std = np.std(list(predictions.values()))
        confidence = max(0.6, 1.0 - (pred_std * 3))  # Higher agreement = higher confidence
        
        # Identify contributing factors
        contributing_factors = self._identify_volatility_drivers(indicators, ensemble_volatility)
        
        # Calculate stress indicators
        stress_indicators = self._calculate_stress_indicators(indicators)
        
        return VolatilityPrediction(
            predicted_volatility=ensemble_volatility,
            current_volatility=current_vol,
            volatility_trend=volatility_trend,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            volatility_regime=volatility_regime,
            stress_indicators=stress_indicators,
            time_horizon=time_horizon,
            prediction_date=datetime.utcnow().isoformat(),
            model_version=self.version
        )
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility into regime categories"""
        
        if volatility < self.regime_thresholds["low"]:
            return "low"
        elif volatility < self.regime_thresholds["medium"]:
            return "medium"
        elif volatility < self.regime_thresholds["high"]:
            return "high"
        else:
            return "extreme"
    
    def _identify_volatility_drivers(
        self, 
        indicators: MarketIndicators, 
        predicted_vol: float
    ) -> List[Dict[str, Any]]:
        """Identify key factors driving volatility"""
        
        drivers = []
        
        # VIX level
        if indicators.vix_level and indicators.vix_level > 25:
            drivers.append({
                "factor": "Elevated VIX",
                "value": indicators.vix_level,
                "impact": "high",
                "description": f"VIX at {indicators.vix_level:.1f} indicates fear in options market"
            })
        
        # Recent price moves
        if indicators.price_returns_1d and abs(indicators.price_returns_1d) > 0.03:
            drivers.append({
                "factor": "Large Price Movement",
                "value": indicators.price_returns_1d,
                "impact": "medium",
                "description": f"Recent 1-day return of {indicators.price_returns_1d:.2%} indicates volatility"
            })
        
        # Volume surge
        if indicators.volume_ratio and indicators.volume_ratio > 1.5:
            drivers.append({
                "factor": "Volume Surge",
                "value": indicators.volume_ratio,
                "impact": "medium",
                "description": f"Trading volume {indicators.volume_ratio:.1f}x normal levels"
            })
        
        # Put/call ratio
        if indicators.put_call_ratio and indicators.put_call_ratio > 1.0:
            drivers.append({
                "factor": "High Put/Call Ratio",
                "value": indicators.put_call_ratio,
                "impact": "medium",
                "description": "Increased put buying suggests bearish sentiment"
            })
        
        # Credit stress
        if indicators.credit_spreads and indicators.credit_spreads > 200:
            drivers.append({
                "factor": "Credit Market Stress",
                "value": indicators.credit_spreads,
                "impact": "high",
                "description": f"Credit spreads at {indicators.credit_spreads} bps indicate stress"
            })
        
        # Geopolitical risk
        if indicators.geopolitical_risk_index and indicators.geopolitical_risk_index > 0.6:
            drivers.append({
                "factor": "Geopolitical Uncertainty",
                "value": indicators.geopolitical_risk_index,
                "impact": "medium",
                "description": "Elevated geopolitical risk contributing to uncertainty"
            })
        
        return drivers
    
    def _calculate_stress_indicators(self, indicators: MarketIndicators) -> Dict[str, float]:
        """Calculate various market stress indicators"""
        
        stress_indicators = {}
        
        # VIX stress (normalized 0-1)
        if indicators.vix_level:
            stress_indicators["vix_stress"] = min(1.0, max(0.0, (indicators.vix_level - 10) / 40))
        
        # Credit stress
        if indicators.credit_spreads:
            stress_indicators["credit_stress"] = min(1.0, max(0.0, (indicators.credit_spreads - 50) / 250))
        
        # Liquidity stress
        if indicators.bid_ask_spread:
            stress_indicators["liquidity_stress"] = min(1.0, max(0.0, (indicators.bid_ask_spread - 0.001) / 0.005))
        
        # Options market stress
        if indicators.put_call_ratio:
            stress_indicators["options_stress"] = min(1.0, max(0.0, (indicators.put_call_ratio - 0.6) / 0.8))
        
        # Volume stress
        if indicators.volume_ratio:
            stress_indicators["volume_stress"] = min(1.0, max(0.0, (indicators.volume_ratio - 1.0) / 2.0))
        
        # Composite stress index
        if stress_indicators:
            stress_indicators["composite_stress"] = np.mean(list(stress_indicators.values()))
        
        return stress_indicators
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        performance = {
            "model_version": self.version,
            "ensemble_composition": {
                "random_forest": 0.4,
                "gradient_boosting": 0.4,
                "neural_network": 0.2
            },
            "feature_importance": self.feature_importance,
            "volatility_regimes": self.regime_thresholds,
            "last_trained": datetime.utcnow().isoformat()
        }
        
        return performance
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models to disk"""
        save_path = path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save models and scalers
        for model_name, model in self.models.items():
            joblib.dump(model, f"{save_path}/{model_name}_model.pkl")
            joblib.dump(self.scalers[model_name], f"{save_path}/{model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            "version": self.version,
            "feature_importance": self.feature_importance,
            "feature_weights": self.feature_weights,
            "regime_thresholds": self.regime_thresholds
        }
        joblib.dump(metadata, f"{save_path}/metadata.pkl")
        
        logger.info(f"Volatility models saved to {save_path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load trained models from disk"""
        load_path = path or self.model_path
        
        try:
            # Load models and scalers
            for model_name in self.models.keys():
                self.models[model_name] = joblib.load(f"{load_path}/{model_name}_model.pkl")
                self.scalers[model_name] = joblib.load(f"{load_path}/{model_name}_scaler.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{load_path}/metadata.pkl")
            self.version = metadata.get("version", self.version)
            self.feature_importance = metadata.get("feature_importance", {})
            self.feature_weights = metadata.get("feature_weights", self.feature_weights)
            self.regime_thresholds = metadata.get("regime_thresholds", self.regime_thresholds)
            
            logger.info(f"Volatility models loaded from {load_path}")
            
        except FileNotFoundError:
            logger.warning(f"No saved models found at {load_path}, using default models")


# Convenience functions for integration
async def predict_market_volatility(indicators: Dict[str, float]) -> Dict[str, Any]:
    """Predict market volatility from indicators dictionary"""
    
    # Convert dictionary to MarketIndicators object
    market_indicators = MarketIndicators(
        price_returns_1d=indicators.get("price_returns_1d"),
        price_returns_5d=indicators.get("price_returns_5d"),
        price_returns_20d=indicators.get("price_returns_20d"),
        realized_volatility_5d=indicators.get("realized_volatility_5d"),
        realized_volatility_20d=indicators.get("realized_volatility_20d"),
        vix_level=indicators.get("vix_level"),
        volume_ratio=indicators.get("volume_ratio"),
        bid_ask_spread=indicators.get("bid_ask_spread"),
        put_call_ratio=indicators.get("put_call_ratio"),
        bond_yield_10y=indicators.get("bond_yield_10y"),
        bond_yield_volatility=indicators.get("bond_yield_volatility"),
        fx_volatility=indicators.get("fx_volatility"),
        options_skew=indicators.get("options_skew"),
        intraday_range=indicators.get("intraday_range"),
        overnight_gap=indicators.get("overnight_gap"),
        economic_policy_uncertainty=indicators.get("economic_policy_uncertainty"),
        geopolitical_risk_index=indicators.get("geopolitical_risk_index"),
        credit_spreads=indicators.get("credit_spreads")
    )
    
    # Create model and get prediction
    model = MarketVolatilityModel()
    
    # Train models if not already trained
    try:
        model.load_models()
    except:
        logger.info("Training market volatility models...")
        model.train_models()
        model.save_models()
    
    prediction = model.predict_volatility(market_indicators)
    
    return {
        "predicted_volatility": prediction.predicted_volatility,
        "current_volatility": prediction.current_volatility,
        "volatility_trend": prediction.volatility_trend,
        "risk_level": prediction.risk_level,
        "confidence": prediction.confidence,
        "contributing_factors": prediction.contributing_factors,
        "volatility_regime": prediction.volatility_regime,
        "stress_indicators": prediction.stress_indicators,
        "time_horizon": prediction.time_horizon,
        "prediction_date": prediction.prediction_date,
        "model_version": prediction.model_version
    }


async def train_volatility_models() -> Dict[str, Any]:
    """Train market volatility models"""
    
    model = MarketVolatilityModel()
    training_results = model.train_models()
    model.save_models()
    
    return {
        "status": "completed",
        "training_results": training_results,
        "model_version": model.version,
        "models_trained": list(model.models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }