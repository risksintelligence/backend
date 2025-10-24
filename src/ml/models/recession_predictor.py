"""
Economic Recession Probability Model
Advanced ML model for predicting recession probability using economic indicators
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class RecessionPrediction:
    """Recession prediction result"""
    probability: float
    confidence: float
    risk_level: str
    key_indicators: Dict[str, float]
    contributing_factors: List[Dict[str, Any]]
    model_version: str
    prediction_date: str
    time_horizon: str  # e.g., "3_months", "6_months", "12_months"


@dataclass
class EconomicIndicators:
    """Economic indicators for recession modeling"""
    # Yield curve indicators
    yield_curve_10y2y: Optional[float] = None
    yield_curve_10y3m: Optional[float] = None
    
    # Labor market indicators
    unemployment_rate: Optional[float] = None
    jobless_claims: Optional[float] = None
    nonfarm_payrolls_change: Optional[float] = None
    labor_participation_rate: Optional[float] = None
    
    # Economic output indicators
    gdp_growth_rate: Optional[float] = None
    industrial_production_change: Optional[float] = None
    manufacturing_pmi: Optional[float] = None
    
    # Financial indicators
    sp500_volatility: Optional[float] = None
    credit_spreads: Optional[float] = None
    consumer_confidence: Optional[float] = None
    
    # Inflation indicators
    cpi_change: Optional[float] = None
    core_cpi_change: Optional[float] = None
    
    # Leading indicators
    leading_economic_index: Optional[float] = None
    housing_starts_change: Optional[float] = None
    
    # Business cycle indicators
    ism_manufacturing: Optional[float] = None
    ism_services: Optional[float] = None


class RecessionPredictor:
    """
    Advanced ML model for predicting recession probability
    Uses ensemble methods and economic theory-based features
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_path = model_path or "models/recession_predictor"
        self.version = "1.0.0"
        
        # Economic theory-based feature weights
        self.feature_weights = {
            "yield_curve_10y2y": 0.20,      # Strong recession predictor
            "yield_curve_10y3m": 0.15,      # Another yield curve measure
            "unemployment_rate": 0.12,       # Key labor market indicator
            "gdp_growth_rate": 0.10,        # Economic output
            "sp500_volatility": 0.08,       # Market stress
            "manufacturing_pmi": 0.08,       # Manufacturing health
            "consumer_confidence": 0.07,     # Consumer sentiment
            "industrial_production_change": 0.06,  # Production activity
            "credit_spreads": 0.06,          # Credit market stress
            "leading_economic_index": 0.08   # Composite leading indicator
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of models for recession prediction"""
        
        # Model 1: Random Forest (handles non-linear relationships)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Model 2: Gradient Boosting (sequential learning)
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        
        # Model 3: Logistic Regression (interpretable baseline)
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        # Scalers for each model
        self.scalers['random_forest'] = RobustScaler()
        self.scalers['gradient_boosting'] = StandardScaler()
        self.scalers['logistic'] = StandardScaler()
    
    async def _create_recession_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data using real historical economic data from FRED API
        """
        from src.data.sources.fred import FREDClient
        from src.core.config import get_settings
        
        settings = get_settings()
        fred_client = FREDClient()
        
        # Fetch historical data from 1980 to present
        end_date = datetime.now()
        start_date = datetime(1980, 1, 1)
        
        try:
            # Fetch key economic indicators
            series_data = await fred_client.get_multiple_series(
                series_ids=[
                    'DGS10',    # 10-Year Treasury Rate
                    'DGS2',     # 2-Year Treasury Rate
                    'DGS3MO',   # 3-Month Treasury Rate
                    'UNRATE',   # Unemployment Rate
                    'GDP',      # GDP
                    'VIXCLS',   # VIX Volatility Index
                    'NAPMPREC', # Manufacturing PMI
                    'UMCSENT',  # Consumer Sentiment
                    'INDPRO',   # Industrial Production Index
                    'TEDRATE'   # TED Spread (credit spreads proxy)
                ],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Process the data into features and labels
            df = pd.DataFrame(series_data)
            
            # Calculate derived features
            df['yield_curve_10y2y'] = df['DGS10'] - df['DGS2']
            df['yield_curve_10y3m'] = df['DGS10'] - df['DGS3MO']
            df['gdp_growth_rate'] = df['GDP'].pct_change(periods=4) * 100  # YoY growth
            df['industrial_production_change'] = df['INDPRO'].pct_change(periods=12) * 100
            
            # Define recession periods based on NBER dates
            recession_periods = [
                ('1980-01-01', '1980-07-01'),
                ('1981-07-01', '1982-11-01'),
                ('1990-07-01', '1991-03-01'),
                ('2001-03-01', '2001-11-01'),
                ('2007-12-01', '2009-06-01'),
                ('2020-02-01', '2020-04-01')  # COVID recession
            ]
            
            # Create recession labels
            df['is_recession'] = 0
            for start, end in recession_periods:
                mask = (df.index >= start) & (df.index <= end)
                df.loc[mask, 'is_recession'] = 1
            
            # Select features for model
            feature_columns = [
                'yield_curve_10y2y', 'yield_curve_10y3m', 'UNRATE',
                'gdp_growth_rate', 'VIXCLS', 'NAPMPREC',
                'UMCSENT', 'industrial_production_change',
                'TEDRATE'
            ]
            
            # Remove rows with missing data
            df_clean = df[feature_columns + ['is_recession']].dropna()
            
            X = df_clean[feature_columns].values
            y = df_clean['is_recession'].values
            
            logger.info(f"Created training dataset with {len(X)} samples from real economic data")
            logger.info(f"Recession samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to fetch real data from FRED: {str(e)}")
            logger.error("Cannot use synthetic data - real data required for production")
            raise ValueError("Real economic data is required for model training")
    
    async def train_models(self) -> Dict[str, float]:
        """Train all models in the ensemble"""
        
        # Get training data from real economic APIs
        X, y = await self._create_recession_training_data()
        
        # Feature names for interpretation
        feature_names = [
            'yield_curve_10y2y', 'yield_curve_10y3m', 'unemployment_rate',
            'gdp_growth_rate', 'sp500_volatility', 'manufacturing_pmi',
            'consumer_confidence', 'industrial_production_change',
            'credit_spreads', 'leading_economic_index'
        ]
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
            
            # Train final model on all data
            model.fit(X_scaled, y)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = importance
            
            results[model_name] = {
                'cv_mean_auc': cv_scores.mean(),
                'cv_std_auc': cv_scores.std(),
                'final_train_score': model.score(X_scaled, y)
            }
            
            logger.info(f"{model_name} - CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def _prepare_features(self, indicators: EconomicIndicators) -> np.ndarray:
        """Prepare features from economic indicators"""
        
        # Extract features in consistent order
        features = [
            indicators.yield_curve_10y2y or 0.0,
            indicators.yield_curve_10y3m or 0.0,
            indicators.unemployment_rate or 0.0,
            indicators.gdp_growth_rate or 0.0,
            indicators.sp500_volatility or 0.0,
            indicators.manufacturing_pmi or 50.0,  # Default to neutral
            indicators.consumer_confidence or 100.0,  # Default to neutral
            indicators.industrial_production_change or 0.0,
            indicators.credit_spreads or 150.0,  # Default spread
            indicators.leading_economic_index or 0.0
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_recession_probability(
        self, 
        indicators: EconomicIndicators,
        time_horizon: str = "12_months"
    ) -> RecessionPrediction:
        """Predict recession probability using ensemble of models"""
        
        # Prepare features
        X = self._prepare_features(indicators)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            
            # Get prediction
            prob = model.predict_proba(X_scaled)[0, 1]  # Probability of recession
            pred = model.predict(X_scaled)[0]
            
            predictions[model_name] = pred
            probabilities[model_name] = prob
        
        # Ensemble prediction (weighted average)
        model_weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'logistic': 0.2}
        ensemble_probability = sum(
            probabilities[model] * weight 
            for model, weight in model_weights.items()
        )
        
        # Calculate confidence (based on model agreement)
        prob_std = np.std(list(probabilities.values()))
        confidence = max(0.5, 1.0 - (prob_std * 2))  # Higher agreement = higher confidence
        
        # Determine risk level
        if ensemble_probability >= 0.7:
            risk_level = "Critical"
        elif ensemble_probability >= 0.5:
            risk_level = "High"
        elif ensemble_probability >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Identify key contributing factors
        contributing_factors = self._identify_contributing_factors(indicators, ensemble_probability)
        
        # Key indicators dictionary
        key_indicators = {
            "yield_curve": indicators.yield_curve_10y2y or 0.0,
            "unemployment": indicators.unemployment_rate or 0.0,
            "gdp_growth": indicators.gdp_growth_rate or 0.0,
            "market_volatility": indicators.sp500_volatility or 0.0,
            "manufacturing_pmi": indicators.manufacturing_pmi or 50.0
        }
        
        return RecessionPrediction(
            probability=ensemble_probability,
            confidence=confidence,
            risk_level=risk_level,
            key_indicators=key_indicators,
            contributing_factors=contributing_factors,
            model_version=self.version,
            prediction_date=datetime.utcnow().isoformat(),
            time_horizon=time_horizon
        )
    
    def _identify_contributing_factors(
        self, 
        indicators: EconomicIndicators, 
        probability: float
    ) -> List[Dict[str, Any]]:
        """Identify which factors are contributing most to recession risk"""
        
        factors = []
        
        # Yield curve analysis
        if indicators.yield_curve_10y2y is not None and indicators.yield_curve_10y2y < 0:
            factors.append({
                "factor": "Inverted Yield Curve",
                "value": indicators.yield_curve_10y2y,
                "impact": "high",
                "description": "10Y-2Y yield curve is inverted, historically a strong recession predictor"
            })
        
        # Labor market stress
        if indicators.unemployment_rate is not None and indicators.unemployment_rate > 6.0:
            factors.append({
                "factor": "Rising Unemployment",
                "value": indicators.unemployment_rate,
                "impact": "medium",
                "description": f"Unemployment rate at {indicators.unemployment_rate}% indicates labor market stress"
            })
        
        # Economic contraction
        if indicators.gdp_growth_rate is not None and indicators.gdp_growth_rate < 0:
            factors.append({
                "factor": "GDP Contraction",
                "value": indicators.gdp_growth_rate,
                "impact": "high",
                "description": f"GDP declining at {indicators.gdp_growth_rate}% annualized rate"
            })
        
        # Manufacturing weakness
        if indicators.manufacturing_pmi is not None and indicators.manufacturing_pmi < 50:
            factors.append({
                "factor": "Manufacturing Contraction",
                "value": indicators.manufacturing_pmi,
                "impact": "medium",
                "description": f"Manufacturing PMI below 50 ({indicators.manufacturing_pmi}) indicates contraction"
            })
        
        # Market stress
        if indicators.sp500_volatility is not None and indicators.sp500_volatility > 25:
            factors.append({
                "factor": "Market Volatility",
                "value": indicators.sp500_volatility,
                "impact": "medium",
                "description": f"S&P 500 volatility at {indicators.sp500_volatility}% indicates market stress"
            })
        
        return factors
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        performance = {
            "model_version": self.version,
            "ensemble_composition": {
                "random_forest": 0.4,
                "gradient_boosting": 0.4,
                "logistic_regression": 0.2
            },
            "feature_importance": self.feature_importance,
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
            "feature_weights": self.feature_weights
        }
        joblib.dump(metadata, f"{save_path}/metadata.pkl")
        
        logger.info(f"Models saved to {save_path}")
    
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
            
            logger.info(f"Models loaded from {load_path}")
            
        except FileNotFoundError:
            logger.warning(f"No saved models found at {load_path}, using default models")


# Convenience functions for integration
async def predict_recession_risk(indicators: Dict[str, float]) -> Dict[str, Any]:
    """Predict recession risk from economic indicators dictionary"""
    
    # Convert dictionary to EconomicIndicators object
    economic_indicators = EconomicIndicators(
        yield_curve_10y2y=indicators.get("yield_curve_10y2y"),
        yield_curve_10y3m=indicators.get("yield_curve_10y3m"),
        unemployment_rate=indicators.get("unemployment_rate"),
        gdp_growth_rate=indicators.get("gdp_growth_rate"),
        sp500_volatility=indicators.get("sp500_volatility"),
        manufacturing_pmi=indicators.get("manufacturing_pmi"),
        consumer_confidence=indicators.get("consumer_confidence"),
        industrial_production_change=indicators.get("industrial_production_change"),
        credit_spreads=indicators.get("credit_spreads"),
        leading_economic_index=indicators.get("leading_economic_index")
    )
    
    # Create predictor and get prediction
    predictor = RecessionPredictor()
    
    # Train models if not already trained (in production, load pre-trained models)
    try:
        predictor.load_models()
    except:
        logger.info("Training recession prediction models...")
        predictor.train_models()
        predictor.save_models()
    
    prediction = predictor.predict_recession_probability(economic_indicators)
    
    return {
        "recession_probability": prediction.probability,
        "confidence": prediction.confidence,
        "risk_level": prediction.risk_level,
        "key_indicators": prediction.key_indicators,
        "contributing_factors": prediction.contributing_factors,
        "model_version": prediction.model_version,
        "prediction_date": prediction.prediction_date,
        "time_horizon": prediction.time_horizon
    }


async def train_recession_models() -> Dict[str, Any]:
    """Train recession prediction models using real economic data"""
    
    predictor = RecessionPredictor()
    training_results = await predictor.train_models()
    predictor.save_models()
    
    return {
        "status": "completed",
        "training_results": training_results,
        "model_version": predictor.version,
        "models_trained": list(predictor.models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }