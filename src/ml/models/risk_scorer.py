import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskScoreComponents:
    """Risk score breakdown by category."""
    economic: float
    market: float
    geopolitical: float
    technical: float
    overall: float
    confidence: float


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mse: float
    mae: float
    r2: float
    mape: float
    accuracy_score: float


class RiskScorer:
    """
    Advanced risk scoring model using ensemble methods.
    Combines multiple economic indicators to generate comprehensive risk scores.
    """
    
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.model_version = "1.0.0"
        self.training_date = None
        
        # Risk scoring weights by category
        self.category_weights = {
            'economic': 0.35,
            'market': 0.30,
            'geopolitical': 0.20,
            'technical': 0.15
        }
        
        # Initialize models for each risk category
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for each risk category."""
        model_configs = {
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 8,
                'random_state': 42
            }
        }
        
        config = model_configs.get(self.model_type, model_configs['gradient_boosting'])
        
        for category in self.category_weights.keys():
            if self.model_type == 'gradient_boosting':
                self.models[category] = GradientBoostingRegressor(**config)
            else:
                self.models[category] = RandomForestRegressor(**config)
            
            self.scalers[category] = RobustScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for each risk category.
        
        Args:
            data: DataFrame with economic indicators and market data
            
        Returns:
            Dictionary of DataFrames for each risk category
        """
        features = {}
        
        # Economic risk features
        economic_cols = [
            'gdp_growth', 'unemployment_rate', 'inflation_rate', 
            'fed_funds_rate', 'consumer_confidence', 'retail_sales'
        ]
        features['economic'] = data[economic_cols].fillna(method='ffill')
        
        # Market risk features  
        market_cols = [
            'vix_volatility', 'sp500_returns', 'bond_yield_10y',
            'credit_spread', 'dollar_index', 'oil_price'
        ]
        features['market'] = data[market_cols].fillna(method='ffill')
        
        # Geopolitical risk features
        geopolitical_cols = [
            'geopolitical_index', 'trade_tension_index', 'policy_uncertainty',
            'election_cycle', 'international_conflicts'
        ]
        features['geopolitical'] = data[geopolitical_cols].fillna(method='ffill')
        
        # Technical risk features
        technical_cols = [
            'cyber_threat_level', 'supply_chain_disruption', 'infrastructure_risk',
            'technology_adoption_rate', 'digital_transformation_index'
        ]
        features['technical'] = data[technical_cols].fillna(method='ffill')
        
        return features
    
    def create_target_variables(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create target variables for each risk category.
        
        Args:
            data: DataFrame with historical risk scores
            
        Returns:
            Dictionary of target variables for each category
        """
        targets = {}
        
        # Use only real historical risk scores as targets
        for category in self.category_weights.keys():
            if f'{category}_risk_score' in data.columns:
                targets[category] = data[f'{category}_risk_score']
            else:
                # Require real historical data - no synthetic targets allowed
                raise ValueError(f"No historical risk scores available for {category}. Real historical data required.")
        
        return targets
    
    
    def train(self, data: pd.DataFrame) -> Dict[str, ModelMetrics]:
        """
        Train risk scoring models.
        
        Args:
            data: Training data with features and targets
            
        Returns:
            Dictionary of model metrics for each category
        """
        logger.info(f"Training risk scoring models with {len(data)} samples")
        
        features = self.prepare_features(data)
        targets = self.create_target_variables(data)
        
        metrics = {}
        
        for category in self.category_weights.keys():
            logger.info(f"Training {category} risk model")
            
            X = features[category]
            y = targets[category]
            
            # Remove rows with missing targets
            valid_idx = ~y.isna()
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            if len(X_clean) < 10:
                logger.warning(f"Insufficient data for {category} model: {len(X_clean)} samples")
                continue
            
            # Scale features
            X_scaled = self.scalers[category].fit_transform(X_clean)
            
            # Train model
            self.models[category].fit(X_scaled, y_clean)
            
            # Calculate metrics using time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(
                self.models[category], X_scaled, y_clean, 
                cv=tscv, scoring='neg_mean_squared_error'
            )
            
            # Predictions for metrics calculation
            y_pred = self.models[category].predict(X_scaled)
            
            metrics[category] = ModelMetrics(
                mse=mean_squared_error(y_clean, y_pred),
                mae=mean_absolute_error(y_clean, y_pred),
                r2=r2_score(y_clean, y_pred),
                mape=np.mean(np.abs((y_clean - y_pred) / y_clean)) * 100,
                accuracy_score=np.mean(-cv_scores)
            )
            
            # Store feature importance
            if hasattr(self.models[category], 'feature_importances_'):
                self.feature_importance[category] = dict(zip(
                    X.columns, self.models[category].feature_importances_
                ))
            
            logger.info(f"{category} model trained - R²: {metrics[category].r2:.3f}")
        
        self.is_trained = True
        self.training_date = datetime.utcnow()
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> RiskScoreComponents:
        """
        Generate risk scores for input data.
        
        Args:
            data: DataFrame with current feature values
            
        Returns:
            RiskScoreComponents with category and overall scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.prepare_features(data)
        category_scores = {}
        confidences = {}
        
        for category in self.category_weights.keys():
            if category not in self.models:
                continue
                
            X = features[category].iloc[-1:].fillna(0)  # Use most recent data
            X_scaled = self.scalers[category].transform(X)
            
            # Predict risk score
            score = self.models[category].predict(X_scaled)[0]
            category_scores[category] = np.clip(score, 0, 100)
            
            # Calculate confidence based on feature stability
            confidences[category] = self._calculate_confidence(X, category)
        
        # Calculate overall risk score
        overall_score = sum(
            category_scores.get(cat, 50) * weight 
            for cat, weight in self.category_weights.items()
        )
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(confidences.values()))
        
        return RiskScoreComponents(
            economic=category_scores.get('economic', 50),
            market=category_scores.get('market', 50),
            geopolitical=category_scores.get('geopolitical', 50),
            technical=category_scores.get('technical', 50),
            overall=overall_score,
            confidence=overall_confidence
        )
    
    def _calculate_confidence(self, features: pd.DataFrame, category: str) -> float:
        """Calculate prediction confidence based on feature quality."""
        
        # Check for missing values
        missing_ratio = features.isna().sum().sum() / features.size
        
        # Check for extreme values (outside 3 standard deviations)
        z_scores = np.abs((features - features.mean()) / features.std())
        extreme_ratio = (z_scores > 3).sum().sum() / features.size
        
        # Base confidence starts at 0.9
        confidence = 0.9
        
        # Reduce confidence for missing data
        confidence -= missing_ratio * 0.3
        
        # Reduce confidence for extreme values
        confidence -= extreme_ratio * 0.2
        
        # Reduce confidence if model is old
        if self.training_date:
            days_since_training = (datetime.utcnow() - self.training_date).days
            if days_since_training > 30:
                confidence -= min(days_since_training / 365, 0.3)
        
        return np.clip(confidence, 0.1, 1.0)
    
    def get_feature_importance(self, category: str) -> Dict[str, float]:
        """Get feature importance for a specific risk category."""
        return self.feature_importance.get(category, {})
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'category_weights': self.category_weights,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'training_date': self.training_date,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.category_weights = model_data['category_weights']
        self.model_type = model_data['model_type']
        self.model_version = model_data['model_version']
        self.training_date = model_data['training_date']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update category weights for overall risk calculation."""
        if abs(sum(new_weights.values()) - 1.0) > 0.01:
            raise ValueError("Category weights must sum to 1.0")
        
        self.category_weights = new_weights
        logger.info(f"Updated category weights: {new_weights}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'category_weights': self.category_weights,
            'trained_categories': list(self.models.keys()),
            'feature_counts': {
                cat: len(importance) for cat, importance in self.feature_importance.items()
            }
        }