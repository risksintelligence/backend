import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Risk prediction result with confidence intervals."""
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    prediction_date: datetime
    horizon_days: int
    confidence_score: float
    model_used: str


class RiskPredictor:
    """
    Risk prediction model for forecasting future risk levels.
    Uses time series analysis and ensemble methods.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.training_date = None
        self.prediction_horizons = [1, 7, 30, 90]  # Days ahead
        
    def prepare_time_series_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create time series features for prediction."""
        
        # Sort by date
        data = data.sort_values('date')
        
        features = pd.DataFrame(index=data.index)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            features[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            features[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window).mean()
            features[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window).std()
            features[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window).min()
            features[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window).max()
        
        # Trend features
        features[f'{target_col}_diff_1'] = data[target_col].diff(1)
        features[f'{target_col}_diff_7'] = data[target_col].diff(7)
        features[f'{target_col}_pct_change_1'] = data[target_col].pct_change(1)
        features[f'{target_col}_pct_change_7'] = data[target_col].pct_change(7)
        
        # Seasonal features
        if 'date' in data.columns:
            features['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
            features['month'] = pd.to_datetime(data['date']).dt.month
            features['quarter'] = pd.to_datetime(data['date']).dt.quarter
        
        return features.fillna(method='ffill').fillna(0)
    
    def train(self, data: pd.DataFrame, target_col: str = 'overall_score') -> Dict[int, float]:
        """
        Train prediction models for different horizons.
        
        Args:
            data: Historical data with risk scores
            target_col: Column name for target variable
            
        Returns:
            Dictionary of model performance scores by horizon
        """
        logger.info(f"Training risk prediction models with {len(data)} samples")
        
        # Prepare features
        features = self.prepare_time_series_features(data, target_col)
        target = data[target_col]
        
        performance = {}
        
        for horizon in self.prediction_horizons:
            logger.info(f"Training model for {horizon}-day horizon")
            
            # Create target shifted by horizon
            y_shifted = target.shift(-horizon)
            
            # Remove rows with missing targets
            valid_idx = ~(y_shifted.isna() | features.isna().any(axis=1))
            X_train = features[valid_idx]
            y_train = y_shifted[valid_idx]
            
            if len(X_train) < 50:
                logger.warning(f"Insufficient data for {horizon}-day model: {len(X_train)} samples")
                continue
            
            # Split into train/test (time series split)
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_idx]
            X_test_split = X_train.iloc[split_idx:]
            y_train_split = y_train.iloc[:split_idx]
            y_test_split = y_train.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_test_scaled = scaler.transform(X_test_split)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train_split)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test_split, y_pred)
            
            # Store model and scaler
            self.models[horizon] = model
            self.scalers[horizon] = scaler
            performance[horizon] = mae
            
            logger.info(f"{horizon}-day model trained - MAE: {mae:.3f}")
        
        self.is_trained = True
        self.training_date = datetime.utcnow()
        
        return performance
    
    def predict(self, data: pd.DataFrame, target_col: str = 'overall_score', 
                horizon: int = 7) -> PredictionResult:
        """
        Make risk prediction for specified horizon.
        
        Args:
            data: Recent data for prediction
            target_col: Target column name
            horizon: Days ahead to predict
            
        Returns:
            PredictionResult with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if horizon not in self.models:
            raise ValueError(f"No model trained for {horizon}-day horizon")
        
        # Prepare features
        features = self.prepare_time_series_features(data, target_col)
        
        # Use most recent data point
        X = features.iloc[-1:].fillna(0)
        X_scaled = self.scalers[horizon].transform(X)
        
        # Make prediction
        model = self.models[horizon]
        predicted_value = model.predict(X_scaled)[0]
        
        # Calculate confidence intervals using ensemble predictions
        if hasattr(model, 'estimators_'):
            # Get predictions from individual trees
            tree_predictions = [
                tree.predict(X_scaled)[0] for tree in model.estimators_
            ]
            
            # Calculate confidence intervals (5th and 95th percentiles)
            confidence_lower = np.percentile(tree_predictions, 5)
            confidence_upper = np.percentile(tree_predictions, 95)
            
            # Calculate confidence score based on prediction variance
            prediction_std = np.std(tree_predictions)
            confidence_score = max(0.1, 1.0 - (prediction_std / 50.0))  # Normalize to 0-1
        else:
            # Fallback confidence calculation
            confidence_lower = predicted_value * 0.9
            confidence_upper = predicted_value * 1.1
            confidence_score = 0.7
        
        return PredictionResult(
            predicted_value=np.clip(predicted_value, 0, 100),
            confidence_lower=np.clip(confidence_lower, 0, 100),
            confidence_upper=np.clip(confidence_upper, 0, 100),
            prediction_date=datetime.utcnow(),
            horizon_days=horizon,
            confidence_score=confidence_score,
            model_used=f"RandomForest_{horizon}d"
        )
    
    def predict_multiple_horizons(self, data: pd.DataFrame, 
                                target_col: str = 'overall_score') -> Dict[int, PredictionResult]:
        """Make predictions for all trained horizons."""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            if horizon in self.models:
                try:
                    predictions[horizon] = self.predict(data, target_col, horizon)
                except Exception as e:
                    logger.error(f"Error predicting {horizon}-day horizon: {e}")
        
        return predictions
    
    def get_feature_importance(self, horizon: int) -> Dict[str, float]:
        """Get feature importance for specific horizon model."""
        if horizon not in self.models:
            return {}
        
        model = self.models[horizon]
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            return dict(zip(feature_names, model.feature_importances_))
        
        return {}
    
    def calculate_prediction_accuracy(self, data: pd.DataFrame, 
                                    target_col: str = 'overall_score') -> Dict[int, Dict[str, float]]:
        """Calculate accuracy metrics for all horizons."""
        if not self.is_trained:
            return {}
        
        accuracy = {}
        features = self.prepare_time_series_features(data, target_col)
        target = data[target_col]
        
        for horizon in self.models.keys():
            # Create shifted target
            y_shifted = target.shift(-horizon)
            
            # Get valid data
            valid_idx = ~(y_shifted.isna() | features.isna().any(axis=1))
            X_valid = features[valid_idx]
            y_valid = y_shifted[valid_idx]
            
            if len(X_valid) < 10:
                continue
            
            # Make predictions
            X_scaled = self.scalers[horizon].transform(X_valid)
            y_pred = self.models[horizon].predict(X_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_valid, y_pred)
            mse = mean_squared_error(y_valid, y_pred)
            mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
            
            # Direction accuracy (did we predict the right trend?)
            y_diff_actual = np.diff(y_valid)
            y_diff_pred = np.diff(y_pred)
            direction_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(y_diff_pred))
            
            accuracy[horizon] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mape': mape,
                'direction_accuracy': direction_accuracy
            }
        
        return accuracy
    
    def update_model(self, new_data: pd.DataFrame, target_col: str = 'overall_score'):
        """Update models with new data (incremental learning)."""
        if not self.is_trained:
            logger.warning("Cannot update untrained model. Use train() first.")
            return
        
        logger.info(f"Updating models with {len(new_data)} new samples")
        
        # For now, retrain with all data
        # In production, implement true incremental learning
        self.train(new_data, target_col)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'prediction_horizons': self.prediction_horizons,
            'trained_horizons': list(self.models.keys()),
            'model_type': 'RandomForestRegressor',
            'features_count': len(self.scalers[self.prediction_horizons[0]].feature_names_in_) if self.scalers else 0
        }