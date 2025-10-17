"""
Advanced ML-based risk prediction models using Random Forest and ensemble methods.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

from src.ml.features.economic import EconomicFeatureEngineer
from src.ml.features.financial import FinancialFeatureEngineer
from src.ml.features.supply_chain import SupplyChainFeatureEngineer
from src.ml.features.disruption import DisruptionFeatureEngineer
# from src.ml.evaluation.metrics import RiskMetrics  # Will implement later
from src.cache.cache_manager import CacheManager
from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    model_name: str
    model_type: str
    version: str
    trained_at: datetime
    features_used: List[str]
    performance_metrics: Dict[str, float]
    data_period: Tuple[datetime, datetime]
    model_path: Optional[str] = None


@dataclass
class PredictionResult:
    """Result of risk prediction."""
    risk_score: float
    confidence: float
    risk_level: str
    prediction_date: datetime
    horizon_days: int
    feature_importance: Dict[str, float]
    model_version: str


class RiskPredictor:
    """
    Advanced ML-based risk prediction using ensemble methods.
    
    Features:
    - Random Forest regression for continuous risk scores
    - Random Forest classification for risk levels
    - Feature engineering pipeline integration
    - Model persistence and versioning
    - Performance evaluation and monitoring
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize the risk predictor."""
        self.cache_manager = cache_manager or CacheManager()
        
        # Feature engineers
        self.economic_engineer = EconomicFeatureEngineer()
        self.financial_engineer = FinancialFeatureEngineer()
        self.supply_chain_engineer = SupplyChainFeatureEngineer()
        self.disruption_engineer = DisruptionFeatureEngineer()
        
        # Models
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Metadata
        self.model_metadata = None
        self.feature_names = []
        
        # Model configuration
        self.model_config = {
            'regression': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'classification': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 25.0,
            'medium': 50.0,
            'high': 75.0,
            'critical': 90.0
        }
    
    async def prepare_features(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Prepare comprehensive feature matrix for training/prediction.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"Preparing features from {start_date} to {end_date}")
            
            # Collect features from all domains
            features_data = {}
            
            # Economic features
            economic_features = await self.economic_engineer.engineer_features(
                end_date=end_date
            )
            features_data.update(economic_features.features)
            
            # Financial features
            financial_features = await self.financial_engineer.extract_features(
                start_date=start_date,
                end_date=end_date
            )
            features_data.update(financial_features.features)
            
            # Supply chain features
            supply_chain_features = await self.supply_chain_engineer.extract_features(
                start_date=start_date,
                end_date=end_date
            )
            features_data.update(supply_chain_features.features)
            
            # Disruption features
            disruption_features = await self.disruption_engineer.extract_features(
                start_date=start_date,
                end_date=end_date
            )
            features_data.update(disruption_features.features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features_data])
            
            # Store feature names
            self.feature_names = list(feature_df.columns)
            
            logger.info(f"Prepared {len(self.feature_names)} features")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def _generate_synthetic_targets(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic target variables for training.
        
        This creates realistic risk scores based on feature patterns
        until real historical risk data is available.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Tuple of (continuous_targets, categorical_targets)
        """
        # Create synthetic risk scores based on feature patterns
        n_samples = len(features_df)
        
        # Base risk from economic indicators
        economic_risk = 0.0
        if 'unemployment_trend' in features_df.columns:
            economic_risk += features_df['unemployment_trend'].fillna(0) * 15
        if 'inflation_trend' in features_df.columns:
            economic_risk += features_df['inflation_trend'].fillna(0) * 10
        if 'gdp_growth_trend' in features_df.columns:
            economic_risk -= features_df['gdp_growth_trend'].fillna(0) * 8
        
        # Financial system risk
        financial_risk = 0.0
        if 'banking_stability' in features_df.columns:
            financial_risk += (100 - features_df['banking_stability'].fillna(50)) * 0.3
        if 'credit_risk_score' in features_df.columns:
            financial_risk += features_df['credit_risk_score'].fillna(0) * 0.4
        
        # Supply chain disruption risk
        supply_risk = 0.0
        if 'trade_disruption_index' in features_df.columns:
            supply_risk += features_df['trade_disruption_index'].fillna(0) * 0.5
        
        # Disruption event risk
        disruption_risk = 0.0
        if 'cyber_incidents_score' in features_df.columns:
            disruption_risk += features_df['cyber_incidents_score'].fillna(0) * 0.3
        if 'natural_disaster_risk' in features_df.columns:
            disruption_risk += features_df['natural_disaster_risk'].fillna(0) * 0.2
        
        # Combine risks with some random noise
        np.random.seed(42)
        noise = np.random.normal(0, 5, n_samples)
        
        risk_scores = np.clip(
            economic_risk + financial_risk + supply_risk + disruption_risk + noise,
            0, 100
        )
        
        # Create categorical targets
        risk_categories = []
        for score in risk_scores:
            if score < self.risk_thresholds['low']:
                risk_categories.append('low')
            elif score < self.risk_thresholds['medium']:
                risk_categories.append('medium')
            elif score < self.risk_thresholds['high']:
                risk_categories.append('high')
            else:
                risk_categories.append('critical')
        
        return risk_scores, np.array(risk_categories)
    
    async def train_models(self, start_date: datetime, end_date: datetime, 
                          validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train both regression and classification models.
        
        Args:
            start_date: Training data start date
            end_date: Training data end date
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting model training")
            
            # Prepare features
            features_df = await self.prepare_features(start_date, end_date)
            
            if features_df.empty:
                raise ValueError("No features available for training")
            
            # Ensure feature names are set
            if not self.feature_names:
                self.feature_names = list(features_df.columns)
            
            # Generate targets (synthetic for now)
            continuous_targets, categorical_targets = self._generate_synthetic_targets(features_df)
            
            # Prepare feature matrix
            X = features_df.values
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode categorical targets
            y_categorical_encoded = self.label_encoder.fit_transform(categorical_targets)
            
            # Split data
            X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = train_test_split(
                X_scaled, continuous_targets, y_categorical_encoded,
                test_size=validation_split, random_state=42, stratify=y_categorical_encoded
            )
            
            # Train regression model
            logger.info("Training regression model")
            self.regression_model = RandomForestRegressor(**self.model_config['regression'])
            self.regression_model.fit(X_train, y_reg_train)
            
            # Train classification model
            logger.info("Training classification model")
            self.classification_model = RandomForestClassifier(**self.model_config['classification'])
            self.classification_model.fit(X_train, y_clf_train)
            
            # Evaluate models
            evaluation_results = self._evaluate_models(
                X_val, y_reg_val, y_clf_val, categorical_targets[len(X_train):]
            )
            
            # Create model metadata
            self.model_metadata = ModelMetadata(
                model_name="RiskPredictor",
                model_type="RandomForest_Ensemble",
                version=f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trained_at=datetime.now(),
                features_used=self.feature_names,
                performance_metrics=evaluation_results,
                data_period=(start_date, end_date)
            )
            
            logger.info("Model training completed successfully")
            return {
                'success': True,
                'metadata': self.model_metadata,
                'evaluation': evaluation_results,
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def _evaluate_models(self, X_val: np.ndarray, y_reg_val: np.ndarray, 
                        y_clf_val: np.ndarray, y_clf_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate both regression and classification models.
        
        Args:
            X_val: Validation features
            y_reg_val: Regression targets
            y_clf_val: Classification targets (encoded)
            y_clf_labels: Classification targets (original labels)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Regression evaluation
        y_reg_pred = self.regression_model.predict(X_val)
        metrics['regression_mse'] = mean_squared_error(y_reg_val, y_reg_pred)
        metrics['regression_mae'] = mean_absolute_error(y_reg_val, y_reg_pred)
        metrics['regression_r2'] = r2_score(y_reg_val, y_reg_pred)
        metrics['regression_rmse'] = np.sqrt(metrics['regression_mse'])
        
        # Classification evaluation
        y_clf_pred = self.classification_model.predict(X_val)
        metrics['classification_accuracy'] = (y_clf_pred == y_clf_val).mean()
        
        # Feature importance
        reg_importance = self.regression_model.feature_importances_
        clf_importance = self.classification_model.feature_importances_
        
        # Store top features
        if len(self.feature_names) == len(reg_importance):
            top_features_reg = sorted(
                zip(self.feature_names, reg_importance),
                key=lambda x: x[1], reverse=True
            )[:10]
            top_features_clf = sorted(
                zip(self.feature_names, clf_importance),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            metrics['top_regression_features'] = dict(top_features_reg)
            metrics['top_classification_features'] = dict(top_features_clf)
        
        return metrics
    
    async def predict_risk(self, prediction_date: datetime, 
                          horizon_days: int = 30) -> PredictionResult:
        """
        Generate risk prediction for a specific date and horizon.
        
        Args:
            prediction_date: Date to predict for
            horizon_days: Prediction horizon in days
            
        Returns:
            Prediction result with risk score and metadata
        """
        try:
            if self.regression_model is None or self.classification_model is None:
                raise ValueError("Models not trained. Call train_models() first.")
            
            # Prepare features for prediction date
            features_df = await self.prepare_features(
                start_date=prediction_date - timedelta(days=30),
                end_date=prediction_date
            )
            
            if features_df.empty:
                raise ValueError("No features available for prediction")
            
            # Scale features
            X = features_df.values
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            risk_score = float(self.regression_model.predict(X_scaled)[0])
            risk_score = np.clip(risk_score, 0, 100)  # Ensure valid range
            
            # Get risk level
            risk_class_idx = self.classification_model.predict(X_scaled)[0]
            risk_level = self.label_encoder.inverse_transform([risk_class_idx])[0]
            
            # Calculate confidence based on model uncertainty
            reg_preds = []
            for estimator in self.regression_model.estimators_[:10]:  # Sample subset
                pred = estimator.predict(X_scaled)[0]
                reg_preds.append(pred)
            
            prediction_std = np.std(reg_preds)
            confidence = max(0.5, 1.0 - (prediction_std / 50.0))  # Normalize std to confidence
            
            # Get feature importance for this prediction
            feature_importance = {}
            if len(self.feature_names) == len(self.regression_model.feature_importances_):
                for name, importance in zip(self.feature_names, self.regression_model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            return PredictionResult(
                risk_score=risk_score,
                confidence=confidence,
                risk_level=risk_level,
                prediction_date=prediction_date,
                horizon_days=horizon_days,
                feature_importance=feature_importance,
                model_version=self.model_metadata.version if self.model_metadata else "unknown"
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_models(self, model_dir: str = "models") -> str:
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            Path to saved model files
        """
        try:
            model_path = Path(model_dir)
            model_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models
            reg_path = model_path / f"risk_regression_{timestamp}.joblib"
            clf_path = model_path / f"risk_classification_{timestamp}.joblib"
            scaler_path = model_path / f"feature_scaler_{timestamp}.joblib"
            encoder_path = model_path / f"label_encoder_{timestamp}.joblib"
            metadata_path = model_path / f"model_metadata_{timestamp}.joblib"
            
            joblib.dump(self.regression_model, reg_path)
            joblib.dump(self.classification_model, clf_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            joblib.dump(self.model_metadata, metadata_path)
            
            logger.info(f"Models saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, model_dir: str, timestamp: str = None) -> bool:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
            timestamp: Specific timestamp to load (latest if None)
            
        Returns:
            True if models loaded successfully
        """
        try:
            model_path = Path(model_dir)
            
            if timestamp:
                # Load specific timestamp
                reg_path = model_path / f"risk_regression_{timestamp}.joblib"
                clf_path = model_path / f"risk_classification_{timestamp}.joblib"
                scaler_path = model_path / f"feature_scaler_{timestamp}.joblib"
                encoder_path = model_path / f"label_encoder_{timestamp}.joblib"
                metadata_path = model_path / f"model_metadata_{timestamp}.joblib"
            else:
                # Find latest models
                reg_files = list(model_path.glob("risk_regression_*.joblib"))
                if not reg_files:
                    raise FileNotFoundError("No saved models found")
                
                latest_file = max(reg_files, key=lambda x: x.stat().st_mtime)
                timestamp = latest_file.stem.split("_")[-2] + "_" + latest_file.stem.split("_")[-1]
                
                reg_path = model_path / f"risk_regression_{timestamp}.joblib"
                clf_path = model_path / f"risk_classification_{timestamp}.joblib"
                scaler_path = model_path / f"feature_scaler_{timestamp}.joblib"
                encoder_path = model_path / f"label_encoder_{timestamp}.joblib"
                metadata_path = model_path / f"model_metadata_{timestamp}.joblib"
            
            # Load models
            self.regression_model = joblib.load(reg_path)
            self.classification_model = joblib.load(clf_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.model_metadata = joblib.load(metadata_path)
            
            logger.info(f"Models loaded from {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        if self.model_metadata is None:
            return {"status": "no_model_loaded"}
        
        return {
            "status": "loaded",
            "metadata": {
                "name": self.model_metadata.model_name,
                "type": self.model_metadata.model_type,
                "version": self.model_metadata.version,
                "trained_at": self.model_metadata.trained_at.isoformat(),
                "feature_count": len(self.model_metadata.features_used),
                "data_period": [
                    self.model_metadata.data_period[0].isoformat(),
                    self.model_metadata.data_period[1].isoformat()
                ]
            },
            "performance": self.model_metadata.performance_metrics,
            "config": self.model_config
        }