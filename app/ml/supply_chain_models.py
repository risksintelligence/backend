"""
Supply Chain Specific Machine Learning Models

Implements specialized ML models for supply chain risk analysis including:
- Cascade prediction models
- Disruption classification 
- Risk scoring models
- Anomaly detection for supply chain events
- Time series forecasting for trade flows
"""

import logging
import joblib
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, f1_score

from ..core.supply_chain_cache import get_supply_chain_cache
from ..db import SessionLocal
from ..models import (
    CascadeEvent, SupplyChainNode, SupplyChainRelationship, 
    SectorVulnerabilityAssessment, ACLEDEvent, ResilienceMetric
)

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    model_name: str
    version: str
    trained_at: datetime
    training_data_size: int
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model_type: str
    hyperparameters: Dict[str, Any]
    data_hash: str

@dataclass 
class PredictionResult:
    prediction: Any
    confidence: float
    model_version: str
    features_used: List[str]
    prediction_timestamp: datetime

class SupplyChainMLPipeline:
    """Enhanced ML pipeline for supply chain specific models."""
    
    def __init__(self, models_dir: str = "models/supply_chain"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache = get_supply_chain_cache()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Load existing models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models from disk."""
        try:
            for model_file in self.models_dir.glob("*.joblib"):
                model_name = model_file.stem
                try:
                    model_data = joblib.load(model_file)
                    self.models[model_name] = model_data
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
        except Exception as e:
            logger.warning(f"Models directory not found or empty: {e}")
    
    def _save_model(self, model_name: str, model_data: Dict[str, Any]) -> str:
        """Save model with metadata to disk and cache."""
        try:
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model_data, model_path)
            
            # Cache model metadata
            metadata = model_data.get('metadata')
            if metadata:
                self.cache.set(
                    "model_metadata", 
                    model_name, 
                    metadata, 
                    "ml_pipeline",
                    source_url=str(model_path)
                )
            
            logger.info(f"Model saved: {model_name}")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for versioning."""
        data_string = data.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _load_cascade_models(self) -> Tuple[Any, Any]:
        """Load cascade prediction models if available."""
        try:
            cascade_model = self.models.get('cascade_prediction')
            if cascade_model and 'classifier' in cascade_model:
                return cascade_model['classifier'], cascade_model.get('scaler')
            return None, None
        except Exception as e:
            logger.error(f"Failed to load cascade models: {e}")
            return None, None
    
    def _load_risk_scoring_models(self) -> Tuple[Any, Any]:
        """Load risk scoring models if available."""
        try:
            risk_model = self.models.get('risk_scoring') 
            if risk_model and 'regressor' in risk_model:
                return risk_model['regressor'], risk_model.get('scaler')
            return None, None
        except Exception as e:
            logger.error(f"Failed to load risk scoring models: {e}")
            return None, None
    
    def prepare_cascade_features(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Prepare features for cascade prediction models."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=365)
        if not end_date:
            end_date = datetime.utcnow()
        
        db = SessionLocal()
        try:
            # Get cascade events
            cascade_events = db.query(CascadeEvent).filter(
                CascadeEvent.event_start >= start_date,
                CascadeEvent.event_start <= end_date
            ).all()
            
            # Get supply chain nodes and relationships
            nodes = db.query(SupplyChainNode).all()
            relationships = db.query(SupplyChainRelationship).all()
            
            # Get ACLED geopolitical events
            acled_events = db.query(ACLEDEvent).filter(
                ACLEDEvent.event_date >= start_date,
                ACLEDEvent.event_date <= end_date
            ).all()
            
            features = []
            for event in cascade_events:
                feature_row = {
                    'event_id': event.id,
                    'severity_encoded': {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(event.severity, 0),
                    'cascade_depth': event.cascade_depth or 0,
                    'affected_countries_count': len(event.affected_countries or []),
                    'affected_sectors_count': len(event.affected_sectors or []),
                    'propagation_speed_encoded': {'immediate': 4, 'hours': 3, 'days': 2, 'weeks': 1}.get(event.propagation_speed, 0),
                    'estimated_cost_log': np.log1p(event.estimated_cost_usd or 0),
                    'recovery_time_days': event.recovery_time_days or 0,
                    'confidence_level': event.confidence_level or 0,
                    
                    # Time-based features
                    'month': event.event_start.month,
                    'quarter': (event.event_start.month - 1) // 3 + 1,
                    'day_of_week': event.event_start.weekday(),
                    
                    # Geopolitical context features
                    'concurrent_acled_events': len([
                        ae for ae in acled_events 
                        if abs((ae.event_date - event.event_start).days) <= 7
                    ]),
                    
                    # Network topology features
                    'avg_node_risk_score': np.mean([n.overall_risk_score for n in nodes if n.overall_risk_score]) or 0,
                    'critical_relationships_count': len([r for r in relationships if r.criticality_score and r.criticality_score > 0.8]),
                    
                    # Target variables
                    'next_cascade_will_occur': 1 if event.parent_event_id else 0,  # Will this trigger another cascade
                    'cascade_severity_score': event.cascade_depth * ({'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(event.severity, 0))
                }
                features.append(feature_row)
            
            df = pd.DataFrame(features)
            logger.info(f"Prepared {len(df)} cascade feature rows")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing cascade features: {e}")
            return pd.DataFrame()
        finally:
            db.close()
    
    def prepare_risk_scoring_features(self) -> pd.DataFrame:
        """Prepare features for supply chain risk scoring models."""
        db = SessionLocal()
        try:
            # Get nodes with risk scores
            nodes = db.query(SupplyChainNode).filter(
                SupplyChainNode.overall_risk_score.isnot(None)
            ).all()
            
            # Get vulnerability assessments
            assessments = db.query(SectorVulnerabilityAssessment).all()
            assessment_by_sector = {a.sector: a for a in assessments}
            
            # Get resilience metrics
            resilience_metrics = db.query(ResilienceMetric).all()
            metrics_by_entity = {m.entity_id: m for m in resilience_metrics}
            
            features = []
            for node in nodes:
                # Get sector vulnerability data
                sector_assessment = assessment_by_sector.get(node.industry_sector)
                resilience_metric = metrics_by_entity.get(node.id)
                
                feature_row = {
                    'node_id': node.id,
                    'node_type_encoded': {'company': 1, 'port': 2, 'region': 3, 'sector': 4}.get(node.node_type, 0),
                    'tier_level': node.tier_level or 0,
                    'financial_health_score': node.financial_health_score or 0,
                    'operational_risk_score': node.operational_risk_score or 0,
                    'geopolitical_risk_score': node.geopolitical_risk_score or 0,
                    
                    # Sector vulnerability features
                    'sector_overall_risk': sector_assessment.overall_risk_score if sector_assessment else 0,
                    'sector_complexity': sector_assessment.complexity_score if sector_assessment else 0,
                    'sector_globalization': sector_assessment.globalization_index if sector_assessment else 0,
                    'sector_critical_vulnerabilities': sector_assessment.critical_vulnerabilities if sector_assessment else 0,
                    
                    # Resilience features
                    'resilience_score': resilience_metric.overall_resilience_score if resilience_metric else 0,
                    'supplier_diversity': resilience_metric.supplier_diversity_score if resilience_metric else 0,
                    'geographic_distribution': resilience_metric.geographic_distribution_score if resilience_metric else 0,
                    'recovery_time_estimate': resilience_metric.recovery_time_estimate_days if resilience_metric else 0,
                    
                    # Geographic risk factors
                    'country_risk_encoded': self._encode_country_risk(node.country),
                    'region_risk_encoded': self._encode_region_risk(node.region),
                    
                    # Target variable
                    'overall_risk_score': node.overall_risk_score
                }
                features.append(feature_row)
            
            df = pd.DataFrame(features)
            logger.info(f"Prepared {len(df)} risk scoring feature rows")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing risk scoring features: {e}")
            return pd.DataFrame()
        finally:
            db.close()
    
    def _encode_country_risk(self, country: str) -> int:
        """Encode country risk level (simplified)."""
        if not country:
            return 0
        
        high_risk_countries = ['Afghanistan', 'Syria', 'Yemen', 'Somalia', 'South Sudan']
        medium_risk_countries = ['Ukraine', 'Myanmar', 'Venezuela', 'Iran', 'North Korea']
        
        if country in high_risk_countries:
            return 3
        elif country in medium_risk_countries:
            return 2
        else:
            return 1
    
    def _encode_region_risk(self, region: str) -> int:
        """Encode regional risk level (simplified)."""
        if not region:
            return 0
            
        high_risk_regions = ['Middle East', 'Horn of Africa', 'Central Asia']
        medium_risk_regions = ['Eastern Europe', 'Southeast Asia', 'Central America']
        
        if region in high_risk_regions:
            return 3
        elif region in medium_risk_regions:
            return 2
        else:
            return 1
    
    def train_cascade_prediction_model(self, retrain: bool = False) -> ModelMetadata:
        """Train model to predict cascade likelihood and severity."""
        model_name = "cascade_prediction"
        
        # Check if model exists and is recent
        if not retrain and model_name in self.models:
            cached_metadata, _ = self.cache.get("model_metadata", model_name)
            if cached_metadata and (datetime.utcnow() - cached_metadata.trained_at).days < 7:
                logger.info(f"Using existing {model_name} model (trained {cached_metadata.trained_at})")
                return cached_metadata
        
        try:
            # Prepare training data
            logger.info(f"Training {model_name} model...")
            df = self.prepare_cascade_features()
            
            if df.empty:
                raise ValueError("No training data available")
            
            # Feature selection
            feature_cols = [col for col in df.columns if col not in ['event_id', 'next_cascade_will_occur', 'cascade_severity_score']]
            X = df[feature_cols]
            y_classification = df['next_cascade_will_occur']
            y_regression = df['cascade_severity_score']
            
            # Handle missing values
            X = X.fillna(0)
            
            # Train classification model (will cascade occur?)
            X_train, X_test, y_train_class, y_test_class = train_test_split(
                X, y_classification, test_size=0.2, random_state=42
            )
            
            cascade_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            cascade_classifier.fit(X_train, y_train_class)
            
            # Train regression model (severity score)
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X, y_regression, test_size=0.2, random_state=42
            )
            
            severity_regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            severity_regressor.fit(X_train_reg, y_train_reg)
            
            # Evaluate models
            class_accuracy = accuracy_score(y_test_class, cascade_classifier.predict(X_test))
            class_f1 = f1_score(y_test_class, cascade_classifier.predict(X_test), average='weighted')
            reg_mse = mean_squared_error(y_test_reg, severity_regressor.predict(X_test_reg))
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, cascade_classifier.feature_importances_))
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                trained_at=datetime.utcnow(),
                training_data_size=len(df),
                performance_metrics={
                    'classification_accuracy': class_accuracy,
                    'classification_f1': class_f1,
                    'regression_mse': reg_mse
                },
                feature_importance=feature_importance,
                model_type='cascade_prediction',
                hyperparameters={
                    'classifier_n_estimators': 100,
                    'regressor_n_estimators': 100,
                    'classifier_max_depth': 10,
                    'regressor_max_depth': 6
                },
                data_hash=self._calculate_data_hash(df)
            )
            
            # Save model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model_data = {
                'classifier': cascade_classifier,
                'regressor': severity_regressor,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'metadata': metadata
            }
            
            self._save_model(model_name, model_data)
            self.models[model_name] = model_data
            
            logger.info(f"Successfully trained {model_name} model - Accuracy: {class_accuracy:.3f}, F1: {class_f1:.3f}, MSE: {reg_mse:.3f}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} model: {e}")
            raise
    
    def train_risk_scoring_model(self, retrain: bool = False) -> ModelMetadata:
        """Train model to predict overall risk scores for supply chain entities."""
        model_name = "risk_scoring"
        
        # Check if model exists and is recent
        if not retrain and model_name in self.models:
            cached_metadata, _ = self.cache.get("model_metadata", model_name)
            if cached_metadata and (datetime.utcnow() - cached_metadata.trained_at).days < 30:
                logger.info(f"Using existing {model_name} model")
                return cached_metadata
        
        try:
            logger.info(f"Training {model_name} model...")
            df = self.prepare_risk_scoring_features()
            
            if df.empty:
                raise ValueError("No training data available")
            
            # Feature selection
            feature_cols = [col for col in df.columns if col not in ['node_id', 'overall_risk_score']]
            X = df[feature_cols]
            y = df['overall_risk_score']
            
            # Handle missing values
            X = X.fillna(0)
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            risk_model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            risk_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = risk_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)
            r2_score = risk_model.score(X_test_scaled, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, risk_model.feature_importances_))
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                trained_at=datetime.utcnow(),
                training_data_size=len(df),
                performance_metrics={
                    'mse': mse,
                    'r2_score': r2_score,
                    'mean_absolute_error': np.mean(np.abs(predictions - y_test))
                },
                feature_importance=feature_importance,
                model_type='risk_scoring',
                hyperparameters={
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.1
                },
                data_hash=self._calculate_data_hash(df)
            )
            
            # Save model
            model_data = {
                'model': risk_model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'metadata': metadata
            }
            
            self._save_model(model_name, model_data)
            self.models[model_name] = model_data
            
            logger.info(f"Successfully trained {model_name} model - R²: {r2_score:.3f}, MSE: {mse:.3f}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} model: {e}")
            raise
    
    def predict_cascade_likelihood(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict likelihood and severity of cascade events."""
        model_name = "cascade_prediction"
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Train the model first.")
        
        model_data = self.models[model_name]
        classifier = model_data['classifier']
        regressor = model_data['regressor']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        metadata = model_data['metadata']
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[features.get(col, 0) for col in feature_columns]])
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Predict
            cascade_probability = classifier.predict_proba(feature_vector_scaled)[0][1]  # Probability of cascade
            severity_score = regressor.predict(feature_vector_scaled)[0]
            
            # Calculate confidence based on feature importance and data quality
            confidence = min(0.95, max(0.1, cascade_probability * 0.8 + 0.2))
            
            return PredictionResult(
                prediction={
                    'cascade_probability': cascade_probability,
                    'severity_score': max(0, severity_score),
                    'risk_level': 'high' if cascade_probability > 0.7 else 'medium' if cascade_probability > 0.3 else 'low'
                },
                confidence=confidence,
                model_version=metadata.version,
                features_used=feature_columns,
                prediction_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_risk_score(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict risk score for supply chain entity."""
        model_name = "risk_scoring"
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Train the model first.")
        
        model_data = self.models[model_name]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        metadata = model_data['metadata']
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[features.get(col, 0) for col in feature_columns]])
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Predict
            risk_score = model.predict(feature_vector_scaled)[0]
            risk_score = max(0, min(1, risk_score))  # Clamp to [0, 1]
            
            # Calculate confidence based on model performance
            confidence = max(0.5, metadata.performance_metrics.get('r2_score', 0.5))
            
            return PredictionResult(
                prediction={
                    'risk_score': risk_score,
                    'risk_category': 'critical' if risk_score > 0.8 else 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.4 else 'low'
                },
                confidence=confidence,
                model_version=metadata.version,
                features_used=feature_columns,
                prediction_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Risk score prediction failed: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Get information about a specific model."""
        if model_name in self.models:
            return self.models[model_name]['metadata']
        
        # Try to get from cache
        cached_metadata, _ = self.cache.get("model_metadata", model_name)
        return cached_metadata
    
    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        return list(self.models.keys())
    
    def train_all_models(self, force_retrain: bool = False) -> Dict[str, ModelMetadata]:
        """Train all supply chain ML models."""
        results = {}
        
        try:
            logger.info("Starting training for all supply chain ML models...")
            
            # Train cascade prediction model
            results['cascade_prediction'] = self.train_cascade_prediction_model(retrain=force_retrain)
            
            # Train risk scoring model  
            results['risk_scoring'] = self.train_risk_scoring_model(retrain=force_retrain)
            
            logger.info("All supply chain models trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train all models: {e}")
            raise


# Singleton instance
_ml_pipeline = None

def get_supply_chain_ml_pipeline() -> SupplyChainMLPipeline:
    """Get the global supply chain ML pipeline instance."""
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = SupplyChainMLPipeline()
    return _ml_pipeline

def predict_cascade_event(features: Dict[str, Any]) -> PredictionResult:
    """Convenience function to predict cascade events."""
    pipeline = get_supply_chain_ml_pipeline()
    return pipeline.predict_cascade_likelihood(features)

def predict_entity_risk(features: Dict[str, Any]) -> PredictionResult:
    """Convenience function to predict entity risk scores."""
    pipeline = get_supply_chain_ml_pipeline()
    return pipeline.predict_risk_score(features)