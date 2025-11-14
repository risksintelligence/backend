"""
ML Model Registry and Management Service
Handles model versioning, drift detection, and automated retraining
"""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.base import BaseEstimator

from src.core.config import get_settings
from src.core.database import get_database_pool
from src.core.logging import get_logger
from src.monitoring.observability import get_observability_service

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class ModelMetadata:
    """Metadata for registered ML models."""
    model_id: str
    model_type: str  # 'regime_classifier', 'forecaster', 'anomaly_detector'
    version: str
    created_at: datetime
    trained_at: datetime
    training_data_hash: str
    performance_metrics: Dict[str, float]
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    file_path: str
    file_size_bytes: int
    is_active: bool = False
    deployment_notes: str = ""

@dataclass
class ModelDriftMetrics:
    """Model drift detection metrics."""
    model_id: str
    timestamp: datetime
    feature_drift_score: float
    prediction_drift_score: float
    performance_degradation: float
    data_distribution_shift: Dict[str, float]
    alert_threshold_exceeded: bool
    recommendations: List[str]

class ModelRegistryService:
    """Service for managing ML model lifecycle and monitoring."""
    
    def __init__(self):
        self.model_dir = Path(settings.MODEL_STORAGE_PATH or "/tmp/models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.drift_detection_window = 7  # days
        self.performance_threshold = 0.85  # minimum acceptable accuracy
        self.drift_alert_threshold = 0.3  # drift score threshold
        
        # In-memory model cache
        self._model_cache: Dict[str, Tuple[BaseEstimator, ModelMetadata]] = {}
        
    async def register_model(
        self,
        model: BaseEstimator,
        model_type: str,
        training_data_hash: str,
        performance_metrics: Dict[str, float],
        features_used: List[str],
        hyperparameters: Dict[str, Any],
        deployment_notes: str = ""
    ) -> ModelMetadata:
        """Register a new model version in the registry."""
        
        # Generate model ID and version
        timestamp = datetime.now(timezone.utc)
        model_id = f"{model_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        version = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Save model to disk
        model_filename = f"{model_id}.pkl"
        model_path = self.model_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        file_size = model_path.stat().st_size
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=timestamp,
            trained_at=timestamp,
            training_data_hash=training_data_hash,
            performance_metrics=performance_metrics,
            features_used=features_used,
            hyperparameters=hyperparameters,
            file_path=str(model_path),
            file_size_bytes=file_size,
            deployment_notes=deployment_notes
        )
        
        # Store metadata in database
        await self._store_model_metadata(metadata)
        
        # Add to cache
        self._model_cache[model_id] = (model, metadata)
        
        logger.info(f"Registered model {model_id} version {version}")
        
        return metadata
    
    async def get_active_model(self, model_type: str) -> Optional[Tuple[BaseEstimator, ModelMetadata]]:
        """Get the currently active model for a given type."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT model_id, model_data 
                    FROM ml_models 
                    WHERE model_type = $1 AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """, model_type)
                
                if not row:
                    return None
                
                model_id = row['model_id']
                
                # Check cache first
                if model_id in self._model_cache:
                    return self._model_cache[model_id]
                
                # Load from disk
                model_data = row['model_data']
                metadata = ModelMetadata(**model_data)
                
                if Path(metadata.file_path).exists():
                    with open(metadata.file_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Cache for future use
                    self._model_cache[model_id] = (model, metadata)
                    return (model, metadata)
                else:
                    logger.warning(f"Model file not found: {metadata.file_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading active model {model_type}: {e}")
            return None
    
    async def promote_model(self, model_id: str) -> bool:
        """Promote a model to active status."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get model metadata
                row = await conn.fetchrow("""
                    SELECT model_type, model_data 
                    FROM ml_models 
                    WHERE model_id = $1
                """, model_id)
                
                if not row:
                    logger.error(f"Model not found: {model_id}")
                    return False
                
                model_type = row['model_type']
                
                async with conn.transaction():
                    # Deactivate current active model
                    await conn.execute("""
                        UPDATE ml_models 
                        SET is_active = FALSE 
                        WHERE model_type = $1 AND is_active = TRUE
                    """, model_type)
                    
                    # Activate new model
                    await conn.execute("""
                        UPDATE ml_models 
                        SET is_active = TRUE 
                        WHERE model_id = $1
                    """, model_id)
                
                # Clear cache to force reload
                self._model_cache.clear()
                
                logger.info(f"Promoted model {model_id} to active for type {model_type}")
                
                # Record promotion event
                observability = get_observability_service()
                observability.record_ml_prediction(model_type, model_id, None)
                
                return True
                
        except Exception as e:
            logger.error(f"Error promoting model {model_id}: {e}")
            return False
    
    async def detect_model_drift(self, model_type: str, recent_predictions: List[Dict[str, Any]]) -> ModelDriftMetrics:
        """Detect drift in model performance and data distribution."""
        
        timestamp = datetime.now(timezone.utc)
        
        # Get active model metadata
        active_model_data = await self.get_active_model(model_type)
        if not active_model_data:
            return ModelDriftMetrics(
                model_id="unknown",
                timestamp=timestamp,
                feature_drift_score=0.0,
                prediction_drift_score=0.0,
                performance_degradation=0.0,
                data_distribution_shift={},
                alert_threshold_exceeded=False,
                recommendations=["No active model found"]
            )
        
        model, metadata = active_model_data
        
        try:
            # Calculate feature drift using statistical tests
            feature_drift_score = await self._calculate_feature_drift(
                metadata.features_used, 
                recent_predictions
            )
            
            # Calculate prediction drift
            prediction_drift_score = await self._calculate_prediction_drift(
                model_type, 
                recent_predictions
            )
            
            # Calculate performance degradation
            performance_degradation = await self._calculate_performance_degradation(
                model_type,
                metadata.performance_metrics
            )
            
            # Detect data distribution shifts
            distribution_shift = await self._detect_distribution_shift(
                metadata.features_used,
                recent_predictions
            )
            
            # Determine if alert threshold is exceeded
            max_drift = max(feature_drift_score, prediction_drift_score)
            alert_threshold_exceeded = (
                max_drift > self.drift_alert_threshold or
                performance_degradation > 0.1  # 10% performance drop
            )
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(
                feature_drift_score,
                prediction_drift_score,
                performance_degradation,
                alert_threshold_exceeded
            )
            
            drift_metrics = ModelDriftMetrics(
                model_id=metadata.model_id,
                timestamp=timestamp,
                feature_drift_score=feature_drift_score,
                prediction_drift_score=prediction_drift_score,
                performance_degradation=performance_degradation,
                data_distribution_shift=distribution_shift,
                alert_threshold_exceeded=alert_threshold_exceeded,
                recommendations=recommendations
            )
            
            # Store drift metrics
            await self._store_drift_metrics(drift_metrics)
            
            if alert_threshold_exceeded:
                logger.warning(f"Model drift detected for {model_type}: drift={max_drift:.3f}, degradation={performance_degradation:.3f}")
                
                # Trigger alert through observability service
                observability = get_observability_service()
                observability.record_alert_trigger("model_drift", "high" if max_drift > 0.5 else "medium")
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error detecting drift for model {metadata.model_id}: {e}")
            return ModelDriftMetrics(
                model_id=metadata.model_id,
                timestamp=timestamp,
                feature_drift_score=0.0,
                prediction_drift_score=0.0,
                performance_degradation=0.0,
                data_distribution_shift={},
                alert_threshold_exceeded=False,
                recommendations=[f"Drift detection error: {e}"]
            )
    
    async def _calculate_feature_drift(self, feature_names: List[str], recent_data: List[Dict[str, Any]]) -> float:
        """Calculate feature drift using Kolmogorov-Smirnov test."""
        try:
            if not recent_data:
                return 0.0
            
            # Extract features from recent data
            recent_features = []
            for data_point in recent_data:
                features = data_point.get('features', {})
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                recent_features.append(feature_vector)
            
            if not recent_features:
                return 0.0
            
            recent_features = np.array(recent_features)
            
            # Compare with training data distribution (simplified)
            # In production, you'd compare with stored training data statistics
            feature_drift_scores = []
            for i, feature_name in enumerate(feature_names):
                feature_values = recent_features[:, i]
                
                # Calculate distribution metrics
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                
                # Simple drift score based on z-score variance
                # This is simplified - use proper statistical tests in production
                drift_score = min(std_val / (abs(mean_val) + 1e-6), 1.0)
                feature_drift_scores.append(drift_score)
            
            return np.mean(feature_drift_scores)
            
        except Exception as e:
            logger.error(f"Error calculating feature drift: {e}")
            return 0.0
    
    async def _calculate_prediction_drift(self, model_type: str, recent_predictions: List[Dict[str, Any]]) -> float:
        """Calculate drift in prediction patterns."""
        try:
            if not recent_predictions:
                return 0.0
            
            # Extract predictions
            predictions = [pred.get('prediction', 0.0) for pred in recent_predictions]
            
            if not predictions:
                return 0.0
            
            predictions = np.array(predictions)
            
            # Calculate prediction distribution metrics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Historical baseline (simplified - should use stored baselines)
            historical_mean = 50.0 if model_type == 'regime_classifier' else 0.0
            historical_std = 25.0 if model_type == 'regime_classifier' else 1.0
            
            # Calculate drift as normalized difference
            mean_drift = abs(mean_pred - historical_mean) / (historical_std + 1e-6)
            std_drift = abs(std_pred - historical_std) / (historical_std + 1e-6)
            
            return min((mean_drift + std_drift) / 2.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating prediction drift: {e}")
            return 0.0
    
    async def _calculate_performance_degradation(self, model_type: str, baseline_metrics: Dict[str, float]) -> float:
        """Calculate performance degradation compared to baseline."""
        try:
            # Get recent performance metrics from database
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT AVG(accuracy_score) as avg_accuracy
                    FROM ml_model_predictions 
                    WHERE model_type = $1 
                    AND created_at > $2
                """, model_type, datetime.now(timezone.utc) - timedelta(days=self.drift_detection_window))
                
                if not row or not row['avg_accuracy']:
                    return 0.0
                
                recent_accuracy = float(row['avg_accuracy'])
                baseline_accuracy = baseline_metrics.get('accuracy', self.performance_threshold)
                
                # Calculate degradation as relative performance drop
                degradation = max(0, (baseline_accuracy - recent_accuracy) / baseline_accuracy)
                
                return degradation
                
        except Exception as e:
            logger.error(f"Error calculating performance degradation: {e}")
            return 0.0
    
    async def _detect_distribution_shift(self, feature_names: List[str], recent_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Detect shifts in data distribution per feature."""
        distribution_shifts = {}
        
        try:
            for feature_name in feature_names:
                feature_values = [
                    data_point.get('features', {}).get(feature_name, 0.0) 
                    for data_point in recent_data
                ]
                
                if feature_values:
                    feature_values = np.array(feature_values)
                    
                    # Calculate basic distribution metrics
                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values)
                    
                    # Simple shift detection (replace with proper statistical tests)
                    shift_score = min(std_val / (abs(mean_val) + 1e-6), 1.0)
                    distribution_shifts[feature_name] = float(shift_score)
                else:
                    distribution_shifts[feature_name] = 0.0
                    
        except Exception as e:
            logger.error(f"Error detecting distribution shift: {e}")
        
        return distribution_shifts
    
    def _generate_drift_recommendations(
        self, 
        feature_drift: float,
        prediction_drift: float,
        performance_degradation: float,
        alert_threshold_exceeded: bool
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        if alert_threshold_exceeded:
            recommendations.append("🚨 Immediate attention required - significant drift detected")
        
        if feature_drift > self.drift_alert_threshold:
            recommendations.append(f"📊 Feature drift detected ({feature_drift:.3f}) - review data sources")
            
        if prediction_drift > self.drift_alert_threshold:
            recommendations.append(f"🎯 Prediction drift detected ({prediction_drift:.3f}) - consider model retraining")
            
        if performance_degradation > 0.1:
            recommendations.append(f"📉 Performance degraded by {performance_degradation:.1%} - retraining recommended")
        
        if feature_drift > 0.1 or prediction_drift > 0.1:
            recommendations.append("🔄 Schedule model retraining with recent data")
            
        if performance_degradation > 0.05:
            recommendations.append("📈 Monitor model performance more closely")
            
        if not recommendations:
            recommendations.append("✅ Model performance is stable")
            
        return recommendations
    
    async def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_models (
                        model_id, model_type, version, model_data, is_active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (model_id) DO UPDATE SET
                        model_data = EXCLUDED.model_data,
                        updated_at = NOW()
                """,
                metadata.model_id,
                metadata.model_type,
                metadata.version,
                json.dumps(asdict(metadata), default=str),
                metadata.is_active,
                metadata.created_at
                )
                
        except Exception as e:
            logger.error(f"Error storing model metadata: {e}")
            raise
    
    async def _store_drift_metrics(self, drift_metrics: ModelDriftMetrics):
        """Store drift detection metrics."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_drift_metrics (
                        model_id, timestamp, drift_data
                    ) VALUES ($1, $2, $3)
                """,
                drift_metrics.model_id,
                drift_metrics.timestamp,
                json.dumps(asdict(drift_metrics), default=str)
                )
                
        except Exception as e:
            logger.error(f"Error storing drift metrics: {e}")

    async def cleanup_old_models(self, keep_versions: int = 5):
        """Clean up old model versions, keeping only the most recent ones."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get models to delete (keep latest N versions per type)
                old_models = await conn.fetch("""
                    WITH ranked_models AS (
                        SELECT model_id, model_type, model_data,
                               ROW_NUMBER() OVER (PARTITION BY model_type ORDER BY created_at DESC) as rn
                        FROM ml_models 
                        WHERE is_active = FALSE
                    )
                    SELECT model_id, model_data 
                    FROM ranked_models 
                    WHERE rn > $1
                """, keep_versions)
                
                deleted_count = 0
                for row in old_models:
                    model_id = row['model_id']
                    model_data = json.loads(row['model_data'])
                    
                    # Delete model file
                    model_path = Path(model_data.get('file_path', ''))
                    if model_path.exists():
                        model_path.unlink()
                        logger.info(f"Deleted model file: {model_path}")
                    
                    # Delete database record
                    await conn.execute("DELETE FROM ml_models WHERE model_id = $1", model_id)
                    
                    deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} old model versions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            return 0

# Global service instance
_model_registry_service = None

def get_model_registry_service() -> ModelRegistryService:
    global _model_registry_service
    if _model_registry_service is None:
        _model_registry_service = ModelRegistryService()
    return _model_registry_service