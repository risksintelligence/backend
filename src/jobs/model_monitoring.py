#!/usr/bin/env python3
"""
ML Model Monitoring and Drift Detection Job
Automated monitoring of model performance and data drift
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.model_registry_service import get_model_registry_service, ModelDriftMetrics
from src.core.database import get_database_pool
from src.core.logging import get_logger
from src.monitoring.observability import get_observability_service
from src.services.alerts_delivery import get_alert_delivery_service

logger = get_logger(__name__)

class ModelMonitoringJob:
    """Background job for monitoring ML models."""
    
    def __init__(self):
        self.model_registry = get_model_registry_service()
        self.observability = get_observability_service()
        self.alert_service = get_alert_delivery_service()
        
        self.model_types = ['regime_classifier', 'forecaster', 'anomaly_detector']
        self.monitoring_window_hours = 24
    
    async def run_monitoring_cycle(self):
        """Execute complete model monitoring cycle."""
        start_time = datetime.now(timezone.utc)
        logger.info("Starting ML model monitoring cycle")
        
        monitoring_results = {
            'timestamp': start_time.isoformat(),
            'models_monitored': 0,
            'drift_alerts': [],
            'performance_issues': [],
            'recommendations': []
        }
        
        try:
            for model_type in self.model_types:
                logger.info(f"Monitoring {model_type} models")
                
                # Get recent predictions for drift analysis
                recent_predictions = await self._get_recent_predictions(model_type)
                
                if recent_predictions:
                    # Run drift detection
                    drift_metrics = await self.model_registry.detect_model_drift(
                        model_type, 
                        recent_predictions
                    )
                    
                    monitoring_results['models_monitored'] += 1
                    
                    # Check for alerts
                    if drift_metrics.alert_threshold_exceeded:
                        monitoring_results['drift_alerts'].append({
                            'model_type': model_type,
                            'model_id': drift_metrics.model_id,
                            'feature_drift': drift_metrics.feature_drift_score,
                            'prediction_drift': drift_metrics.prediction_drift_score,
                            'performance_degradation': drift_metrics.performance_degradation
                        })
                        
                        # Send alert notification
                        await self._send_drift_alert(model_type, drift_metrics)
                    
                    # Collect recommendations
                    monitoring_results['recommendations'].extend([
                        f"{model_type}: {rec}" for rec in drift_metrics.recommendations
                    ])
                    
                    # Record metrics for observability
                    self.observability.record_ml_prediction(
                        model_type, 
                        drift_metrics.model_id,
                        drift_metrics.feature_drift_score
                    )
                
                else:
                    logger.warning(f"No recent predictions found for {model_type}")
            
            # Check for models needing retraining
            await self._check_retraining_schedule()
            
            # Clean up old model versions
            cleanup_count = await self.model_registry.cleanup_old_models()
            if cleanup_count > 0:
                monitoring_results['recommendations'].append(f"Cleaned up {cleanup_count} old model versions")
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(f"Model monitoring completed in {duration:.2f}s: {monitoring_results['models_monitored']} models monitored")
            
            # Record job execution
            self.observability.record_job_execution("model_monitoring", duration, True)
            
            return monitoring_results
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"Model monitoring failed after {duration:.2f}s: {e}")
            
            # Record failed execution
            self.observability.record_job_execution("model_monitoring", duration, False)
            
            monitoring_results['error'] = str(e)
            return monitoring_results
    
    async def _get_recent_predictions(self, model_type: str) -> List[Dict[str, Any]]:
        """Get recent model predictions for drift analysis."""
        try:
            since_time = datetime.now(timezone.utc) - timedelta(hours=self.monitoring_window_hours)
            
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT prediction_data, features, accuracy_score, created_at
                    FROM ml_model_predictions 
                    WHERE model_type = $1 
                    AND created_at >= $2
                    ORDER BY created_at DESC
                    LIMIT 1000
                """, model_type, since_time)
                
                predictions = []
                for row in rows:
                    try:
                        prediction_data = row['prediction_data'] if isinstance(row['prediction_data'], dict) else {}
                        features = row['features'] if isinstance(row['features'], dict) else {}
                        
                        predictions.append({
                            'prediction': prediction_data.get('value', 0.0),
                            'features': features,
                            'accuracy': row['accuracy_score'] if row['accuracy_score'] is not None else 0.0,
                            'timestamp': row['created_at'].isoformat()
                        })
                    except Exception as e:
                        logger.warning(f"Skipping malformed prediction data: {e}")
                        continue
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting recent predictions for {model_type}: {e}")
            return []
    
    async def _send_drift_alert(self, model_type: str, drift_metrics: ModelDriftMetrics):
        """Send alert notification for model drift."""
        try:
            alert_data = {
                'alert_type': 'model_drift',
                'model_type': model_type,
                'model_id': drift_metrics.model_id,
                'severity': 'high' if max(drift_metrics.feature_drift_score, drift_metrics.prediction_drift_score) > 0.5 else 'medium',
                'metrics': {
                    'feature_drift': drift_metrics.feature_drift_score,
                    'prediction_drift': drift_metrics.prediction_drift_score,
                    'performance_degradation': drift_metrics.performance_degradation
                },
                'recommendations': drift_metrics.recommendations,
                'timestamp': drift_metrics.timestamp.isoformat()
            }
            
            # This would integrate with the alert delivery service
            # For now, just log the alert
            logger.warning(f"Model drift alert: {model_type} - {alert_data}")
            
            # Record alert trigger in observability
            self.observability.record_alert_trigger("model_drift", alert_data['severity'])
            
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}")
    
    async def _check_retraining_schedule(self):
        """Check if any models need scheduled retraining."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Find models that haven't been retrained in 30 days
                stale_models = await conn.fetch("""
                    SELECT model_type, model_id, model_data, created_at
                    FROM ml_models 
                    WHERE is_active = TRUE 
                    AND created_at < $1
                """, datetime.now(timezone.utc) - timedelta(days=30))
                
                for row in stale_models:
                    model_type = row['model_type']
                    model_id = row['model_id']
                    created_at = row['created_at']
                    
                    days_old = (datetime.now(timezone.utc) - created_at).days
                    
                    logger.info(f"Model {model_id} ({model_type}) is {days_old} days old - consider retraining")
                    
                    # This could trigger automated retraining
                    # For now, just log recommendation
                
        except Exception as e:
            logger.error(f"Error checking retraining schedule: {e}")
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        try:
            report_time = datetime.now(timezone.utc)
            
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get model summary
                model_summary = await conn.fetch("""
                    SELECT model_type, COUNT(*) as total_models,
                           SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_models,
                           MAX(created_at) as latest_model
                    FROM ml_models 
                    GROUP BY model_type
                """)
                
                # Get recent drift metrics
                recent_drift = await conn.fetch("""
                    SELECT model_id, drift_data, timestamp
                    FROM ml_drift_metrics 
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, report_time - timedelta(days=7))
                
                # Get prediction volume
                prediction_volume = await conn.fetch("""
                    SELECT model_type, COUNT(*) as prediction_count,
                           AVG(accuracy_score) as avg_accuracy
                    FROM ml_model_predictions 
                    WHERE created_at >= $1
                    GROUP BY model_type
                """, report_time - timedelta(days=7))
                
                report = {
                    'generated_at': report_time.isoformat(),
                    'model_summary': [
                        {
                            'model_type': row['model_type'],
                            'total_models': row['total_models'],
                            'active_models': row['active_models'],
                            'latest_model': row['latest_model'].isoformat() if row['latest_model'] else None
                        }
                        for row in model_summary
                    ],
                    'prediction_volume': [
                        {
                            'model_type': row['model_type'],
                            'predictions_7d': row['prediction_count'],
                            'avg_accuracy': float(row['avg_accuracy']) if row['avg_accuracy'] else 0.0
                        }
                        for row in prediction_volume
                    ],
                    'recent_drift_events': len(recent_drift),
                    'system_health': 'healthy'  # Simplified health assessment
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            return {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }

async def main():
    """Main entry point for model monitoring job."""
    try:
        monitoring_job = ModelMonitoringJob()
        
        # Run monitoring cycle
        results = await monitoring_job.run_monitoring_cycle()
        
        # Generate and log report
        report = await monitoring_job.generate_monitoring_report()
        
        print("=== ML Model Monitoring Report ===")
        print(f"Generated: {report['generated_at']}")
        
        if 'error' in report:
            print(f"Error: {report['error']}")
            return
        
        print(f"Models monitored: {results['models_monitored']}")
        print(f"Drift alerts: {len(results['drift_alerts'])}")
        
        for model_info in report['model_summary']:
            print(f"- {model_info['model_type']}: {model_info['active_models']} active, {model_info['total_models']} total")
        
        if results['drift_alerts']:
            print("\nDrift Alerts:")
            for alert in results['drift_alerts']:
                print(f"- {alert['model_type']} ({alert['model_id']}): drift={alert['feature_drift']:.3f}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations'][-5:]:  # Show last 5
                print(f"- {rec}")
        
    except Exception as e:
        logger.error(f"Model monitoring job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())