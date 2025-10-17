#!/usr/bin/env python3
"""
Model training script for RiskX risk prediction models.

This script trains ML models using historical economic and financial data,
validates performance, and saves models for production use.
"""
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.risk_predictor import RiskPredictor
from src.cache.cache_manager import CacheManager
from src.core.config import settings
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline for risk prediction.
    """
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.cache_manager = CacheManager()
        self.predictor = RiskPredictor(cache_manager=self.cache_manager)
        
        # Training configuration
        self.training_config = {
            'training_period_months': 24,  # Use 2 years of data for training
            'validation_split': 0.2,
            'model_save_dir': 'models/risk_prediction',
            'performance_threshold': {
                'regression_r2': 0.6,
                'regression_mae': 15.0,
                'classification_accuracy': 0.7
            }
        }
    
    async def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete model training pipeline.
        
        Returns:
            Training results and model information
        """
        try:
            logger.info("Starting model training pipeline")
            
            # Calculate training date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.training_config['training_period_months'] * 30)
            
            logger.info(f"Training period: {start_date.date()} to {end_date.date()}")
            
            # Train models
            training_results = await self.predictor.train_models(
                start_date=start_date,
                end_date=end_date,
                validation_split=self.training_config['validation_split']
            )
            
            if not training_results['success']:
                raise RuntimeError("Model training failed")
            
            # Validate model performance
            performance_ok = self._validate_performance(training_results['evaluation'])
            
            if not performance_ok:
                logger.warning("Model performance below threshold but continuing")
            
            # Save models
            model_save_path = self.predictor.save_models(
                model_dir=self.training_config['model_save_dir']
            )
            
            # Test prediction
            test_prediction = await self.predictor.predict_risk(
                prediction_date=datetime.now(),
                horizon_days=30
            )
            
            pipeline_results = {
                'success': True,
                'training_results': training_results,
                'model_save_path': model_save_path,
                'performance_validation': performance_ok,
                'test_prediction': {
                    'risk_score': test_prediction.risk_score,
                    'risk_level': test_prediction.risk_level,
                    'confidence': test_prediction.confidence,
                    'model_version': test_prediction.model_version
                },
                'model_info': self.predictor.get_model_info()
            }
            
            logger.info("Model training pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            raise
    
    def _validate_performance(self, evaluation_metrics: Dict[str, float]) -> bool:
        """
        Validate model performance against thresholds.
        
        Args:
            evaluation_metrics: Model evaluation results
            
        Returns:
            True if performance meets thresholds
        """
        thresholds = self.training_config['performance_threshold']
        
        checks = []
        
        # Regression performance
        if 'regression_r2' in evaluation_metrics:
            r2_ok = evaluation_metrics['regression_r2'] >= thresholds['regression_r2']
            checks.append(r2_ok)
            logger.info(f"R² score: {evaluation_metrics['regression_r2']:.3f} (threshold: {thresholds['regression_r2']})")
        
        if 'regression_mae' in evaluation_metrics:
            mae_ok = evaluation_metrics['regression_mae'] <= thresholds['regression_mae']
            checks.append(mae_ok)
            logger.info(f"MAE: {evaluation_metrics['regression_mae']:.3f} (threshold: {thresholds['regression_mae']})")
        
        # Classification performance
        if 'classification_accuracy' in evaluation_metrics:
            acc_ok = evaluation_metrics['classification_accuracy'] >= thresholds['classification_accuracy']
            checks.append(acc_ok)
            logger.info(f"Accuracy: {evaluation_metrics['classification_accuracy']:.3f} (threshold: {thresholds['classification_accuracy']})")
        
        performance_ok = all(checks) if checks else False
        logger.info(f"Performance validation: {'PASS' if performance_ok else 'FAIL'}")
        
        return performance_ok
    
    async def retrain_if_needed(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Check if models need retraining and retrain if necessary.
        
        Args:
            force_retrain: Force retraining regardless of model age
            
        Returns:
            Retraining results
        """
        try:
            model_info = self.predictor.get_model_info()
            
            needs_retrain = force_retrain
            
            if model_info['status'] == 'no_model_loaded':
                needs_retrain = True
                logger.info("No model loaded - training required")
            elif model_info['status'] == 'loaded':
                # Check model age
                trained_at = datetime.fromisoformat(model_info['metadata']['trained_at'])
                days_old = (datetime.now() - trained_at).days
                
                if days_old > 30:  # Retrain monthly
                    needs_retrain = True
                    logger.info(f"Model is {days_old} days old - retraining recommended")
                else:
                    logger.info(f"Model is {days_old} days old - no retraining needed")
            
            if needs_retrain:
                logger.info("Starting model retraining")
                return await self.run_training_pipeline()
            else:
                return {
                    'success': True,
                    'action': 'no_retraining_needed',
                    'model_info': model_info
                }
                
        except Exception as e:
            logger.error(f"Error during retrain check: {e}")
            raise


async def main():
    """Main training script entry point."""
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train RiskX ML models')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force retraining regardless of model age')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if retraining is needed')
    
    args = parser.parse_args()
    
    try:
        # Initialize training pipeline
        pipeline = ModelTrainingPipeline()
        
        if args.check_only:
            # Just check if retraining is needed
            model_info = pipeline.predictor.get_model_info()
            print(f"Model status: {model_info['status']}")
            
            if model_info['status'] == 'loaded':
                trained_at = datetime.fromisoformat(model_info['metadata']['trained_at'])
                days_old = (datetime.now() - trained_at).days
                print(f"Model age: {days_old} days")
                print(f"Retraining recommended: {days_old > 30}")
            
        else:
            # Run training or retraining
            if args.force_retrain:
                results = await pipeline.run_training_pipeline()
            else:
                results = await pipeline.retrain_if_needed()
            
            print("\nTraining Results:")
            print(f"Success: {results['success']}")
            
            if 'test_prediction' in results:
                test_pred = results['test_prediction']
                print(f"Test prediction - Risk Score: {test_pred['risk_score']:.1f}")
                print(f"Test prediction - Risk Level: {test_pred['risk_level']}")
                print(f"Test prediction - Confidence: {test_pred['confidence']:.3f}")
            
            if 'model_save_path' in results:
                print(f"Models saved to: {results['model_save_path']}")
        
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())