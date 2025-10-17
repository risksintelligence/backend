"""
Tests for ML Risk Predictor models
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.ml.models.risk_predictor import RiskPredictor, PredictionResult, ModelMetadata
from src.cache.cache_manager import CacheManager


class TestRiskPredictor:
    """Test ML-based risk prediction models"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing"""
        cache = Mock(spec=CacheManager)
        cache.get = Mock(return_value=None)
        cache.set = Mock(return_value=True)
        cache.delete = Mock(return_value=True)
        return cache
    
    @pytest.fixture
    def risk_predictor(self, mock_cache_manager):
        """Create risk predictor instance for testing"""
        return RiskPredictor(cache_manager=mock_cache_manager)
    
    @pytest.fixture
    def sample_features_df(self):
        """Sample feature DataFrame for testing"""
        return pd.DataFrame({
            'unemployment_trend': [0.1, 0.15, 0.05],
            'inflation_trend': [0.02, 0.03, 0.01],
            'gdp_growth_trend': [0.02, 0.015, 0.025],
            'banking_stability': [85.0, 78.0, 92.0],
            'credit_risk_score': [0.3, 0.5, 0.2],
            'trade_disruption_index': [0.4, 0.6, 0.3],
            'cyber_incidents_score': [0.2, 0.4, 0.1],
            'natural_disaster_risk': [0.3, 0.5, 0.2]
        })
    
    def test_initialization(self, risk_predictor):
        """Test risk predictor initialization"""
        assert risk_predictor.cache_manager is not None
        assert risk_predictor.economic_engineer is not None
        assert risk_predictor.financial_engineer is not None
        assert risk_predictor.supply_chain_engineer is not None
        assert risk_predictor.disruption_engineer is not None
        assert risk_predictor.regression_model is None  # Not trained yet
        assert risk_predictor.classification_model is None  # Not trained yet
        assert len(risk_predictor.model_config) == 2
        assert 'regression' in risk_predictor.model_config
        assert 'classification' in risk_predictor.model_config
    
    @pytest.mark.asyncio
    async def test_prepare_features(self, risk_predictor):
        """Test feature preparation"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Mock all feature engineers with their correct method names and return types
        from src.ml.features.economic import EconomicFeatures
        from src.ml.features.financial import FinancialFeatures
        from src.ml.features.supply_chain import SupplyChainFeatures  
        from src.ml.features.disruption import DisruptionFeatures
        
        mock_economic_features = EconomicFeatures(
            features={'gdp_growth_12m': 2.1, 'unemployment_rate_trend_slope': 0.1},
            metadata={'feature_count': 2},
            timestamp=datetime.now(),
            data_quality_score=0.9
        )
        
        mock_financial_features = FinancialFeatures(
            bank_stability_score=85.0,
            credit_stress_index=0.3,
            liquidity_ratio=1.2,
            capital_adequacy_ratio=0.12,
            non_performing_loans_ratio=0.02,
            interest_rate_risk=0.15,
            systemic_risk_score=0.25,
            market_volatility=0.18,
            features={'banking_stability': 85.0, 'credit_risk_score': 0.3},
            metadata={'feature_count': 2}
        )
        
        mock_supply_chain_features = SupplyChainFeatures(
            trade_disruption_index=0.4,
            port_congestion_score=0.3,
            supplier_diversity_index=0.7,
            logistics_performance_score=0.8,
            trade_dependency_ratio=0.6,
            inventory_turnover_rate=2.5,
            shipping_cost_index=0.2,
            supply_chain_resilience_score=0.75,
            features={'trade_disruption_index': 0.4, 'supply_chain_risk': 0.3},
            metadata={'feature_count': 2}
        )
        
        mock_disruption_features = DisruptionFeatures(
            natural_disaster_risk=0.3,
            cyber_incident_frequency=0.2,
            geopolitical_tension_index=0.25,
            pandemic_disruption_level=0.1,
            climate_anomaly_score=0.4,
            infrastructure_vulnerability=0.35,
            social_unrest_indicator=0.15,
            overall_disruption_risk=0.28,
            features={'cyber_incidents_score': 0.2, 'natural_disaster_risk': 0.3},
            metadata={'feature_count': 2}
        )
        
        with patch.object(risk_predictor.economic_engineer, 'engineer_features', 
                         new_callable=AsyncMock, return_value=mock_economic_features), \
             patch.object(risk_predictor.financial_engineer, 'extract_features',
                         new_callable=AsyncMock, return_value=mock_financial_features), \
             patch.object(risk_predictor.supply_chain_engineer, 'extract_features',
                         new_callable=AsyncMock, return_value=mock_supply_chain_features), \
             patch.object(risk_predictor.disruption_engineer, 'extract_features',
                         new_callable=AsyncMock, return_value=mock_disruption_features):
            
            features_df = await risk_predictor.prepare_features(start_date, end_date)
            
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) == 1  # Single row of features
            assert len(risk_predictor.feature_names) > 0
    
    def test_generate_synthetic_targets(self, risk_predictor, sample_features_df):
        """Test synthetic target generation"""
        continuous_targets, categorical_targets = risk_predictor._generate_synthetic_targets(
            sample_features_df
        )
        
        assert len(continuous_targets) == len(sample_features_df)
        assert len(categorical_targets) == len(sample_features_df)
        assert all(0 <= score <= 100 for score in continuous_targets)
        assert all(cat in ['low', 'medium', 'high', 'critical'] for cat in categorical_targets)
    
    @pytest.mark.asyncio
    async def test_train_models(self, risk_predictor, sample_features_df):
        """Test model training"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Mock feature preparation
        with patch.object(risk_predictor, 'prepare_features', 
                         new_callable=AsyncMock, return_value=sample_features_df):
            
            results = await risk_predictor.train_models(start_date, end_date)
            
            assert results['success'] is True
            assert 'metadata' in results
            assert 'evaluation' in results
            assert 'feature_count' in results
            
            # Check that models are trained
            assert risk_predictor.regression_model is not None
            assert risk_predictor.classification_model is not None
            assert risk_predictor.model_metadata is not None
            
            # Check metadata
            metadata = risk_predictor.model_metadata
            assert metadata.model_name == "RiskPredictor"
            assert metadata.model_type == "RandomForest_Ensemble"
            assert isinstance(metadata.trained_at, datetime)
            assert len(metadata.features_used) > 0
    
    @pytest.mark.asyncio
    async def test_predict_risk(self, risk_predictor, sample_features_df):
        """Test risk prediction"""
        # First train the model
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        with patch.object(risk_predictor, 'prepare_features', 
                         new_callable=AsyncMock, return_value=sample_features_df):
            await risk_predictor.train_models(start_date, end_date)
        
        # Now test prediction
        prediction_date = datetime(2024, 1, 1)
        
        with patch.object(risk_predictor, 'prepare_features', 
                         new_callable=AsyncMock, return_value=sample_features_df.iloc[:1]):
            
            result = await risk_predictor.predict_risk(prediction_date, horizon_days=30)
            
            assert isinstance(result, PredictionResult)
            assert 0 <= result.risk_score <= 100
            assert 0 <= result.confidence <= 1
            assert result.risk_level in ['low', 'medium', 'high', 'critical']
            assert result.prediction_date == prediction_date
            assert result.horizon_days == 30
            assert isinstance(result.feature_importance, dict)
            assert len(result.feature_importance) > 0
    
    def test_predict_risk_without_trained_model(self, risk_predictor):
        """Test prediction fails without trained model"""
        prediction_date = datetime(2024, 1, 1)
        
        with pytest.raises(ValueError, match="Models not trained"):
            import asyncio
            asyncio.run(risk_predictor.predict_risk(prediction_date))
    
    def test_model_saving_and_loading(self, risk_predictor, sample_features_df, tmp_path):
        """Test model persistence"""
        # Train models first
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        with patch.object(risk_predictor, 'prepare_features', 
                         new_callable=AsyncMock, return_value=sample_features_df):
            import asyncio
            asyncio.run(risk_predictor.train_models(start_date, end_date))
        
        # Save models
        save_path = risk_predictor.save_models(str(tmp_path))
        assert save_path == str(tmp_path)
        
        # Create new predictor and load models
        new_predictor = RiskPredictor()
        load_success = new_predictor.load_models(str(tmp_path))
        
        assert load_success is True
        assert new_predictor.regression_model is not None
        assert new_predictor.classification_model is not None
        assert new_predictor.model_metadata is not None
    
    def test_get_model_info_no_model(self, risk_predictor):
        """Test model info without trained model"""
        info = risk_predictor.get_model_info()
        assert info['status'] == 'no_model_loaded'
    
    def test_get_model_info_with_model(self, risk_predictor, sample_features_df):
        """Test model info with trained model"""
        # Train model first
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        with patch.object(risk_predictor, 'prepare_features', 
                         new_callable=AsyncMock, return_value=sample_features_df):
            import asyncio
            asyncio.run(risk_predictor.train_models(start_date, end_date))
        
        info = risk_predictor.get_model_info()
        
        assert info['status'] == 'loaded'
        assert 'metadata' in info
        assert 'performance' in info
        assert 'config' in info
        assert info['metadata']['name'] == 'RiskPredictor'
        assert info['metadata']['type'] == 'RandomForest_Ensemble'
    
    def test_model_evaluation_metrics(self, risk_predictor, sample_features_df):
        """Test model evaluation produces proper metrics"""
        # Create larger sample for better evaluation
        larger_sample = pd.concat([sample_features_df] * 10, ignore_index=True)
        
        # Generate targets
        y_reg, y_clf = risk_predictor._generate_synthetic_targets(larger_sample)
        
        # Mock trained models
        risk_predictor.regression_model = Mock()
        risk_predictor.classification_model = Mock()
        risk_predictor.feature_names = list(larger_sample.columns)
        
        # Mock predictions
        risk_predictor.regression_model.predict.return_value = y_reg[:10]
        risk_predictor.classification_model.predict.return_value = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        risk_predictor.regression_model.feature_importances_ = np.random.rand(len(larger_sample.columns))
        risk_predictor.classification_model.feature_importances_ = np.random.rand(len(larger_sample.columns))
        
        # Test evaluation
        X_val = larger_sample.values[:10]
        y_reg_val = y_reg[:10]
        y_clf_val = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_clf_labels = ['low', 'medium', 'high'] * 3 + ['low']
        
        metrics = risk_predictor._evaluate_models(X_val, y_reg_val, y_clf_val, y_clf_labels)
        
        assert 'regression_mse' in metrics
        assert 'regression_mae' in metrics
        assert 'regression_r2' in metrics
        assert 'regression_rmse' in metrics
        assert 'classification_accuracy' in metrics
        assert 'top_regression_features' in metrics
        assert 'top_classification_features' in metrics


class TestPredictionResult:
    """Test PredictionResult dataclass"""
    
    def test_prediction_result_creation(self):
        """Test creating prediction result"""
        result = PredictionResult(
            risk_score=75.5,
            confidence=0.85,
            risk_level='high',
            prediction_date=datetime(2024, 1, 1),
            horizon_days=30,
            feature_importance={'unemployment': 0.3, 'inflation': 0.2},
            model_version='v1.0_test'
        )
        
        assert result.risk_score == 75.5
        assert result.confidence == 0.85
        assert result.risk_level == 'high'
        assert result.horizon_days == 30
        assert len(result.feature_importance) == 2
        assert result.model_version == 'v1.0_test'


class TestModelMetadata:
    """Test ModelMetadata dataclass"""
    
    def test_model_metadata_creation(self):
        """Test creating model metadata"""
        features = ['feature1', 'feature2', 'feature3']
        metrics = {'mse': 0.15, 'r2': 0.85}
        data_period = (datetime(2023, 1, 1), datetime(2023, 12, 31))
        
        metadata = ModelMetadata(
            model_name='TestModel',
            model_type='RandomForest',
            version='v1.0_test',
            trained_at=datetime.now(),
            features_used=features,
            performance_metrics=metrics,
            data_period=data_period
        )
        
        assert metadata.model_name == 'TestModel'
        assert metadata.model_type == 'RandomForest'
        assert metadata.version == 'v1.0_test'
        assert len(metadata.features_used) == 3
        assert metadata.performance_metrics['r2'] == 0.85
        assert metadata.data_period == data_period