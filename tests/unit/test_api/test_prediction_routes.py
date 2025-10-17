"""
Tests for prediction API routes
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.routes.prediction import router
from src.ml.models.risk_predictor import PredictionResult


@pytest.fixture
def app():
    """Create FastAPI app for testing"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_prediction_result():
    """Mock prediction result"""
    return PredictionResult(
        risk_score=67.5,
        confidence=0.87,
        risk_level='medium',
        prediction_date=datetime(2024, 1, 1),
        horizon_days=30,
        feature_importance={
            'unemployment_rate': 0.25,
            'inflation_rate': 0.20,
            'trade_balance': 0.15,
            'bank_stability': 0.12,
            'supply_disruption': 0.10
        },
        model_version='ml_v1.0_20240101'
    )


class TestPredictionRoutes:
    """Test prediction API endpoints"""
    
    def test_risk_forecast_endpoint_exists(self, client):
        """Test that risk forecast endpoint exists"""
        response = client.get("/api/v1/risk/forecast")
        # Should not be 404 (Not Found)
        assert response.status_code != 404
    
    @patch('src.api.routes.prediction.RiskPredictor')
    @patch('src.api.routes.prediction.RiskScorer')
    @patch('src.api.routes.prediction.CacheManager')
    def test_risk_forecast_with_ml_model(self, mock_cache, mock_scorer, mock_predictor, 
                                       client, mock_prediction_result):
        """Test risk forecast using ML model"""
        # Setup mocks
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        mock_predictor_instance.load_models.return_value = True
        mock_predictor_instance.get_model_info.return_value = {
            'metadata': {'version': 'ml_v1.0_20240101'}
        }
        mock_predictor_instance.predict_risk = AsyncMock(return_value=mock_prediction_result)
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.set = Mock()
        
        response = client.get("/api/v1/risk/forecast?horizon_days=7&include_factors=true")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'prediction_id' in data
        assert 'timestamp' in data
        assert 'horizon_days' in data
        assert 'predictions' in data
        assert 'model_version' in data
        assert data['horizon_days'] == 7
        assert len(data['predictions']) == 7  # One prediction per day
    
    @patch('src.api.routes.prediction.RiskPredictor')
    @patch('src.api.routes.prediction.RiskScorer')
    @patch('src.api.routes.prediction.CacheManager')
    def test_risk_forecast_fallback_to_basic_scorer(self, mock_cache, mock_scorer, 
                                                   mock_predictor, client):
        """Test risk forecast falls back to basic scorer when ML model unavailable"""
        # Setup mocks - ML model fails to load
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        mock_predictor_instance.load_models.return_value = False
        
        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance
        mock_scorer_instance.calculate_risk_score = AsyncMock(return_value=55.0)
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.set = Mock()
        
        response = client.get("/api/v1/risk/forecast?horizon_days=3")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['model_version'] == 'fallback_v1.0'
        assert len(data['predictions']) == 3
        
        # Check that predictions have expected structure
        for prediction in data['predictions']:
            assert 'date' in prediction
            assert 'risk_score' in prediction
            assert 'risk_level' in prediction
            assert 'confidence' in prediction
            assert prediction['risk_level'] in ['low', 'moderate', 'high']
    
    def test_risk_forecast_parameter_validation(self, client):
        """Test parameter validation for risk forecast"""
        # Test invalid horizon_days
        response = client.get("/api/v1/risk/forecast?horizon_days=400")  # > 365
        assert response.status_code == 422
        
        response = client.get("/api/v1/risk/forecast?horizon_days=0")  # < 1
        assert response.status_code == 422
        
        # Test invalid confidence_level
        response = client.get("/api/v1/risk/forecast?confidence_level=0.4")  # < 0.5
        assert response.status_code == 422
        
        response = client.get("/api/v1/risk/forecast?confidence_level=1.1")  # > 0.99
        assert response.status_code == 422
    
    @patch('src.api.routes.prediction.RiskPredictor')
    @patch('src.api.routes.prediction.RiskScorer')
    @patch('src.api.routes.prediction.CacheManager')
    def test_risk_forecast_ml_prediction_failure_fallback(self, mock_cache, mock_scorer, 
                                                         mock_predictor, client):
        """Test fallback when ML prediction fails"""
        # Setup mocks - ML model loads but prediction fails
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        mock_predictor_instance.load_models.return_value = True
        mock_predictor_instance.get_model_info.return_value = {
            'metadata': {'version': 'ml_v1.0_20240101'}
        }
        mock_predictor_instance.predict_risk = AsyncMock(
            side_effect=Exception("Prediction failed")
        )
        
        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance
        mock_scorer_instance.calculate_risk_score = AsyncMock(return_value=45.0)
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.set = Mock()
        
        response = client.get("/api/v1/risk/forecast?horizon_days=2")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should fall back to basic scorer
        assert len(data['predictions']) == 2
        assert data['model_version'] == 'fallback_v1.0'
    
    @patch('src.api.routes.prediction.RiskPredictor')
    @patch('src.api.routes.prediction.RiskScorer')
    @patch('src.api.routes.prediction.CacheManager')
    def test_risk_forecast_response_structure(self, mock_cache, mock_scorer, 
                                            mock_predictor, client, mock_prediction_result):
        """Test that response follows the correct structure"""
        # Setup mocks for ML model
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        mock_predictor_instance.load_models.return_value = True
        mock_predictor_instance.get_model_info.return_value = {
            'metadata': {'version': 'ml_v1.0_20240101'}
        }
        mock_predictor_instance.predict_risk = AsyncMock(return_value=mock_prediction_result)
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.set = Mock()
        
        response = client.get("/api/v1/risk/forecast?horizon_days=1&include_factors=true")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level structure
        required_fields = ['prediction_id', 'timestamp', 'horizon_days', 'predictions', 
                          'confidence_level', 'model_version', 'explanation']
        for field in required_fields:
            assert field in data
        
        # Check prediction structure
        prediction = data['predictions'][0]
        prediction_fields = ['date', 'risk_score', 'risk_level', 'confidence', 'contributing_factors']
        for field in prediction_fields:
            assert field in prediction
        
        # Check contributing factors structure (from ML model)
        for factor in prediction['contributing_factors']:
            assert 'name' in factor
            assert 'weight' in factor
            assert 'trend' in factor
            assert factor['trend'] == 'ml_derived'
    
    def test_risk_forecast_default_parameters(self, client):
        """Test risk forecast with default parameters"""
        with patch('src.api.routes.prediction.RiskPredictor'), \
             patch('src.api.routes.prediction.RiskScorer'), \
             patch('src.api.routes.prediction.CacheManager'):
            
            response = client.get("/api/v1/risk/forecast")
            
            # Should use default values: horizon_days=30, confidence_level=0.95, include_factors=True
            assert response.status_code == 200
            data = response.json()
            assert data['horizon_days'] == 30
            assert data['confidence_level'] == 0.95
    
    @patch('src.api.routes.prediction.RiskPredictor')
    @patch('src.api.routes.prediction.RiskScorer')
    @patch('src.api.routes.prediction.CacheManager')
    def test_risk_forecast_caching(self, mock_cache, mock_scorer, mock_predictor, client):
        """Test that predictions are cached"""
        # Setup mocks
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        mock_predictor_instance.load_models.return_value = False  # Use fallback
        
        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance
        mock_scorer_instance.calculate_risk_score = AsyncMock(return_value=50.0)
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.set = Mock()
        
        response = client.get("/api/v1/risk/forecast?horizon_days=1")
        
        assert response.status_code == 200
        
        # Verify cache was called
        mock_cache_instance.set.assert_called_once()
        call_args = mock_cache_instance.set.call_args
        assert call_args[0][0].startswith('prediction:forecast_')  # Cache key
        assert call_args[1]['ttl'] == 3600  # TTL
    
    def test_risk_forecast_explanation_included(self, client):
        """Test that explanation is included when include_factors=True"""
        with patch('src.api.routes.prediction.RiskPredictor'), \
             patch('src.api.routes.prediction.RiskScorer'), \
             patch('src.api.routes.prediction.CacheManager'):
            
            response = client.get("/api/v1/risk/forecast?include_factors=true")
            
            assert response.status_code == 200
            data = response.json()
            
            assert 'explanation' in data
            explanation = data['explanation']
            assert 'methodology' in explanation
            assert 'uncertainty_model' in explanation
            assert 'key_assumptions' in explanation
            assert 'confidence_bounds' in explanation
    
    def test_risk_forecast_no_explanation_when_disabled(self, client):
        """Test that explanation is not included when include_factors=False"""
        with patch('src.api.routes.prediction.RiskPredictor'), \
             patch('src.api.routes.prediction.RiskScorer'), \
             patch('src.api.routes.prediction.CacheManager'):
            
            response = client.get("/api/v1/risk/forecast?include_factors=false")
            
            assert response.status_code == 200
            data = response.json()
            
            # Explanation should be None
            assert data['explanation'] is None
            
            # Contributing factors should be empty
            for prediction in data['predictions']:
                assert prediction['contributing_factors'] == []