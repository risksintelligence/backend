"""
Unit tests for ML models and serving infrastructure.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.ml.models.recession_predictor import RecessionPredictor
from src.ml.models.supply_chain_risk_model import SupplyChainRiskModel
from src.ml.models.market_volatility_model import MarketVolatilityModel
from src.ml.models.geopolitical_risk_model import GeopoliticalRiskModel
from src.ml.serving.model_server import ModelServer
from src.ml.training.model_trainer import ModelTrainingPipeline


@pytest.mark.unit
class TestRecessionPredictor:
    """Test recession prediction model."""
    
    @pytest.fixture
    def recession_model(self):
        """Create recession predictor instance."""
        return RecessionPredictor()
    
    @pytest.fixture
    def sample_economic_data(self):
        """Sample economic data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
        return pd.DataFrame({
            'date': dates,
            'DGS10': np.random.normal(2.5, 0.5, len(dates)),
            'DGS2': np.random.normal(2.0, 0.4, len(dates)),
            'DGS3MO': np.random.normal(1.5, 0.3, len(dates)),
            'UNRATE': np.random.normal(5.0, 1.0, len(dates)),
            'GDP': np.random.normal(2.0, 0.5, len(dates))
        })
    
    @pytest.mark.asyncio
    async def test_recession_model_initialization(self, recession_model):
        """Test model initialization."""
        assert recession_model.model is None
        assert recession_model.scaler is None
        assert recession_model.feature_names == []
        assert recession_model.is_trained is False
    
    @pytest.mark.asyncio
    async def test_prepare_features(self, recession_model, sample_economic_data):
        """Test feature preparation."""
        features = recession_model._prepare_features(sample_economic_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'yield_curve_spread' in features.columns
        assert 'unemployment_rate' in features.columns
        assert 'gdp_growth' in features.columns
    
    @pytest.mark.asyncio
    async def test_model_training(self, recession_model):
        """Test model training process."""
        with patch.object(recession_model, '_create_recession_training_data') as mock_data:
            # Mock training data
            X_train = np.random.random((100, 5))
            y_train = np.random.randint(0, 2, 100)
            mock_data.return_value = (X_train, y_train)
            
            # Train model
            result = await recession_model.train()
            
            assert result['status'] == 'success'
            assert recession_model.is_trained is True
            assert recession_model.model is not None
            assert recession_model.scaler is not None
    
    @pytest.mark.asyncio
    async def test_prediction_with_trained_model(self, recession_model, sample_economic_data):
        """Test prediction with trained model."""
        # Mock trained model
        recession_model.is_trained = True
        recession_model.model = MagicMock()
        recession_model.scaler = MagicMock()
        recession_model.feature_names = ['yield_curve_spread', 'unemployment_rate', 'gdp_growth']
        
        # Mock model prediction
        recession_model.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        recession_model.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        result = await recession_model.predict(sample_economic_data)
        
        assert result['status'] == 'success'
        assert 'prediction' in result
        assert 'probability' in result['prediction']
        assert 0 <= result['prediction']['probability'] <= 1
    
    @pytest.mark.asyncio
    async def test_prediction_without_trained_model(self, recession_model, sample_economic_data):
        """Test prediction fails without trained model."""
        result = await recession_model.predict(sample_economic_data)
        
        assert result['status'] == 'error'
        assert 'not trained' in result['error'].lower()


@pytest.mark.unit
class TestSupplyChainRiskModel:
    """Test supply chain risk model."""
    
    @pytest.fixture
    def supply_chain_model(self):
        """Create supply chain risk model instance."""
        return SupplyChainRiskModel()
    
    @pytest.fixture
    def sample_supply_chain_data(self):
        """Sample supply chain data for testing."""
        return {
            'shipping_costs': [100, 120, 110, 130],
            'delivery_delays': [2, 5, 3, 8],
            'inventory_levels': [80, 60, 70, 45],
            'supplier_reliability': [95, 85, 90, 75]
        }
    
    @pytest.mark.asyncio
    async def test_supply_chain_model_initialization(self, supply_chain_model):
        """Test model initialization."""
        assert supply_chain_model.model is None
        assert supply_chain_model.scaler is None
        assert supply_chain_model.is_trained is False
    
    @pytest.mark.asyncio
    async def test_risk_calculation(self, supply_chain_model, sample_supply_chain_data):
        """Test risk calculation."""
        risk_score = supply_chain_model._calculate_risk_factors(sample_supply_chain_data)
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
    
    @pytest.mark.asyncio
    async def test_feature_engineering(self, supply_chain_model, sample_supply_chain_data):
        """Test feature engineering."""
        features = supply_chain_model._engineer_features(sample_supply_chain_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    @pytest.mark.asyncio
    async def test_prediction(self, supply_chain_model, sample_supply_chain_data):
        """Test supply chain risk prediction."""
        # Mock trained model
        supply_chain_model.is_trained = True
        supply_chain_model.model = MagicMock()
        supply_chain_model.scaler = MagicMock()
        
        supply_chain_model.model.predict.return_value = np.array([65.5])
        supply_chain_model.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        
        result = await supply_chain_model.predict(sample_supply_chain_data)
        
        assert result['status'] == 'success'
        assert 'prediction' in result
        assert 'risk_score' in result['prediction']


@pytest.mark.unit
class TestModelServer:
    """Test ML model serving infrastructure."""
    
    @pytest.fixture
    def model_server(self):
        """Create model server instance."""
        return ModelServer()
    
    @pytest.mark.asyncio
    async def test_model_server_initialization(self, model_server):
        """Test model server initialization."""
        assert model_server.models == {}
        assert model_server.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_model_loading(self, model_server):
        """Test model loading."""
        with patch.object(model_server, '_load_model') as mock_load:
            mock_load.return_value = MagicMock()
            
            await model_server.initialize()
            
            assert model_server.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_assessment(self, model_server):
        """Test comprehensive risk assessment."""
        # Mock all models
        mock_recession = AsyncMock()
        mock_supply_chain = AsyncMock()
        mock_volatility = AsyncMock()
        mock_geopolitical = AsyncMock()
        
        mock_recession.predict.return_value = {
            'status': 'success',
            'prediction': {'probability': 0.25}
        }
        mock_supply_chain.predict.return_value = {
            'status': 'success',
            'prediction': {'risk_score': 45.0}
        }
        mock_volatility.predict.return_value = {
            'status': 'success',
            'prediction': {'volatility_score': 35.0}
        }
        mock_geopolitical.predict.return_value = {
            'status': 'success',
            'prediction': {'risk_score': 55.0}
        }
        
        model_server.models = {
            'recession_predictor': mock_recession,
            'supply_chain_risk': mock_supply_chain,
            'market_volatility': mock_volatility,
            'geopolitical_risk': mock_geopolitical
        }
        model_server.is_initialized = True
        
        result = await model_server.get_comprehensive_risk_assessment()
        
        assert 'overall_risk_score' in result
        assert 'confidence' in result
        assert 'factors' in result
        assert isinstance(result['overall_risk_score'], float)
    
    @pytest.mark.asyncio
    async def test_individual_model_predictions(self, model_server):
        """Test individual model predictions."""
        mock_model = AsyncMock()
        mock_model.predict.return_value = {
            'status': 'success',
            'prediction': {'probability': 0.35}
        }
        
        model_server.models = {'recession_predictor': mock_model}
        model_server.is_initialized = True
        
        result = await model_server.predict_recession_probability()
        
        assert result['status'] == 'success'
        assert 'prediction' in result
    
    @pytest.mark.asyncio
    async def test_model_status(self, model_server):
        """Test model status reporting."""
        model_server.models = {
            'recession_predictor': MagicMock(is_trained=True),
            'supply_chain_risk': MagicMock(is_trained=False)
        }
        model_server.is_initialized = True
        
        status = model_server.get_model_status()
        
        assert 'total_models' in status
        assert 'trained_models' in status
        assert 'model_details' in status
        assert status['total_models'] == 2
        assert status['trained_models'] == 1


@pytest.mark.unit
class TestModelTrainingPipeline:
    """Test model training pipeline."""
    
    @pytest.fixture
    def training_pipeline(self):
        """Create training pipeline instance."""
        return ModelTrainingPipeline()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, training_pipeline):
        """Test pipeline initialization."""
        assert training_pipeline.models == {}
        assert training_pipeline.training_results == {}
    
    @pytest.mark.asyncio
    async def test_training_data_fetching(self, training_pipeline):
        """Test training data fetching."""
        with patch('src.data.sources.fred.get_key_indicators') as mock_fred:
            with patch('src.data.sources.bea.get_gdp_data') as mock_bea:
                mock_fred.return_value = {'indicators': {'GDP': {}}}
                mock_bea.return_value = {'gdp_data': []}
                
                data = await training_pipeline.fetch_training_data()
                
                assert isinstance(data, dict)
                assert len(data) > 0
    
    @pytest.mark.asyncio
    async def test_model_training_execution(self, training_pipeline):
        """Test model training execution."""
        with patch.object(training_pipeline, 'fetch_training_data') as mock_fetch:
            with patch.object(training_pipeline, '_train_individual_model') as mock_train:
                mock_fetch.return_value = {'training_data': 'mock_data'}
                mock_train.return_value = {'status': 'success', 'accuracy': 0.85}
                
                results = await training_pipeline.train_all_models()
                
                assert results['status'] == 'success'
                assert 'training_summary' in results
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, training_pipeline):
        """Test model evaluation."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        
        X_test = np.random.random((4, 3))
        y_test = np.array([0, 1, 1, 1])
        
        metrics = training_pipeline._evaluate_model(mock_model, X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert isinstance(metrics['accuracy'], float)


@pytest.mark.unit
class TestMarketVolatilityModel:
    """Test market volatility model."""
    
    @pytest.fixture
    def volatility_model(self):
        """Create market volatility model instance."""
        return MarketVolatilityModel()
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        return pd.DataFrame({
            'date': dates,
            'price': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates)),
            'vix': np.random.normal(20, 5, len(dates))
        })
    
    @pytest.mark.asyncio
    async def test_volatility_calculation(self, volatility_model, sample_market_data):
        """Test volatility calculation."""
        volatility = volatility_model._calculate_volatility(sample_market_data['price'])
        
        assert isinstance(volatility, float)
        assert volatility >= 0
    
    @pytest.mark.asyncio
    async def test_volatility_prediction(self, volatility_model, sample_market_data):
        """Test volatility prediction."""
        # Mock trained model
        volatility_model.is_trained = True
        volatility_model.model = MagicMock()
        volatility_model.scaler = MagicMock()
        
        volatility_model.model.predict.return_value = np.array([25.5])
        volatility_model.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        result = await volatility_model.predict(sample_market_data)
        
        assert result['status'] == 'success'
        assert 'prediction' in result
        assert 'volatility_score' in result['prediction']


@pytest.mark.unit
class TestGeopoliticalRiskModel:
    """Test geopolitical risk model."""
    
    @pytest.fixture
    def geopolitical_model(self):
        """Create geopolitical risk model instance."""
        return GeopoliticalRiskModel()
    
    @pytest.fixture
    def sample_geopolitical_data(self):
        """Sample geopolitical data for testing."""
        return {
            'conflict_index': 45.0,
            'sanctions_count': 15,
            'diplomatic_relations': 60.0,
            'trade_restrictions': 25.0,
            'political_stability': 70.0
        }
    
    @pytest.mark.asyncio
    async def test_geopolitical_risk_calculation(self, geopolitical_model, sample_geopolitical_data):
        """Test geopolitical risk calculation."""
        risk_score = geopolitical_model._calculate_composite_risk(sample_geopolitical_data)
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
    
    @pytest.mark.asyncio
    async def test_geopolitical_prediction(self, geopolitical_model, sample_geopolitical_data):
        """Test geopolitical risk prediction."""
        # Mock trained model
        geopolitical_model.is_trained = True
        geopolitical_model.model = MagicMock()
        geopolitical_model.scaler = MagicMock()
        
        geopolitical_model.model.predict.return_value = np.array([55.5])
        geopolitical_model.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        
        result = await geopolitical_model.predict(sample_geopolitical_data)
        
        assert result['status'] == 'success'
        assert 'prediction' in result
        assert 'risk_score' in result['prediction']


@pytest.mark.unit
class TestMLModelErrorHandling:
    """Test ML model error handling."""
    
    @pytest.mark.asyncio
    async def test_prediction_with_invalid_data(self):
        """Test prediction with invalid data."""
        model = RecessionPredictor()
        
        # Test with empty data
        result = await model.predict(pd.DataFrame())
        assert result['status'] == 'error'
        
        # Test with missing columns
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = await model.predict(invalid_data)
        assert result['status'] == 'error'
    
    @pytest.mark.asyncio
    async def test_model_training_with_insufficient_data(self):
        """Test model training with insufficient data."""
        model = RecessionPredictor()
        
        with patch.object(model, '_create_recession_training_data') as mock_data:
            # Mock insufficient training data
            mock_data.return_value = (np.array([[1, 2]]), np.array([1]))  # Only 1 sample
            
            result = await model.train()
            assert result['status'] == 'error'
    
    @pytest.mark.asyncio
    async def test_model_server_with_uninitialized_models(self):
        """Test model server with uninitialized models."""
        model_server = ModelServer()
        
        result = await model_server.get_comprehensive_risk_assessment()
        
        assert 'error' in result or result.get('overall_risk_score') == 0.0
    
    @pytest.mark.asyncio
    async def test_model_server_with_failed_model(self):
        """Test model server with failed model prediction."""
        model_server = ModelServer()
        
        mock_model = AsyncMock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        model_server.models = {'recession_predictor': mock_model}
        model_server.is_initialized = True
        
        result = await model_server.predict_recession_probability()
        
        assert result['status'] == 'error'
        assert 'error' in result


@pytest.mark.unit
class TestMLModelIntegration:
    """Test ML model integration with external data sources."""
    
    @pytest.mark.asyncio
    async def test_recession_model_with_fred_data(self):
        """Test recession model with FRED data integration."""
        model = RecessionPredictor()
        
        with patch('src.data.sources.fred.get_multiple_series') as mock_fred:
            mock_fred.return_value = {
                'DGS10': {'observations': [{'date': '2024-01-01', 'value': '2.5'}]},
                'DGS2': {'observations': [{'date': '2024-01-01', 'value': '2.0'}]},
                'UNRATE': {'observations': [{'date': '2024-01-01', 'value': '3.5'}]}
            }
            
            training_data = await model._create_recession_training_data()
            
            assert isinstance(training_data, tuple)
            assert len(training_data) == 2  # X and y
    
    @pytest.mark.asyncio
    async def test_supply_chain_model_with_external_data(self):
        """Test supply chain model with external data sources."""
        model = SupplyChainRiskModel()
        
        with patch('src.data.sources.cisa.get_cybersecurity_threats') as mock_cisa:
            with patch('src.data.sources.supply_chain.get_supply_chain_risks') as mock_supply:
                mock_cisa.return_value = {'overall_cybersecurity_risk': 45.0}
                mock_supply.return_value = {'overall_supply_chain_risk': 55.0}
                
                result = await model._fetch_external_data()
                
                assert isinstance(result, dict)
                assert len(result) > 0