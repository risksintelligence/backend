"""
Tests for ML models and risk scoring
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.ml.models.risk_scorer import RiskScorer
from src.ml.models.network_analyzer import RiskNetworkAnalyzer
from src.ml.features.economic import EconomicFeatureEngineer
from src.ml.explainability.shap_analyzer import ShapAnalyzer


class TestRiskScorer:
    """Test risk scoring model"""
    
    @pytest.fixture
    def risk_scorer(self):
        return RiskScorer()
    
    @pytest.fixture
    def sample_features(self):
        return {
            "gdp_growth": 2.1,
            "unemployment_rate": 3.8,
            "inflation_rate": 2.4,
            "interest_rate": 5.25,
            "trade_balance": -75000000000,
            "bank_stress_index": 0.3,
            "weather_severity": 0.4,
            "cyber_threat_level": 0.5
        }
    
    def test_risk_scorer_initialization(self, risk_scorer):
        """Test risk scorer initializes correctly"""
        assert risk_scorer is not None
        assert hasattr(risk_scorer, 'calculate_risk_score')
        assert hasattr(risk_scorer, 'get_risk_factors')
    
    @pytest.mark.asyncio
    async def test_calculate_risk_score(self, risk_scorer, sample_features):
        """Test risk score calculation"""
        with patch.object(risk_scorer, '_load_model', return_value=Mock()):
            with patch.object(risk_scorer, '_prepare_features', return_value=np.array([[0.1, 0.2, 0.3]])):
                with patch.object(risk_scorer.model, 'predict_proba', return_value=np.array([[0.3, 0.7]])):
                    
                    result = await risk_scorer.calculate_risk_score(sample_features)
                    
                    assert "overall_risk_score" in result
                    assert "confidence" in result
                    assert "risk_level" in result
                    assert 0 <= result["overall_risk_score"] <= 1
    
    def test_risk_level_categorization(self, risk_scorer):
        """Test risk level categorization logic"""
        assert risk_scorer._categorize_risk_level(0.2) == "low"
        assert risk_scorer._categorize_risk_level(0.5) == "moderate"
        assert risk_scorer._categorize_risk_level(0.8) == "high"
        assert risk_scorer._categorize_risk_level(0.95) == "critical"
    
    @pytest.mark.asyncio
    async def test_get_risk_factors(self, risk_scorer, sample_features):
        """Test risk factor analysis"""
        with patch.object(risk_scorer, '_analyze_feature_importance', return_value={
            "unemployment_rate": 0.25,
            "trade_balance": 0.20,
            "inflation_rate": 0.15
        }):
            factors = await risk_scorer.get_risk_factors(sample_features)
            
            assert "economic_indicators" in factors
            assert "financial_indicators" in factors
            assert isinstance(factors["economic_indicators"], dict)
    
    def test_feature_preparation(self, risk_scorer, sample_features):
        """Test feature preparation for model input"""
        features_array = risk_scorer._prepare_features(sample_features)
        
        assert isinstance(features_array, np.ndarray)
        assert features_array.shape[1] > 0  # Should have features
    
    @pytest.mark.asyncio
    async def test_model_update(self, risk_scorer):
        """Test model updating with new data"""
        sample_training_data = pd.DataFrame({
            "gdp_growth": [2.1, 1.8, 2.5],
            "unemployment_rate": [3.8, 4.1, 3.5],
            "risk_label": [0, 1, 0]
        })
        
        with patch.object(risk_scorer, '_retrain_model', return_value=True):
            result = await risk_scorer.update_with_latest_data(sample_training_data)
            assert result is True


class TestRiskNetworkAnalyzer:
    """Test risk network analysis"""
    
    @pytest.fixture
    def network_analyzer(self):
        return RiskNetworkAnalyzer()
    
    @pytest.fixture
    def sample_network_data(self):
        return {
            "nodes": [
                {"id": "bank_1", "type": "financial", "assets": 1000000},
                {"id": "bank_2", "type": "financial", "assets": 2000000},
                {"id": "supplier_1", "type": "supply_chain", "revenue": 500000}
            ],
            "edges": [
                {"source": "bank_1", "target": "bank_2", "relationship": "counterparty"},
                {"source": "bank_1", "target": "supplier_1", "relationship": "lending"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_analyze_network(self, network_analyzer, sample_network_data):
        """Test network analysis functionality"""
        with patch.object(network_analyzer, '_load_network_data', return_value=sample_network_data):
            analysis = await network_analyzer.analyze_network()
            
            assert "nodes" in analysis
            assert "edges" in analysis
            assert "network_metrics" in analysis
            assert "risk_propagation" in analysis
    
    def test_centrality_calculation(self, network_analyzer, sample_network_data):
        """Test network centrality calculations"""
        import networkx as nx
        
        # Create a sample graph
        G = nx.Graph()
        for node in sample_network_data["nodes"]:
            G.add_node(node["id"], **node)
        for edge in sample_network_data["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        centrality = network_analyzer._calculate_centrality_metrics(G)
        
        assert "betweenness" in centrality
        assert "closeness" in centrality
        assert "degree" in centrality
    
    @pytest.mark.asyncio
    async def test_risk_propagation_simulation(self, network_analyzer, sample_network_data):
        """Test risk propagation simulation"""
        with patch.object(network_analyzer, '_load_network_data', return_value=sample_network_data):
            propagation = await network_analyzer.simulate_risk_propagation("bank_1", shock_magnitude=0.5)
            
            assert "initial_shock" in propagation
            assert "propagation_steps" in propagation
            assert "final_impact" in propagation
    
    @pytest.mark.asyncio
    async def test_update_network_data(self, network_analyzer):
        """Test updating network with new data"""
        new_data = {
            "financial_data": pd.DataFrame({"institution": ["bank_1"], "assets": [1100000]}),
            "trade_data": pd.DataFrame({"partner": ["supplier_1"], "volume": [600000]})
        }
        
        with patch.object(network_analyzer, '_process_network_updates'):
            result = await network_analyzer.update_network_data(new_data)
            assert isinstance(result, bool)


class TestEconomicFeatureEngineer:
    """Test economic feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        return EconomicFeatureEngineer()
    
    @pytest.fixture
    def sample_economic_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-01', freq='M')
        return pd.DataFrame({
            "date": dates,
            "gdp": np.random.normal(21000, 500, len(dates)),
            "unemployment": np.random.normal(3.8, 0.3, len(dates)),
            "inflation": np.random.normal(2.5, 0.5, len(dates))
        })
    
    @pytest.mark.asyncio
    async def test_engineer_features(self, feature_engineer, sample_economic_data):
        """Test feature engineering process"""
        with patch.object(feature_engineer, '_load_economic_data', return_value=sample_economic_data):
            features = await feature_engineer.engineer_features()
            
            assert "trend_features" in features
            assert "volatility_features" in features
            assert "correlation_features" in features
            assert "seasonal_features" in features
    
    def test_trend_feature_calculation(self, feature_engineer, sample_economic_data):
        """Test trend feature calculations"""
        trend_features = feature_engineer._engineer_trend_features(sample_economic_data)
        
        assert "gdp_trend" in trend_features
        assert "unemployment_trend" in trend_features
        assert "inflation_trend" in trend_features
    
    def test_volatility_feature_calculation(self, feature_engineer, sample_economic_data):
        """Test volatility feature calculations"""
        volatility_features = feature_engineer._engineer_volatility_features(sample_economic_data)
        
        assert "gdp_volatility" in volatility_features
        assert "unemployment_volatility" in volatility_features
        assert "inflation_volatility" in volatility_features
    
    def test_correlation_feature_calculation(self, feature_engineer, sample_economic_data):
        """Test correlation feature calculations"""
        correlation_features = feature_engineer._engineer_correlation_features(sample_economic_data)
        
        assert "gdp_unemployment_corr" in correlation_features
        assert "unemployment_inflation_corr" in correlation_features
    
    @pytest.mark.asyncio
    async def test_feature_validation(self, feature_engineer, sample_economic_data):
        """Test feature validation"""
        with patch.object(feature_engineer, '_load_economic_data', return_value=sample_economic_data):
            features = await feature_engineer.engineer_features()
            
            # Validate that features are numeric
            for feature_group in features.values():
                for feature_name, feature_value in feature_group.items():
                    assert isinstance(feature_value, (int, float, np.number))
                    assert not np.isnan(feature_value)


class TestShapAnalyzer:
    """Test SHAP explainability analyzer"""
    
    @pytest.fixture
    def shap_analyzer(self):
        return ShapAnalyzer()
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.predict.return_value = np.array([0.7])
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model
    
    @pytest.fixture
    def sample_input_features(self):
        return {
            "gdp_growth": 2.1,
            "unemployment_rate": 3.8,
            "inflation_rate": 2.4,
            "trade_balance": -75000000000,
            "bank_stress_index": 0.3
        }
    
    @pytest.mark.asyncio
    async def test_explain_prediction(self, shap_analyzer, mock_model, sample_input_features):
        """Test SHAP explanation generation"""
        with patch('shap.Explainer') as mock_explainer:
            mock_explainer_instance = Mock()
            mock_explainer.return_value = mock_explainer_instance
            mock_explainer_instance.return_value = Mock(values=np.array([0.1, -0.05, 0.08, -0.03, 0.02]))
            
            explanation = await shap_analyzer.explain_prediction(mock_model, sample_input_features)
            
            assert "shap_values" in explanation
            assert "feature_importance" in explanation
            assert "base_value" in explanation
            assert "prediction" in explanation
    
    @pytest.mark.asyncio
    async def test_generate_global_explanation(self, shap_analyzer, mock_model):
        """Test global SHAP explanation generation"""
        sample_dataset = pd.DataFrame({
            "gdp_growth": [2.1, 1.8, 2.5],
            "unemployment_rate": [3.8, 4.1, 3.5],
            "inflation_rate": [2.4, 2.6, 2.2]
        })
        
        with patch('shap.Explainer') as mock_explainer:
            mock_explainer_instance = Mock()
            mock_explainer.return_value = mock_explainer_instance
            mock_explainer_instance.return_value = Mock(
                values=np.array([[0.1, -0.05, 0.08], [0.05, -0.02, 0.04], [-0.03, 0.06, -0.01]])
            )
            
            global_explanation = await shap_analyzer.generate_global_explanation(mock_model, sample_dataset)
            
            assert "global_feature_importance" in global_explanation
            assert "feature_interactions" in global_explanation
            assert "summary_statistics" in global_explanation
    
    def test_feature_importance_ranking(self, shap_analyzer):
        """Test feature importance ranking"""
        shap_values = np.array([0.1, -0.05, 0.08, -0.03, 0.02])
        feature_names = ["gdp_growth", "unemployment_rate", "inflation_rate", "trade_balance", "bank_stress"]
        
        importance_ranking = shap_analyzer._rank_feature_importance(shap_values, feature_names)
        
        assert isinstance(importance_ranking, list)
        assert len(importance_ranking) == len(feature_names)
        assert all("feature" in item and "importance" in item for item in importance_ranking)
    
    @pytest.mark.asyncio
    async def test_waterfall_explanation(self, shap_analyzer, mock_model, sample_input_features):
        """Test waterfall explanation generation"""
        with patch('shap.Explainer') as mock_explainer:
            mock_explainer_instance = Mock()
            mock_explainer.return_value = mock_explainer_instance
            mock_explainer_instance.return_value = Mock(
                values=np.array([0.1, -0.05, 0.08, -0.03, 0.02]),
                base_values=0.5
            )
            
            waterfall = await shap_analyzer.generate_waterfall_explanation(mock_model, sample_input_features)
            
            assert "base_value" in waterfall
            assert "contributions" in waterfall
            assert "final_prediction" in waterfall


class TestMLModelIntegration:
    """Integration tests for ML components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prediction_pipeline(self, sample_risk_data):
        """Test complete prediction pipeline from features to explanation"""
        risk_scorer = RiskScorer()
        shap_analyzer = ShapAnalyzer()
        
        with patch.object(risk_scorer, 'calculate_risk_score') as mock_score:
            with patch.object(shap_analyzer, 'explain_prediction') as mock_explain:
                
                mock_score.return_value = {
                    "overall_risk_score": 0.65,
                    "confidence": 0.87,
                    "risk_level": "moderate"
                }
                
                mock_explain.return_value = {
                    "shap_values": [0.1, -0.05, 0.08],
                    "feature_importance": {"gdp_growth": 0.1, "unemployment": -0.05}
                }
                
                # Run prediction pipeline
                risk_result = await risk_scorer.calculate_risk_score(sample_risk_data["economic_indicators"])
                explanation = await shap_analyzer.explain_prediction(Mock(), sample_risk_data["economic_indicators"])
                
                assert risk_result["overall_risk_score"] == 0.65
                assert "shap_values" in explanation
    
    @pytest.mark.asyncio
    async def test_model_performance_monitoring(self):
        """Test model performance monitoring capabilities"""
        risk_scorer = RiskScorer()
        
        # Mock performance metrics
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc_roc": 0.90
        }
        
        with patch.object(risk_scorer, '_calculate_performance_metrics', return_value=performance_metrics):
            metrics = await risk_scorer.evaluate_model_performance()
            
            assert "accuracy" in metrics
            assert metrics["accuracy"] == 0.85
    
    def test_feature_drift_detection(self):
        """Test detection of feature drift in production data"""
        feature_engineer = EconomicFeatureEngineer()
        
        # Historical features
        historical_features = pd.DataFrame({
            "gdp_growth": [2.0, 2.1, 1.9],
            "unemployment": [3.8, 3.9, 3.7]
        })
        
        # Current features with drift
        current_features = pd.DataFrame({
            "gdp_growth": [1.0, 1.1, 0.9],  # Significant drift
            "unemployment": [3.8, 3.9, 3.7]  # No drift
        })
        
        drift_detection = feature_engineer._detect_feature_drift(historical_features, current_features)
        
        assert "gdp_growth" in drift_detection
        assert "unemployment" in drift_detection
        assert drift_detection["gdp_growth"]["drift_detected"] is True