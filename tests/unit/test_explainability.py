"""
Unit tests for the SHAP Explainability system.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.ml.explainability.shap_analyzer import (
    ShapAnalyzer,
    ShapExplanation,
    GlobalExplanation,
    BiasAnalysis
)


@pytest.mark.unit
class TestShapAnalyzer:
    """Test cases for ShapAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a ShapAnalyzer instance."""
        return ShapAnalyzer()
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        X_train = np.random.random((100, 5))
        y_train = np.random.random(100)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return np.random.random((50, 5))
    
    def test_analyzer_initialization(self, analyzer):
        """Test ShapAnalyzer initialization."""
        assert isinstance(analyzer, ShapAnalyzer)
        assert analyzer.explainers == {}
        assert analyzer.models == {}
        assert analyzer.feature_names == {}
        assert analyzer.explanation_history == []
    
    def test_register_tree_model(self, analyzer, sample_model, feature_names):
        """Test registering a tree-based model."""
        model_id = "test_tree_model"
        
        analyzer.register_model(
            model_id=model_id,
            model=sample_model,
            feature_names=feature_names,
            model_type="tree"
        )
        
        assert model_id in analyzer.models
        assert model_id in analyzer.explainers
        assert model_id in analyzer.feature_names
        assert analyzer.feature_names[model_id] == feature_names
    
    def test_register_linear_model(self, analyzer, feature_names):
        """Test registering a linear model."""
        # Create linear model
        X_train = np.random.random((100, 5))
        y_train = np.random.random(100)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        model_id = "test_linear_model"
        background_data = np.random.random((10, 5))
        
        analyzer.register_model(
            model_id=model_id,
            model=model,
            feature_names=feature_names,
            model_type="linear",
            background_data=background_data
        )
        
        assert model_id in analyzer.models
        assert model_id in analyzer.explainers
        assert analyzer.feature_names[model_id] == feature_names
    
    def test_explain_prediction_single_input(self, analyzer, sample_model, feature_names):
        """Test explanation generation for single prediction."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        # Test with 1D input
        input_data = np.random.random(5)
        explanation = analyzer.explain_prediction(model_id, input_data)
        
        assert isinstance(explanation, ShapExplanation)
        assert explanation.prediction_id.startswith("pred_")
        assert isinstance(explanation.model_prediction, float)
        assert isinstance(explanation.expected_value, float)
        assert len(explanation.shap_values) == 5
        assert len(explanation.feature_names) == 5
        assert len(explanation.feature_values) == 5
        assert 0 <= explanation.confidence_score <= 1
        assert explanation.explanation_type == "individual"
    
    def test_explain_prediction_2d_input(self, analyzer, sample_model, feature_names):
        """Test explanation generation for 2D input."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        # Test with 2D input
        input_data = np.random.random((1, 5))
        explanation = analyzer.explain_prediction(model_id, input_data)
        
        assert isinstance(explanation, ShapExplanation)
        assert len(explanation.shap_values) == 5
    
    def test_explain_prediction_with_custom_id(self, analyzer, sample_model, feature_names):
        """Test explanation with custom prediction ID."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        input_data = np.random.random(5)
        custom_id = "custom_prediction_123"
        
        explanation = analyzer.explain_prediction(model_id, input_data, custom_id)
        
        assert explanation.prediction_id == custom_id
    
    def test_explain_prediction_unregistered_model(self, analyzer):
        """Test error handling for unregistered model."""
        input_data = np.random.random(5)
        
        with pytest.raises(ValueError, match="Model unregistered_model not registered"):
            analyzer.explain_prediction("unregistered_model", input_data)
    
    def test_generate_global_explanation(self, analyzer, sample_model, feature_names, sample_data):
        """Test global explanation generation."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        global_explanation = analyzer.generate_global_explanation(
            model_id, sample_data, sample_size=30
        )
        
        assert isinstance(global_explanation, GlobalExplanation)
        assert global_explanation.model_id == model_id
        assert len(global_explanation.feature_importance) == 5
        assert len(global_explanation.mean_abs_shap_values) == 5
        assert global_explanation.sample_size == 30
        assert isinstance(global_explanation.summary_plot_data, dict)
        assert isinstance(global_explanation.partial_dependence_data, dict)
    
    def test_generate_global_explanation_large_sample(self, analyzer, sample_model, feature_names):
        """Test global explanation with sample size limiting."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        # Large sample data
        large_sample = np.random.random((200, 5))
        
        global_explanation = analyzer.generate_global_explanation(
            model_id, large_sample, sample_size=50
        )
        
        assert global_explanation.sample_size == 50
    
    def test_analyze_bias(self, analyzer, sample_model, feature_names):
        """Test bias analysis functionality."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        # Create test data with protected attributes
        test_data = np.random.random((100, 5))
        test_labels = np.random.randint(0, 2, 100)
        protected_attributes = {
            "gender": np.random.randint(0, 2, 100),
            "age_group": np.random.randint(0, 2, 100)
        }
        
        bias_analysis = analyzer.analyze_bias(
            model_id, test_data, test_labels, protected_attributes
        )
        
        assert isinstance(bias_analysis, BiasAnalysis)
        assert bias_analysis.analysis_id.startswith("bias_")
        assert "gender" in bias_analysis.protected_attributes
        assert "age_group" in bias_analysis.protected_attributes
        assert isinstance(bias_analysis.demographic_parity, dict)
        assert isinstance(bias_analysis.equalized_odds, dict)
        assert isinstance(bias_analysis.individual_fairness, dict)
        assert 0 <= bias_analysis.bias_score <= 1
        assert isinstance(bias_analysis.fairness_recommendations, list)
    
    def test_analyze_bias_custom_id(self, analyzer, sample_model, feature_names):
        """Test bias analysis with custom analysis ID."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        test_data = np.random.random((50, 5))
        test_labels = np.random.randint(0, 2, 50)
        protected_attributes = {"group": np.random.randint(0, 2, 50)}
        
        custom_id = "custom_bias_analysis"
        bias_analysis = analyzer.analyze_bias(
            model_id, test_data, test_labels, protected_attributes, custom_id
        )
        
        assert bias_analysis.analysis_id == custom_id
    
    def test_compare_models(self, analyzer, feature_names):
        """Test model comparison functionality."""
        # Register multiple models
        models = {}
        for i in range(3):
            X_train = np.random.random((100, 5))
            y_train = np.random.random(100)
            model = RandomForestRegressor(n_estimators=10, random_state=42+i)
            model.fit(X_train, y_train)
            
            model_id = f"test_model_{i}"
            analyzer.register_model(model_id, model, feature_names, "tree")
            models[model_id] = model
        
        # Test comparison
        test_data = np.random.random((100, 5))
        model_ids = list(models.keys())
        
        comparison_results = analyzer.compare_models(model_ids, test_data, sample_size=50)
        
        assert "models" in comparison_results
        assert "feature_importance_comparison" in comparison_results
        assert "prediction_consistency" in comparison_results
        assert "explanation_stability" in comparison_results
        assert comparison_results["models"] == model_ids
        
        # Check feature importance comparison
        for model_id in model_ids:
            assert model_id in comparison_results["feature_importance_comparison"]
            importance_dict = comparison_results["feature_importance_comparison"][model_id]
            assert len(importance_dict) == 5
    
    def test_compare_models_missing_model(self, analyzer, feature_names, sample_model):
        """Test model comparison with missing model."""
        # Register one model
        analyzer.register_model("existing_model", sample_model, feature_names, "tree")
        
        # Try to compare with non-existent model
        test_data = np.random.random((50, 5))
        model_ids = ["existing_model", "missing_model"]
        
        # Should not raise error, just skip missing model
        comparison_results = analyzer.compare_models(model_ids, test_data)
        
        assert "existing_model" in comparison_results["feature_importance_comparison"]
        assert "missing_model" not in comparison_results["feature_importance_comparison"]
    
    def test_calculate_confidence_score(self, analyzer):
        """Test confidence score calculation."""
        # High confidence case (one dominant feature)
        high_conf_shap = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        high_confidence = analyzer._calculate_confidence_score(high_conf_shap)
        
        # Low confidence case (uniform contributions)
        low_conf_shap = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        low_confidence = analyzer._calculate_confidence_score(low_conf_shap)
        
        assert 0 <= high_confidence <= 1
        assert 0 <= low_confidence <= 1
        assert high_confidence > low_confidence
    
    def test_calculate_confidence_score_zero_contributions(self, analyzer):
        """Test confidence score with zero contributions."""
        zero_shap = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        confidence = analyzer._calculate_confidence_score(zero_shap)
        
        assert confidence == 0.0
    
    def test_calculate_demographic_parity(self, analyzer):
        """Test demographic parity calculation."""
        predictions = np.array([0.8, 0.3, 0.9, 0.2, 0.7, 0.4])
        protected_attributes = {
            "group": np.array([0, 0, 0, 1, 1, 1])
        }
        
        dp_metrics = analyzer._calculate_demographic_parity(predictions, protected_attributes)
        
        assert "group_parity_diff" in dp_metrics
        assert "group_parity_ratio" in dp_metrics
        assert 0 <= dp_metrics["group_parity_diff"] <= 1
    
    def test_calculate_equalized_odds(self, analyzer):
        """Test equalized odds calculation."""
        predictions = np.array([0.8, 0.3, 0.9, 0.2, 0.7, 0.4])
        true_labels = np.array([1, 0, 1, 0, 1, 0])
        protected_attributes = {
            "group": np.array([0, 0, 0, 1, 1, 1])
        }
        
        eo_metrics = analyzer._calculate_equalized_odds(
            predictions, true_labels, protected_attributes
        )
        
        assert "group_tpr_diff" in eo_metrics
        assert "group_fpr_diff" in eo_metrics
        assert 0 <= eo_metrics["group_tpr_diff"] <= 1
        assert 0 <= eo_metrics["group_fpr_diff"] <= 1
    
    def test_calculate_individual_fairness(self, analyzer, sample_model, feature_names):
        """Test individual fairness calculation."""
        model_id = "test_model"
        feature_names_with_protected = feature_names + ["gender"]
        analyzer.register_model(model_id, sample_model, feature_names_with_protected, "tree")
        
        test_data = np.random.random((50, 6))  # 5 features + 1 protected
        protected_attributes = {
            "gender": np.random.randint(0, 2, 50)
        }
        
        # This will test the method even though the model might not have the exact feature
        fairness_metrics = analyzer._calculate_individual_fairness(
            model_id, test_data, protected_attributes
        )
        
        assert isinstance(fairness_metrics, dict)
    
    def test_export_explanation_json(self, analyzer, sample_model, feature_names):
        """Test exporting explanation to JSON format."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        input_data = np.random.random(5)
        explanation = analyzer.explain_prediction(model_id, input_data)
        
        exported = analyzer.export_explanation(explanation, "json")
        
        assert isinstance(exported, dict)
        assert "prediction_id" in exported
        assert "model_prediction" in exported
        assert "shap_values" in exported
        assert "feature_names" in exported
        assert "confidence_score" in exported
        assert isinstance(exported["shap_values"], list)
    
    def test_export_explanation_unsupported_format(self, analyzer, sample_model, feature_names):
        """Test error handling for unsupported export format."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        input_data = np.random.random(5)
        explanation = analyzer.explain_prediction(model_id, input_data)
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            analyzer.export_explanation(explanation, "unsupported_format")
    
    def test_get_explanation_history(self, analyzer, sample_model, feature_names):
        """Test getting explanation history."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        # Generate multiple explanations
        for i in range(5):
            input_data = np.random.random(5)
            analyzer.explain_prediction(model_id, input_data, f"pred_{i}")
        
        # Test getting limited history
        history = analyzer.get_explanation_history(limit=3)
        assert len(history) == 3
        
        # Test getting all history
        all_history = analyzer.get_explanation_history()
        assert len(all_history) == 5
    
    def test_prepare_summary_plot_data(self, analyzer, feature_names):
        """Test summary plot data preparation."""
        shap_values = np.random.random((10, 5))
        feature_values = np.random.random((10, 5))
        
        plot_data = analyzer._prepare_summary_plot_data(
            shap_values, feature_values, feature_names
        )
        
        assert "feature_names" in plot_data
        assert "feature_importance" in plot_data
        assert "shap_values" in plot_data
        assert "feature_values" in plot_data
        assert "sample_size" in plot_data
        assert plot_data["sample_size"] == 10
        assert len(plot_data["feature_names"]) == 5
    
    def test_prepare_partial_dependence_data(self, analyzer, sample_model, feature_names):
        """Test partial dependence data preparation."""
        model_id = "test_model"
        analyzer.register_model(model_id, sample_model, feature_names, "tree")
        
        sample_data = np.random.random((20, 5))
        
        pd_data = analyzer._prepare_partial_dependence_data(
            model_id, sample_data, feature_names
        )
        
        assert isinstance(pd_data, dict)
        # Should have data for top features
        assert len(pd_data) <= 5
        
        # Check structure of each feature's data
        for feature_name, data in pd_data.items():
            assert "feature_values" in data
            assert "predictions" in data
            assert "importance" in data
            assert isinstance(data["feature_values"], list)
            assert isinstance(data["predictions"], list)
    
    def test_calculate_precision_recall(self, analyzer):
        """Test precision and recall calculations."""
        # Perfect predictions
        perfect_pred = np.array([0.9, 0.1, 0.9, 0.1])
        perfect_labels = np.array([1, 0, 1, 0])
        
        precision = analyzer._calculate_precision(perfect_pred, perfect_labels)
        recall = analyzer._calculate_recall(perfect_pred, perfect_labels)
        
        assert precision == 1.0
        assert recall == 1.0
        
        # No positive predictions
        negative_pred = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([1, 0, 1, 0])
        
        precision_zero = analyzer._calculate_precision(negative_pred, labels)
        assert precision_zero == 0.0
    
    def test_calculate_bias_score(self, analyzer):
        """Test overall bias score calculation."""
        demographic_parity = {"group_diff": 0.1}
        equalized_odds = {"group_tpr_diff": 0.05, "group_fpr_diff": 0.08}
        individual_fairness = {"group_impact": 0.15}
        
        bias_score = analyzer._calculate_bias_score(
            demographic_parity, equalized_odds, individual_fairness
        )
        
        assert 0 <= bias_score <= 1
        assert isinstance(bias_score, float)
    
    def test_generate_fairness_recommendations(self, analyzer):
        """Test fairness recommendations generation."""
        # High bias case
        high_bias_score = 0.3
        high_dp = {"group_diff": 0.2}
        high_eo = {"group_tpr_diff": 0.15}
        
        recommendations = analyzer._generate_fairness_recommendations(
            high_bias_score, high_dp, high_eo
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("bias mitigation" in rec.lower() for rec in recommendations)
        
        # Low bias case
        low_bias_score = 0.02
        low_dp = {"group_diff": 0.01}
        low_eo = {"group_tpr_diff": 0.01}
        
        good_recommendations = analyzer._generate_fairness_recommendations(
            low_bias_score, low_dp, low_eo
        )
        
        assert "Model shows acceptable fairness metrics" in good_recommendations