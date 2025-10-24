"""
SHAP (SHapley Additive exPlanations) Analyzer for RiskX Platform.

This module provides comprehensive model explainability using SHAP values,
enabling transparent AI decision-making in financial risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
import json
from datetime import datetime

# SHAP imports
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ShapExplanation:
    """SHAP explanation result for a single prediction."""
    prediction_id: str
    model_prediction: float
    expected_value: float
    shap_values: np.ndarray
    feature_names: List[str]
    feature_values: np.ndarray
    explanation_type: str
    confidence_score: float
    timestamp: datetime


@dataclass
class GlobalExplanation:
    """Global model explanation using SHAP values."""
    model_id: str
    feature_importance: Dict[str, float]
    mean_abs_shap_values: Dict[str, float]
    interaction_values: Optional[np.ndarray]
    summary_plot_data: Dict[str, Any]
    partial_dependence_data: Dict[str, Any]
    sample_size: int
    timestamp: datetime


@dataclass
class BiasAnalysis:
    """Bias and fairness analysis results."""
    analysis_id: str
    protected_attributes: List[str]
    demographic_parity: Dict[str, float]
    equalized_odds: Dict[str, float]
    individual_fairness: Dict[str, float]
    group_fairness_metrics: Dict[str, Any]
    bias_score: float
    fairness_recommendations: List[str]
    timestamp: datetime


class ShapAnalyzer:
    """
    Advanced SHAP-based model explainer for financial risk models.
    
    Provides individual predictions explanations, global feature importance,
    bias detection, and comprehensive model interpretability analysis.
    """
    
    def __init__(self):
        """Initialize SHAP analyzer."""
        self.explainers = {}
        self.models = {}
        self.feature_names = {}
        self.scalers = {}
        self.explanation_history = []
        
    def register_model(
        self,
        model_id: str,
        model: Any,
        feature_names: List[str],
        model_type: str = "tree",
        background_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Register a model for SHAP analysis.
        
        Args:
            model_id: Unique identifier for the model
            model: Trained machine learning model
            feature_names: List of feature names
            model_type: Type of model (tree, linear, kernel, deep)
            background_data: Background dataset for SHAP explainer
        """
        try:
            self.models[model_id] = model
            self.feature_names[model_id] = feature_names
            
            # Create appropriate SHAP explainer based on model type
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, background_data)
            elif model_type == "kernel":
                explainer = shap.KernelExplainer(model.predict, background_data)
            elif model_type == "deep":
                explainer = shap.DeepExplainer(model, background_data)
            else:
                # Default to kernel explainer
                explainer = shap.KernelExplainer(model.predict, background_data)
            
            self.explainers[model_id] = explainer
            logger.info(f"Registered model {model_id} with {model_type} explainer")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            raise
    
    def explain_prediction(
        self,
        model_id: str,
        input_data: np.ndarray,
        prediction_id: Optional[str] = None
    ) -> ShapExplanation:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            model_id: Identifier of the registered model
            input_data: Input features for prediction
            prediction_id: Optional identifier for this prediction
            
        Returns:
            SHAP explanation with values and metadata
        """
        if model_id not in self.explainers:
            raise ValueError(f"Model {model_id} not registered")
        
        if prediction_id is None:
            prediction_id = f"pred_{int(datetime.utcnow().timestamp())}"
        
        try:
            explainer = self.explainers[model_id]
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            # Ensure input is 2D array
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Generate prediction
            prediction = model.predict(input_data)[0]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For classification models, take the first class
                shap_values = shap_values[0]
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Get expected value (baseline)
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0]
            
            # Calculate confidence score based on SHAP value magnitudes
            confidence_score = self._calculate_confidence_score(shap_values)
            
            explanation = ShapExplanation(
                prediction_id=prediction_id,
                model_prediction=float(prediction),
                expected_value=float(expected_value),
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=input_data[0],
                explanation_type="individual",
                confidence_score=confidence_score,
                timestamp=datetime.utcnow()
            )
            
            self.explanation_history.append(explanation)
            logger.info(f"Generated explanation for prediction {prediction_id}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain prediction {prediction_id}: {e}")
            raise
    
    def generate_global_explanation(
        self,
        model_id: str,
        sample_data: np.ndarray,
        sample_size: Optional[int] = None
    ) -> GlobalExplanation:
        """
        Generate global model explanation using SHAP values.
        
        Args:
            model_id: Identifier of the registered model
            sample_data: Sample dataset for global analysis
            sample_size: Maximum number of samples to use
            
        Returns:
            Global explanation with feature importance and interactions
        """
        if model_id not in self.explainers:
            raise ValueError(f"Model {model_id} not registered")
        
        try:
            explainer = self.explainers[model_id]
            feature_names = self.feature_names[model_id]
            
            # Limit sample size if specified
            if sample_size and len(sample_data) > sample_size:
                sample_indices = np.random.choice(
                    len(sample_data), sample_size, replace=False
                )
                sample_data = sample_data[sample_indices]
            
            # Calculate SHAP values for sample
            shap_values = explainer.shap_values(sample_data)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate feature importance
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = dict(zip(feature_names, mean_abs_shap))
            mean_abs_shap_values = dict(zip(feature_names, mean_abs_shap))
            
            # Calculate interaction values (if supported)
            interaction_values = None
            try:
                if hasattr(explainer, 'shap_interaction_values'):
                    interaction_values = explainer.shap_interaction_values(
                        sample_data[:min(100, len(sample_data))]  # Limit for performance
                    )
            except:
                logger.warning("Interaction values not available for this model type")
            
            # Prepare summary plot data
            summary_plot_data = self._prepare_summary_plot_data(
                shap_values, sample_data, feature_names
            )
            
            # Prepare partial dependence data
            partial_dependence_data = self._prepare_partial_dependence_data(
                model_id, sample_data, feature_names
            )
            
            global_explanation = GlobalExplanation(
                model_id=model_id,
                feature_importance=feature_importance,
                mean_abs_shap_values=mean_abs_shap_values,
                interaction_values=interaction_values,
                summary_plot_data=summary_plot_data,
                partial_dependence_data=partial_dependence_data,
                sample_size=len(sample_data),
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Generated global explanation for model {model_id}")
            return global_explanation
            
        except Exception as e:
            logger.error(f"Failed to generate global explanation: {e}")
            raise
    
    def analyze_bias(
        self,
        model_id: str,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        analysis_id: Optional[str] = None
    ) -> BiasAnalysis:
        """
        Analyze model bias and fairness using SHAP values.
        
        Args:
            model_id: Identifier of the registered model
            test_data: Test dataset features
            test_labels: True labels for test data
            protected_attributes: Dictionary of protected attribute arrays
            analysis_id: Optional identifier for this analysis
            
        Returns:
            Comprehensive bias analysis results
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        if analysis_id is None:
            analysis_id = f"bias_{int(datetime.utcnow().timestamp())}"
        
        try:
            model = self.models[model_id]
            predictions = model.predict(test_data)
            
            # Calculate demographic parity
            demographic_parity = self._calculate_demographic_parity(
                predictions, protected_attributes
            )
            
            # Calculate equalized odds
            equalized_odds = self._calculate_equalized_odds(
                predictions, test_labels, protected_attributes
            )
            
            # Calculate individual fairness using SHAP values
            individual_fairness = self._calculate_individual_fairness(
                model_id, test_data, protected_attributes
            )
            
            # Calculate group fairness metrics
            group_fairness_metrics = self._calculate_group_fairness_metrics(
                predictions, test_labels, protected_attributes
            )
            
            # Calculate overall bias score
            bias_score = self._calculate_bias_score(
                demographic_parity, equalized_odds, individual_fairness
            )
            
            # Generate fairness recommendations
            fairness_recommendations = self._generate_fairness_recommendations(
                bias_score, demographic_parity, equalized_odds
            )
            
            bias_analysis = BiasAnalysis(
                analysis_id=analysis_id,
                protected_attributes=list(protected_attributes.keys()),
                demographic_parity=demographic_parity,
                equalized_odds=equalized_odds,
                individual_fairness=individual_fairness,
                group_fairness_metrics=group_fairness_metrics,
                bias_score=bias_score,
                fairness_recommendations=fairness_recommendations,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Completed bias analysis {analysis_id}")
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze bias: {e}")
            raise
    
    def compare_models(
        self,
        model_ids: List[str],
        test_data: np.ndarray,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare multiple models using SHAP-based metrics.
        
        Args:
            model_ids: List of model identifiers to compare
            test_data: Test dataset for comparison
            sample_size: Number of samples to use for comparison
            
        Returns:
            Comprehensive model comparison results
        """
        comparison_results = {
            "models": model_ids,
            "feature_importance_comparison": {},
            "prediction_consistency": {},
            "explanation_stability": {},
            "performance_metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Limit sample size
            if len(test_data) > sample_size:
                sample_indices = np.random.choice(
                    len(test_data), sample_size, replace=False
                )
                sample_data = test_data[sample_indices]
            else:
                sample_data = test_data
            
            model_predictions = {}
            model_shap_values = {}
            
            # Generate predictions and SHAP values for each model
            for model_id in model_ids:
                if model_id not in self.models:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue
                
                model = self.models[model_id]
                explainer = self.explainers[model_id]
                
                predictions = model.predict(sample_data)
                shap_values = explainer.shap_values(sample_data)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                model_predictions[model_id] = predictions
                model_shap_values[model_id] = shap_values
                
                # Feature importance for this model
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                feature_names = self.feature_names[model_id]
                
                comparison_results["feature_importance_comparison"][model_id] = {
                    feature: float(importance) 
                    for feature, importance in zip(feature_names, mean_abs_shap)
                }
            
            # Calculate prediction consistency
            if len(model_predictions) > 1:
                prediction_correlations = {}
                model_list = list(model_predictions.keys())
                
                for i, model1 in enumerate(model_list):
                    for model2 in model_list[i+1:]:
                        correlation = np.corrcoef(
                            model_predictions[model1],
                            model_predictions[model2]
                        )[0, 1]
                        
                        prediction_correlations[f"{model1}_vs_{model2}"] = float(correlation)
                
                comparison_results["prediction_consistency"] = prediction_correlations
            
            # Calculate explanation stability
            explanation_stability = {}
            for model_id in model_shap_values:
                shap_vals = model_shap_values[model_id]
                
                # Calculate coefficient of variation for SHAP values
                mean_shap = np.mean(shap_vals, axis=0)
                std_shap = np.std(shap_vals, axis=0)
                
                # Avoid division by zero
                cv_shap = np.where(
                    mean_shap != 0,
                    std_shap / np.abs(mean_shap),
                    0
                )
                
                explanation_stability[model_id] = {
                    "mean_cv": float(np.mean(cv_shap)),
                    "max_cv": float(np.max(cv_shap)),
                    "stable_features": int(np.sum(cv_shap < 0.5))
                }
            
            comparison_results["explanation_stability"] = explanation_stability
            
            logger.info(f"Completed model comparison for {len(model_ids)} models")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def _calculate_confidence_score(self, shap_values: np.ndarray) -> float:
        """Calculate confidence score based on SHAP value distribution."""
        # Use the ratio of max absolute SHAP value to sum of absolute values
        abs_shap = np.abs(shap_values)
        if np.sum(abs_shap) == 0:
            return 0.0
        
        max_contribution = np.max(abs_shap)
        total_contribution = np.sum(abs_shap)
        
        # Normalize to [0, 1] range
        confidence = min(max_contribution / total_contribution * 2, 1.0)
        return float(confidence)
    
    def _prepare_summary_plot_data(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Prepare data for SHAP summary plots."""
        try:
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Sort features by importance
            importance_order = np.argsort(feature_importance)[::-1]
            
            plot_data = {
                "feature_names": [feature_names[i] for i in importance_order],
                "feature_importance": feature_importance[importance_order].tolist(),
                "shap_values": shap_values[:, importance_order].tolist(),
                "feature_values": feature_values[:, importance_order].tolist(),
                "sample_size": len(shap_values)
            }
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Failed to prepare summary plot data: {e}")
            return {}
    
    def _prepare_partial_dependence_data(
        self,
        model_id: str,
        sample_data: np.ndarray,
        feature_names: List[str],
        num_points: int = 50
    ) -> Dict[str, Any]:
        """Prepare partial dependence plot data."""
        try:
            model = self.models[model_id]
            pd_data = {}
            
            # Calculate partial dependence for top features
            feature_importance = np.mean(np.abs(
                self.explainers[model_id].shap_values(sample_data[:100])
            ), axis=0)
            
            # Select top 5 most important features
            top_features = np.argsort(feature_importance)[-5:]
            
            for feature_idx in top_features:
                feature_name = feature_names[feature_idx]
                feature_values = sample_data[:, feature_idx]
                
                # Create range of values for this feature
                feature_min, feature_max = np.min(feature_values), np.max(feature_values)
                test_values = np.linspace(feature_min, feature_max, num_points)
                
                # Calculate predictions across feature range
                base_sample = np.mean(sample_data, axis=0).reshape(1, -1)
                predictions = []
                
                for value in test_values:
                    modified_sample = base_sample.copy()
                    modified_sample[0, feature_idx] = value
                    pred = model.predict(modified_sample)[0]
                    predictions.append(pred)
                
                pd_data[feature_name] = {
                    "feature_values": test_values.tolist(),
                    "predictions": predictions,
                    "importance": float(feature_importance[feature_idx])
                }
            
            return pd_data
            
        except Exception as e:
            logger.error(f"Failed to prepare partial dependence data: {e}")
            return {}
    
    def _calculate_demographic_parity(
        self,
        predictions: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate demographic parity metrics."""
        demographic_parity = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            
            if len(unique_values) == 2:
                # Binary protected attribute
                group_0_mask = attr_values == unique_values[0]
                group_1_mask = attr_values == unique_values[1]
                
                group_0_rate = np.mean(predictions[group_0_mask] > 0.5)
                group_1_rate = np.mean(predictions[group_1_mask] > 0.5)
                
                # Demographic parity difference
                dp_diff = abs(group_0_rate - group_1_rate)
                demographic_parity[f"{attr_name}_parity_diff"] = float(dp_diff)
                
                # Demographic parity ratio
                if group_1_rate > 0:
                    dp_ratio = group_0_rate / group_1_rate
                    demographic_parity[f"{attr_name}_parity_ratio"] = float(dp_ratio)
        
        return demographic_parity
    
    def _calculate_equalized_odds(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate equalized odds metrics."""
        equalized_odds = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            
            if len(unique_values) == 2:
                group_0_mask = attr_values == unique_values[0]
                group_1_mask = attr_values == unique_values[1]
                
                # True Positive Rate (TPR)
                group_0_positive = true_labels[group_0_mask] == 1
                group_1_positive = true_labels[group_1_mask] == 1
                
                if np.sum(group_0_positive) > 0:
                    group_0_tpr = np.mean(
                        predictions[group_0_mask][group_0_positive] > 0.5
                    )
                else:
                    group_0_tpr = 0
                
                if np.sum(group_1_positive) > 0:
                    group_1_tpr = np.mean(
                        predictions[group_1_mask][group_1_positive] > 0.5
                    )
                else:
                    group_1_tpr = 0
                
                tpr_diff = abs(group_0_tpr - group_1_tpr)
                equalized_odds[f"{attr_name}_tpr_diff"] = float(tpr_diff)
                
                # False Positive Rate (FPR)
                group_0_negative = true_labels[group_0_mask] == 0
                group_1_negative = true_labels[group_1_mask] == 0
                
                if np.sum(group_0_negative) > 0:
                    group_0_fpr = np.mean(
                        predictions[group_0_mask][group_0_negative] > 0.5
                    )
                else:
                    group_0_fpr = 0
                
                if np.sum(group_1_negative) > 0:
                    group_1_fpr = np.mean(
                        predictions[group_1_mask][group_1_negative] > 0.5
                    )
                else:
                    group_1_fpr = 0
                
                fpr_diff = abs(group_0_fpr - group_1_fpr)
                equalized_odds[f"{attr_name}_fpr_diff"] = float(fpr_diff)
        
        return equalized_odds
    
    def _calculate_individual_fairness(
        self,
        model_id: str,
        test_data: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate individual fairness using SHAP values."""
        individual_fairness = {}
        
        try:
            explainer = self.explainers[model_id]
            shap_values = explainer.shap_values(test_data[:100])  # Limit for performance
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            for attr_name, attr_values in protected_attributes.items():
                attr_sample = attr_values[:100]
                
                # Find feature index corresponding to protected attribute
                feature_names = self.feature_names[model_id]
                if attr_name in feature_names:
                    attr_idx = feature_names.index(attr_name)
                    
                    # Calculate average absolute SHAP value for protected attribute
                    avg_shap_magnitude = np.mean(np.abs(shap_values[:, attr_idx]))
                    individual_fairness[f"{attr_name}_shap_impact"] = float(avg_shap_magnitude)
                    
                    # Calculate relative importance
                    total_shap_magnitude = np.mean(np.sum(np.abs(shap_values), axis=1))
                    if total_shap_magnitude > 0:
                        relative_impact = avg_shap_magnitude / total_shap_magnitude
                        individual_fairness[f"{attr_name}_relative_impact"] = float(relative_impact)
        
        except Exception as e:
            logger.warning(f"Failed to calculate individual fairness: {e}")
        
        return individual_fairness
    
    def _calculate_group_fairness_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate comprehensive group fairness metrics."""
        group_metrics = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            group_metrics[attr_name] = {}
            
            for value in unique_values:
                group_mask = attr_values == value
                group_predictions = predictions[group_mask]
                group_labels = true_labels[group_mask]
                
                # Basic metrics
                accuracy = np.mean((group_predictions > 0.5) == group_labels)
                precision = self._calculate_precision(group_predictions, group_labels)
                recall = self._calculate_recall(group_predictions, group_labels)
                
                group_metrics[attr_name][str(value)] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "sample_size": int(np.sum(group_mask))
                }
        
        return group_metrics
    
    def _calculate_precision(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate precision metric."""
        predicted_positive = predictions > 0.5
        true_positive = np.sum(predicted_positive & (true_labels == 1))
        false_positive = np.sum(predicted_positive & (true_labels == 0))
        
        if true_positive + false_positive == 0:
            return 0.0
        
        return true_positive / (true_positive + false_positive)
    
    def _calculate_recall(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate recall metric."""
        predicted_positive = predictions > 0.5
        true_positive = np.sum(predicted_positive & (true_labels == 1))
        false_negative = np.sum((predictions <= 0.5) & (true_labels == 1))
        
        if true_positive + false_negative == 0:
            return 0.0
        
        return true_positive / (true_positive + false_negative)
    
    def _calculate_bias_score(
        self,
        demographic_parity: Dict[str, float],
        equalized_odds: Dict[str, float],
        individual_fairness: Dict[str, float]
    ) -> float:
        """Calculate overall bias score."""
        all_metrics = []
        
        # Collect all bias metrics
        all_metrics.extend(demographic_parity.values())
        all_metrics.extend(equalized_odds.values())
        all_metrics.extend(individual_fairness.values())
        
        if not all_metrics:
            return 0.0
        
        # Calculate weighted average bias score
        bias_score = np.mean(all_metrics)
        return float(min(bias_score, 1.0))  # Cap at 1.0
    
    def _generate_fairness_recommendations(
        self,
        bias_score: float,
        demographic_parity: Dict[str, float],
        equalized_odds: Dict[str, float]
    ) -> List[str]:
        """Generate actionable fairness recommendations."""
        recommendations = []
        
        if bias_score > 0.1:
            recommendations.append("Consider bias mitigation techniques")
        
        if any(val > 0.05 for val in demographic_parity.values()):
            recommendations.append("Address demographic parity disparities")
        
        if any(val > 0.05 for val in equalized_odds.values()):
            recommendations.append("Improve equalized odds across groups")
        
        if bias_score > 0.2:
            recommendations.append("Consider collecting more balanced training data")
            recommendations.append("Implement fairness constraints during model training")
        
        if not recommendations:
            recommendations.append("Model shows acceptable fairness metrics")
        
        return recommendations
    
    def export_explanation(
        self,
        explanation: ShapExplanation,
        format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export SHAP explanation in specified format."""
        if format_type == "json":
            return {
                "prediction_id": explanation.prediction_id,
                "model_prediction": explanation.model_prediction,
                "expected_value": explanation.expected_value,
                "shap_values": explanation.shap_values.tolist(),
                "feature_names": explanation.feature_names,
                "feature_values": explanation.feature_values.tolist(),
                "explanation_type": explanation.explanation_type,
                "confidence_score": explanation.confidence_score,
                "timestamp": explanation.timestamp.isoformat()
            }
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_explanation_history(
        self,
        limit: int = 10,
        model_id: Optional[str] = None
    ) -> List[ShapExplanation]:
        """Get recent explanation history."""
        explanations = self.explanation_history
        
        if model_id:
            # Filter by model if specified (would need to track model_id in explanations)
            pass
        
        return explanations[-limit:] if limit else explanations