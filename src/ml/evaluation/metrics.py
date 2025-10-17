"""
Risk Metrics Module

Provides specialized metrics for evaluating risk prediction models
including financial risk-specific measures and standard ML metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)

from ...cache.cache_manager import CacheManager
from ...core.config import get_settings


@dataclass
class RiskMetricResults:
    """Container for risk prediction evaluation results"""
    classification_metrics: Dict[str, float]
    regression_metrics: Dict[str, float]
    risk_specific_metrics: Dict[str, float]
    temporal_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    overall_score: float
    metric_breakdown: Dict[str, Dict[str, float]]
    recommendations: List[str]
    risk_assessment: str
    timestamp: datetime


class RiskMetrics:
    """
    Calculates comprehensive metrics for risk prediction models.
    
    Provides:
    - Standard classification/regression metrics
    - Risk-specific financial metrics
    - Temporal stability analysis
    - Early warning system evaluation
    - Model reliability assessment
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Metric calculation configuration
        self.config = {
            "risk_threshold": 0.7,  # Threshold for high risk classification
            "early_warning_horizon": 30,  # Days for early warning evaluation
            "stability_window": 90,  # Days for stability analysis
            "confidence_levels": [0.8, 0.9, 0.95],  # For interval predictions
        }
        
        # Risk-specific metric weights
        self.metric_weights = {
            "accuracy": 0.15,
            "precision": 0.20,
            "recall": 0.20,
            "early_warning": 0.25,
            "stability": 0.15,
            "calibration": 0.05
        }
    
    async def evaluate_model_performance(
        self,
        y_true: Union[np.ndarray, List, pd.Series],
        y_pred: Union[np.ndarray, List, pd.Series],
        y_pred_proba: Optional[Union[np.ndarray, List, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, List, pd.Series]] = None,
        model_name: str = "risk_model"
    ) -> RiskMetricResults:
        """
        Comprehensive evaluation of risk prediction model performance
        """
        self.logger.info(f"Evaluating performance for {model_name}")
        
        try:
            # Convert inputs to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
            
            # Calculate different metric categories
            classification_metrics = self._calculate_classification_metrics(
                y_true, y_pred, y_pred_proba
            )
            
            regression_metrics = self._calculate_regression_metrics(y_true, y_pred)
            
            risk_specific_metrics = self._calculate_risk_specific_metrics(
                y_true, y_pred, y_pred_proba
            )
            
            temporal_metrics = self._calculate_temporal_metrics(
                y_true, y_pred, timestamps
            ) if timestamps is not None else {}
            
            stability_metrics = self._calculate_stability_metrics(
                y_true, y_pred, timestamps
            ) if timestamps is not None else {}
            
            # Create metadata
            metadata = {
                "model_name": model_name,
                "evaluation_date": datetime.now().isoformat(),
                "sample_size": len(y_true),
                "positive_class_ratio": np.mean(y_true),
                "prediction_range": {
                    "min": float(np.min(y_pred)),
                    "max": float(np.max(y_pred)),
                    "mean": float(np.mean(y_pred))
                }
            }
            
            results = RiskMetricResults(
                classification_metrics=classification_metrics,
                regression_metrics=regression_metrics,
                risk_specific_metrics=risk_specific_metrics,
                temporal_metrics=temporal_metrics,
                stability_metrics=stability_metrics,
                metadata=metadata
            )
            
            # Cache results
            await self._cache_evaluation_results(results, model_name)
            
            self.logger.info(f"Performance evaluation completed for {model_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            raise
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate standard classification metrics"""
        
        metrics = {}
        
        try:
            # Convert to binary classification if needed
            y_true_binary = (y_true > self.config["risk_threshold"]).astype(int)
            y_pred_binary = (y_pred > self.config["risk_threshold"]).astype(int)
            
            # Basic classification metrics
            metrics["accuracy"] = float(accuracy_score(y_true_binary, y_pred_binary))
            metrics["precision"] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics["recall"] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            
            # AUC-ROC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics["auc_roc"] = float(roc_auc_score(y_true_binary, y_pred_proba))
                except:
                    metrics["auc_roc"] = 0.0
            
            # Confusion matrix components
            true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            metrics["true_positive_rate"] = float(true_positives / max(true_positives + false_negatives, 1))
            metrics["false_positive_rate"] = float(false_positives / max(false_positives + true_negatives, 1))
            metrics["specificity"] = float(true_negatives / max(true_negatives + false_positives, 1))
            
        except Exception as e:
            self.logger.warning(f"Error calculating classification metrics: {str(e)}")
            metrics = {"error": "Could not calculate classification metrics"}
        
        return metrics
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        metrics = {}
        
        try:
            # Standard regression metrics
            metrics["mean_squared_error"] = float(mean_squared_error(y_true, y_pred))
            metrics["root_mean_squared_error"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mean_absolute_error"] = float(mean_absolute_error(y_true, y_pred))
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r_squared"] = float(1 - (ss_res / max(ss_tot, 1e-10)))
            
            # Mean Absolute Percentage Error
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics["mean_absolute_percentage_error"] = float(mape)
            
            # Directional accuracy (for time series)
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(true_direction == pred_direction)
                metrics["directional_accuracy"] = float(directional_accuracy)
            
        except Exception as e:
            self.logger.warning(f"Error calculating regression metrics: {str(e)}")
            metrics = {"error": "Could not calculate regression metrics"}
        
        return metrics
    
    def _calculate_risk_specific_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate risk-specific financial metrics"""
        
        metrics = {}
        
        try:
            # Risk threshold analysis
            high_risk_threshold = self.config["risk_threshold"]
            
            # True high risk cases
            true_high_risk = y_true > high_risk_threshold
            pred_high_risk = y_pred > high_risk_threshold
            
            # Risk detection rate
            if np.any(true_high_risk):
                risk_detection_rate = np.mean(pred_high_risk[true_high_risk])
                metrics["risk_detection_rate"] = float(risk_detection_rate)
            
            # False alarm rate
            if np.any(~true_high_risk):
                false_alarm_rate = np.mean(pred_high_risk[~true_high_risk])
                metrics["false_alarm_rate"] = float(false_alarm_rate)
            
            # Early warning effectiveness
            metrics.update(self._calculate_early_warning_metrics(y_true, y_pred))
            
            # Model calibration
            if y_pred_proba is not None:
                calibration_score = self._calculate_calibration_score(y_true, y_pred_proba)
                metrics["calibration_score"] = calibration_score
            
            # Risk level distribution alignment
            metrics.update(self._calculate_distribution_alignment(y_true, y_pred))
            
            # Extreme event detection
            metrics.update(self._calculate_extreme_event_metrics(y_true, y_pred))
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk-specific metrics: {str(e)}")
            metrics = {"error": "Could not calculate risk-specific metrics"}
        
        return metrics
    
    def _calculate_early_warning_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate early warning system effectiveness"""
        
        metrics = {}
        
        try:
            # For time series, evaluate how well model predicts future high risk
            if len(y_true) < 10:
                return {"early_warning_note": "Insufficient data for early warning analysis"}
            
            # Simple approach: check if high predictions precede high actual values
            high_risk_threshold = self.config["risk_threshold"]
            
            # Find high risk periods
            high_risk_periods = np.where(y_true > high_risk_threshold)[0]
            
            if len(high_risk_periods) > 0:
                # Check if model predicted high risk before these periods
                warning_success = 0
                total_high_risk_events = len(high_risk_periods)
                
                for period in high_risk_periods:
                    # Look back up to 5 periods for warning signals
                    lookback_start = max(0, period - 5)
                    if lookback_start < period:
                        warning_signals = y_pred[lookback_start:period] > (high_risk_threshold * 0.8)
                        if np.any(warning_signals):
                            warning_success += 1
                
                early_warning_rate = warning_success / total_high_risk_events
                metrics["early_warning_rate"] = float(early_warning_rate)
                metrics["high_risk_events_detected"] = warning_success
                metrics["total_high_risk_events"] = total_high_risk_events
            
        except Exception as e:
            self.logger.warning(f"Error calculating early warning metrics: {str(e)}")
        
        return metrics
    
    def _calculate_calibration_score(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Calculate model calibration score"""
        
        try:
            # Bin predictions and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Average predicted probability in bin
                    avg_predicted_prob = y_pred_proba[in_bin].mean()
                    
                    # Actual frequency of positive class in bin
                    actual_prob = (y_true[in_bin] > self.config["risk_threshold"]).mean()
                    
                    # Add to calibration error
                    calibration_error += np.abs(avg_predicted_prob - actual_prob) * prop_in_bin
            
            # Return calibration score (1 - error)
            return float(1 - calibration_error)
            
        except Exception as e:
            self.logger.warning(f"Error calculating calibration score: {str(e)}")
            return 0.0
    
    def _calculate_distribution_alignment(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate how well predicted risk distribution aligns with actual"""
        
        metrics = {}
        
        try:
            # Compare statistical moments
            true_mean = np.mean(y_true)
            pred_mean = np.mean(y_pred)
            mean_alignment = 1 - abs(true_mean - pred_mean) / max(true_mean, 0.01)
            metrics["mean_alignment"] = float(max(0, mean_alignment))
            
            true_std = np.std(y_true)
            pred_std = np.std(y_pred)
            std_alignment = 1 - abs(true_std - pred_std) / max(true_std, 0.01)
            metrics["std_alignment"] = float(max(0, std_alignment))
            
            # Kolmogorov-Smirnov test statistic (simplified)
            # Sort both arrays and compare cumulative distributions
            n = len(y_true)
            true_sorted = np.sort(y_true)
            pred_sorted = np.sort(y_pred)
            
            # Create empirical CDFs
            true_cdf = np.arange(1, n + 1) / n
            pred_cdf = np.arange(1, n + 1) / n
            
            # Find maximum difference
            ks_statistic = np.max(np.abs(true_cdf - pred_cdf))
            ks_alignment = 1 - ks_statistic
            metrics["distribution_similarity"] = float(ks_alignment)
            
        except Exception as e:
            self.logger.warning(f"Error calculating distribution alignment: {str(e)}")
        
        return metrics
    
    def _calculate_extreme_event_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for extreme risk event detection"""
        
        metrics = {}
        
        try:
            # Define extreme events as top 5% of risk values
            extreme_threshold = np.percentile(y_true, 95)
            
            true_extreme = y_true > extreme_threshold
            pred_extreme = y_pred > extreme_threshold
            
            if np.any(true_extreme):
                # Extreme event detection rate
                extreme_detection_rate = np.mean(pred_extreme[true_extreme])
                metrics["extreme_event_detection_rate"] = float(extreme_detection_rate)
                
                # Extreme event false positive rate
                if np.any(~true_extreme):
                    extreme_fp_rate = np.mean(pred_extreme[~true_extreme])
                    metrics["extreme_event_false_positive_rate"] = float(extreme_fp_rate)
                
                # Extreme event count
                metrics["extreme_events_count"] = int(np.sum(true_extreme))
                metrics["predicted_extreme_events_count"] = int(np.sum(pred_extreme))
            
        except Exception as e:
            self.logger.warning(f"Error calculating extreme event metrics: {str(e)}")
        
        return metrics
    
    def _calculate_temporal_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, float]:
        """Calculate temporal analysis metrics"""
        
        metrics = {}
        
        try:
            # Convert timestamps to datetime if needed
            if not isinstance(timestamps[0], datetime):
                timestamps = pd.to_datetime(timestamps)
            
            # Create time series dataframe
            df = pd.DataFrame({
                'timestamp': timestamps,
                'y_true': y_true,
                'y_pred': y_pred,
                'error': y_true - y_pred
            }).sort_values('timestamp')
            
            # Calculate autocorrelation of errors
            error_autocorr = df['error'].autocorr(lag=1)
            if not np.isnan(error_autocorr):
                metrics["error_autocorrelation"] = float(error_autocorr)
            
            # Performance over time
            df['period'] = pd.cut(df.index, bins=5, labels=['P1', 'P2', 'P3', 'P4', 'P5'])
            
            period_performance = {}
            for period in df['period'].unique():
                if pd.notna(period):
                    period_data = df[df['period'] == period]
                    period_mse = mean_squared_error(period_data['y_true'], period_data['y_pred'])
                    period_performance[f"mse_{period}"] = float(period_mse)
            
            metrics.update(period_performance)
            
            # Calculate performance stability over time
            if len(period_performance) > 1:
                mse_values = [v for k, v in period_performance.items() if 'mse_' in k]
                performance_stability = 1 / (1 + np.std(mse_values))
                metrics["temporal_stability"] = float(performance_stability)
            
        except Exception as e:
            self.logger.warning(f"Error calculating temporal metrics: {str(e)}")
        
        return metrics
    
    def _calculate_stability_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model stability metrics"""
        
        metrics = {}
        
        try:
            # Convert timestamps to datetime if needed
            if not isinstance(timestamps[0], datetime):
                timestamps = pd.to_datetime(timestamps)
            
            # Create time series dataframe
            df = pd.DataFrame({
                'timestamp': timestamps,
                'y_true': y_true,
                'y_pred': y_pred
            }).sort_values('timestamp')
            
            # Rolling window analysis
            window_size = min(30, len(df) // 4)  # Use 30 days or 1/4 of data
            
            if window_size >= 10:
                rolling_accuracy = []
                
                for i in range(window_size, len(df)):
                    window_data = df.iloc[i-window_size:i]
                    window_true = (window_data['y_true'] > self.config["risk_threshold"]).astype(int)
                    window_pred = (window_data['y_pred'] > self.config["risk_threshold"]).astype(int)
                    
                    window_accuracy = accuracy_score(window_true, window_pred)
                    rolling_accuracy.append(window_accuracy)
                
                if rolling_accuracy:
                    # Stability as inverse of accuracy variance
                    accuracy_stability = 1 / (1 + np.var(rolling_accuracy))
                    metrics["accuracy_stability"] = float(accuracy_stability)
                    
                    # Trend in performance
                    if len(rolling_accuracy) > 5:
                        x = np.arange(len(rolling_accuracy))
                        trend_slope = np.polyfit(x, rolling_accuracy, 1)[0]
                        metrics["performance_trend"] = float(trend_slope)
            
        except Exception as e:
            self.logger.warning(f"Error calculating stability metrics: {str(e)}")
        
        return metrics
    
    async def _cache_evaluation_results(self, results: RiskMetricResults, model_name: str):
        """Cache evaluation results"""
        try:
            results_data = {
                "classification_metrics": results.classification_metrics,
                "regression_metrics": results.regression_metrics,
                "risk_specific_metrics": results.risk_specific_metrics,
                "temporal_metrics": results.temporal_metrics,
                "stability_metrics": results.stability_metrics,
                "metadata": results.metadata
            }
            
            # Cache with model-specific key
            cache_key = f"evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d')}"
            await self.cache.set(cache_key, results_data, ttl=86400 * 7)
            
            # Cache as latest results
            latest_key = f"evaluation_results_{model_name}_latest"
            await self.cache.set(latest_key, results_data, ttl=86400)
            
        except Exception as e:
            self.logger.warning(f"Error caching evaluation results: {str(e)}")
    
    async def generate_performance_report(
        self, 
        results: RiskMetricResults
    ) -> PerformanceReport:
        """Generate comprehensive performance report with recommendations"""
        
        try:
            # Calculate overall score
            overall_score = self._calculate_overall_score(results)
            
            # Organize metrics by category
            metric_breakdown = {
                "Classification": results.classification_metrics,
                "Regression": results.regression_metrics,
                "Risk-Specific": results.risk_specific_metrics,
                "Temporal": results.temporal_metrics,
                "Stability": results.stability_metrics
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(results)
            
            # Assess risk level
            risk_assessment = self._assess_model_risk_level(results)
            
            report = PerformanceReport(
                overall_score=overall_score,
                metric_breakdown=metric_breakdown,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            raise
    
    def _calculate_overall_score(self, results: RiskMetricResults) -> float:
        """Calculate weighted overall performance score"""
        
        score_components = {}
        
        # Extract key metrics for scoring
        if "accuracy" in results.classification_metrics:
            score_components["accuracy"] = results.classification_metrics["accuracy"]
        
        if "precision" in results.classification_metrics:
            score_components["precision"] = results.classification_metrics["precision"]
        
        if "recall" in results.classification_metrics:
            score_components["recall"] = results.classification_metrics["recall"]
        
        if "early_warning_rate" in results.risk_specific_metrics:
            score_components["early_warning"] = results.risk_specific_metrics["early_warning_rate"]
        
        if "temporal_stability" in results.temporal_metrics:
            score_components["stability"] = results.temporal_metrics["temporal_stability"]
        
        if "calibration_score" in results.risk_specific_metrics:
            score_components["calibration"] = results.risk_specific_metrics["calibration_score"]
        
        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0
        
        for metric, weight in self.metric_weights.items():
            if metric in score_components:
                weighted_sum += score_components[metric] * weight
                total_weight += weight
        
        # Normalize score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        return min(1.0, max(0.0, overall_score))
    
    def _generate_recommendations(self, results: RiskMetricResults) -> List[str]:
        """Generate improvement recommendations based on results"""
        
        recommendations = []
        
        # Check classification performance
        if results.classification_metrics.get("precision", 0) < 0.7:
            recommendations.append("Consider adjusting decision threshold to reduce false positives")
        
        if results.classification_metrics.get("recall", 0) < 0.7:
            recommendations.append("Model may be missing high-risk cases - consider feature engineering or rebalancing")
        
        # Check risk-specific metrics
        if results.risk_specific_metrics.get("early_warning_rate", 0) < 0.6:
            recommendations.append("Improve early warning capability through lead indicator features")
        
        if results.risk_specific_metrics.get("calibration_score", 0) < 0.8:
            recommendations.append("Model predictions may need calibration adjustment")
        
        # Check stability
        if results.stability_metrics.get("temporal_stability", 1) < 0.8:
            recommendations.append("Model performance varies over time - consider periodic retraining")
        
        # Check extreme events
        if results.risk_specific_metrics.get("extreme_event_detection_rate", 0) < 0.5:
            recommendations.append("Enhance model capability for extreme risk event detection")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _assess_model_risk_level(self, results: RiskMetricResults) -> str:
        """Assess overall risk level of deploying this model"""
        
        # Critical metrics for risk assessment
        accuracy = results.classification_metrics.get("accuracy", 0)
        precision = results.classification_metrics.get("precision", 0)
        recall = results.classification_metrics.get("recall", 0)
        stability = results.stability_metrics.get("temporal_stability", 1)
        
        # Calculate risk factors
        performance_risk = 1 - min(accuracy, precision, recall)
        stability_risk = 1 - stability
        
        overall_risk = (performance_risk + stability_risk) / 2
        
        if overall_risk < 0.2:
            return "Low Risk - Model ready for production deployment"
        elif overall_risk < 0.4:
            return "Medium Risk - Monitor model performance closely"
        else:
            return "High Risk - Significant improvements needed before deployment"