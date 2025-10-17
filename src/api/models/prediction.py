"""
Prediction-related API models for RiskX platform.
Pydantic models for ML predictions, forecasting, and model explanation endpoints.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator


class PredictionType(str, Enum):
    """Prediction type enumeration."""
    RISK_SCORE = "risk_score"
    MARKET_VOLATILITY = "market_volatility"
    SUPPLY_DISRUPTION = "supply_disruption"
    ECONOMIC_INDICATOR = "economic_indicator"
    FINANCIAL_STRESS = "financial_stress"


class ModelType(str, Enum):
    """Model type enumeration."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionRequest(BaseModel):
    """Request model for ML predictions."""
    
    prediction_type: PredictionType = Field(..., description="Type of prediction requested")
    input_data: Dict[str, Union[float, int, str]] = Field(..., description="Input features for prediction")
    time_horizon: int = Field(30, ge=1, le=365, description="Prediction time horizon in days")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    include_explanation: bool = Field(True, description="Include prediction explanation")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    scenario_name: Optional[str] = Field(None, description="Named scenario for prediction context")
    
    @validator('input_data')
    def validate_input_data(cls, v):
        """Validate input data is not empty."""
        if not v:
            raise ValueError("Input data cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_type": "risk_score",
                "input_data": {
                    "unemployment_rate": 3.8,
                    "inflation_rate": 2.1,
                    "gdp_growth": 2.5,
                    "credit_spread": 125.5
                },
                "time_horizon": 30,
                "model_version": "v2.1.0",
                "include_explanation": True,
                "include_confidence_intervals": True,
                "scenario_name": "baseline"
            }
        }


class ModelPrediction(BaseModel):
    """Individual model prediction result."""
    
    value: float = Field(..., description="Predicted value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level classification")
    lower_bound: Optional[float] = Field(None, description="Lower confidence bound")
    upper_bound: Optional[float] = Field(None, description="Upper confidence bound")
    probability_distribution: Optional[Dict[str, float]] = Field(
        None, description="Probability distribution for categorical predictions"
    )
    
    @validator('confidence_level')
    def derive_confidence_level(cls, v, values):
        """Derive confidence level from confidence score."""
        confidence = values.get('confidence', 0)
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    class Config:
        json_schema_extra = {
            "example": {
                "value": 72.3,
                "confidence": 0.85,
                "confidence_level": "high",
                "lower_bound": 68.1,
                "upper_bound": 76.5,
                "probability_distribution": {
                    "low_risk": 0.15,
                    "moderate_risk": 0.35,
                    "high_risk": 0.45,
                    "critical_risk": 0.05
                }
            }
        }


class ModelExplanation(BaseModel):
    """Model explanation and feature importance."""
    
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP values for features")
    prediction_reasoning: List[str] = Field(..., description="Human-readable reasoning")
    uncertainty_sources: List[str] = Field(default_factory=list, description="Sources of uncertainty")
    model_bias_check: Dict[str, Any] = Field(..., description="Bias detection results")
    counterfactual_examples: Optional[List[Dict[str, Any]]] = Field(
        None, description="Counterfactual explanation examples"
    )
    
    @validator('feature_importance')
    def validate_feature_importance_sum(cls, v):
        """Validate feature importance values sum to approximately 1."""
        total = sum(abs(score) for score in v.values())
        if total > 0 and abs(total - 1.0) > 0.1:
            # Normalize if sum is significantly different from 1
            v = {feature: score / total for feature, score in v.items()}
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_importance": {
                    "unemployment_rate": 0.35,
                    "credit_spread": 0.28,
                    "inflation_rate": 0.22,
                    "gdp_growth": 0.15
                },
                "shap_values": {
                    "unemployment_rate": 2.8,
                    "credit_spread": -1.5,
                    "inflation_rate": 0.8,
                    "gdp_growth": -0.6
                },
                "prediction_reasoning": [
                    "Elevated unemployment rate suggests economic stress",
                    "Credit spreads indicate financial market tension",
                    "Inflation rate within normal range"
                ],
                "uncertainty_sources": [
                    "Limited recent data for supply chain indicators",
                    "Model uncertainty in extreme scenarios"
                ],
                "model_bias_check": {
                    "bias_detected": False,
                    "fairness_metrics": {"demographic_parity": 0.95}
                }
            }
        }


class PredictionMetadata(BaseModel):
    """Metadata about the prediction process."""
    
    model_name: str = Field(..., description="Name of the model used")
    model_type: ModelType = Field(..., description="Type of ML model")
    model_version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="Model training date")
    prediction_date: datetime = Field(..., description="Prediction generation date")
    data_vintage: datetime = Field(..., description="Latest input data timestamp")
    computational_time_ms: float = Field(..., description="Prediction computation time")
    cache_hit: bool = Field(False, description="Whether result was cached")
    data_quality_score: float = Field(..., ge=0, le=1, description="Input data quality score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "RiskScore_Ensemble_v2",
                "model_type": "ensemble",
                "model_version": "v2.1.0",
                "training_date": "2024-01-10T08:00:00Z",
                "prediction_date": "2024-01-15T10:30:00Z",
                "data_vintage": "2024-01-15T09:00:00Z",
                "computational_time_ms": 245.7,
                "cache_hit": False,
                "data_quality_score": 0.92
            }
        }


class PredictionResponse(BaseModel):
    """Response model for ML predictions."""
    
    prediction: ModelPrediction = Field(..., description="Main prediction result")
    explanation: Optional[ModelExplanation] = Field(None, description="Prediction explanation")
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    alternative_scenarios: Optional[Dict[str, ModelPrediction]] = Field(
        None, description="Predictions under alternative scenarios"
    )
    historical_accuracy: Optional[Dict[str, float]] = Field(
        None, description="Historical model accuracy metrics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "value": 72.3,
                    "confidence": 0.85,
                    "confidence_level": "high",
                    "lower_bound": 68.1,
                    "upper_bound": 76.5
                },
                "explanation": {
                    "feature_importance": {
                        "unemployment_rate": 0.35,
                        "credit_spread": 0.28
                    },
                    "prediction_reasoning": [
                        "Elevated unemployment rate suggests economic stress"
                    ],
                    "model_bias_check": {"bias_detected": False}
                },
                "metadata": {
                    "model_name": "RiskScore_Ensemble_v2",
                    "model_type": "ensemble",
                    "model_version": "v2.1.0",
                    "prediction_date": "2024-01-15T10:30:00Z",
                    "computational_time_ms": 245.7,
                    "data_quality_score": 0.92
                }
            }
        }


class ForecastRequest(BaseModel):
    """Request model for time series forecasting."""
    
    series_name: str = Field(..., description="Name of time series to forecast")
    forecast_horizon: int = Field(..., ge=1, le=365, description="Forecast horizon in days")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level for intervals")
    seasonal_adjustment: bool = Field(True, description="Apply seasonal adjustment")
    include_scenarios: bool = Field(False, description="Include scenario-based forecasts")
    external_factors: Optional[Dict[str, float]] = Field(
        None, description="External factors to include in forecast"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "series_name": "unemployment_rate",
                "forecast_horizon": 90,
                "confidence_level": 0.95,
                "seasonal_adjustment": True,
                "include_scenarios": True,
                "external_factors": {
                    "policy_intervention": 0.1,
                    "economic_shock": -0.2
                }
            }
        }


class ForecastPoint(BaseModel):
    """Individual forecast point."""
    
    date: date = Field(..., description="Forecast date")
    value: float = Field(..., description="Forecasted value")
    lower_bound: float = Field(..., description="Lower confidence bound")
    upper_bound: float = Field(..., description="Upper confidence bound")
    trend_component: Optional[float] = Field(None, description="Trend component")
    seasonal_component: Optional[float] = Field(None, description="Seasonal component")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-02-15",
                "value": 3.9,
                "lower_bound": 3.6,
                "upper_bound": 4.2,
                "trend_component": 0.1,
                "seasonal_component": -0.05
            }
        }


class ForecastResponse(BaseModel):
    """Response model for time series forecasting."""
    
    series_name: str = Field(..., description="Name of forecasted series")
    forecast: List[ForecastPoint] = Field(..., description="Forecast points")
    model_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    scenario_forecasts: Optional[Dict[str, List[ForecastPoint]]] = Field(
        None, description="Scenario-based forecasts"
    )
    forecast_metadata: Dict[str, Any] = Field(..., description="Forecast metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "series_name": "unemployment_rate",
                "forecast": [
                    {
                        "date": "2024-02-15",
                        "value": 3.9,
                        "lower_bound": 3.6,
                        "upper_bound": 4.2
                    }
                ],
                "model_metrics": {
                    "mape": 0.08,
                    "rmse": 0.15,
                    "mae": 0.12
                },
                "forecast_metadata": {
                    "model_type": "arima",
                    "data_points_used": 720,
                    "forecast_date": "2024-01-15T10:30:00Z"
                }
            }
        }


class ModelPerformanceRequest(BaseModel):
    """Request model for model performance evaluation."""
    
    model_name: str = Field(..., description="Name of model to evaluate")
    evaluation_period_days: int = Field(90, ge=7, le=365, description="Evaluation period in days")
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"],
        description="Performance metrics to calculate"
    )
    include_bias_analysis: bool = Field(True, description="Include bias analysis")
    include_drift_detection: bool = Field(True, description="Include model drift detection")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "RiskScore_Ensemble_v2",
                "evaluation_period_days": 90,
                "metrics": ["accuracy", "precision", "recall", "auc"],
                "include_bias_analysis": True,
                "include_drift_detection": True
            }
        }


class ModelPerformanceResponse(BaseModel):
    """Response model for model performance evaluation."""
    
    model_name: str = Field(..., description="Evaluated model name")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    bias_analysis: Optional[Dict[str, Any]] = Field(None, description="Bias analysis results")
    drift_detection: Optional[Dict[str, Any]] = Field(None, description="Model drift detection")
    evaluation_period: Dict[str, date] = Field(..., description="Evaluation period")
    recommendation: str = Field(..., description="Performance recommendation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "RiskScore_Ensemble_v2",
                "performance_metrics": {
                    "accuracy": 0.87,
                    "precision": 0.84,
                    "recall": 0.89,
                    "auc": 0.91
                },
                "bias_analysis": {
                    "demographic_parity": 0.95,
                    "equality_of_opportunity": 0.93,
                    "bias_detected": False
                },
                "drift_detection": {
                    "drift_detected": False,
                    "drift_score": 0.12,
                    "drift_threshold": 0.3
                },
                "evaluation_period": {
                    "start_date": "2023-10-15",
                    "end_date": "2024-01-15"
                },
                "recommendation": "Model performing within acceptable ranges. Continue monitoring."
            }
        }