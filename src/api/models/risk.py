"""
Risk-related API models for RiskX platform.
Pydantic models for risk assessment, scoring, and analysis endpoints.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    ECONOMIC = "economic"
    FINANCIAL = "financial"
    SUPPLY_CHAIN = "supply_chain"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    CYBER = "cyber"
    ENVIRONMENTAL = "environmental"


class RiskIndicator(BaseModel):
    """Individual risk indicator model."""
    
    name: str = Field(..., description="Indicator name")
    value: float = Field(..., description="Current indicator value")
    normalized_value: float = Field(..., ge=0, le=1, description="Normalized value (0-1)")
    weight: float = Field(..., ge=0, le=1, description="Weight in overall calculation")
    category: RiskCategory = Field(..., description="Risk category")
    description: Optional[str] = Field(None, description="Indicator description")
    source: Optional[str] = Field(None, description="Data source")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Credit Default Swap Spread",
                "value": 125.5,
                "normalized_value": 0.65,
                "weight": 0.25,
                "category": "financial",
                "description": "5-year CDS spread for financial sector",
                "source": "Federal Reserve Economic Data",
                "last_updated": "2024-01-15T10:30:00Z"
            }
        }


class RiskMetrics(BaseModel):
    """Risk metrics and statistics."""
    
    overall_score: float = Field(..., ge=0, le=100, description="Overall risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    volatility: float = Field(..., ge=0, description="Risk volatility measure")
    trend: str = Field(..., description="Risk trend direction", regex="^(increasing|stable|decreasing)$")
    percentile: float = Field(..., ge=0, le=100, description="Percentile ranking")
    
    # Category-specific scores
    category_scores: Dict[RiskCategory, float] = Field(
        ..., description="Risk scores by category"
    )
    
    # Time series data
    historical_scores: List[Dict[str, Union[datetime, float]]] = Field(
        default_factory=list, description="Historical risk scores"
    )
    
    @validator('category_scores')
    def validate_category_scores(cls, v):
        """Validate category scores are within valid range."""
        for category, score in v.items():
            if not 0 <= score <= 100:
                raise ValueError(f"Score for {category} must be between 0 and 100")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 68.5,
                "risk_level": "high",
                "confidence": 0.85,
                "volatility": 12.3,
                "trend": "increasing",
                "percentile": 78.2,
                "category_scores": {
                    "economic": 72.1,
                    "financial": 65.8,
                    "supply_chain": 58.9
                },
                "historical_scores": [
                    {"timestamp": "2024-01-14T00:00:00Z", "score": 66.2},
                    {"timestamp": "2024-01-15T00:00:00Z", "score": 68.5}
                ]
            }
        }


class RiskScoreRequest(BaseModel):
    """Request model for risk score calculation."""
    
    indicators: Dict[str, float] = Field(..., description="Risk indicators with values")
    weights: Optional[Dict[str, float]] = Field(None, description="Custom indicator weights")
    time_horizon: int = Field(30, ge=1, le=365, description="Time horizon in days")
    include_confidence: bool = Field(True, description="Include confidence metrics")
    include_historical: bool = Field(False, description="Include historical data")
    
    @validator('weights')
    def validate_weights(cls, v, values):
        """Validate weights sum to 1.0 and match indicators."""
        if v is None:
            return v
        
        indicators = values.get('indicators', {})
        
        # Check weights match indicators
        if set(v.keys()) != set(indicators.keys()):
            raise ValueError("Weight keys must match indicator keys")
        
        # Check weights sum to approximately 1.0
        weight_sum = sum(v.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "indicators": {
                    "unemployment_rate": 3.8,
                    "inflation_rate": 2.1,
                    "credit_spread": 125.5
                },
                "weights": {
                    "unemployment_rate": 0.4,
                    "inflation_rate": 0.3,
                    "credit_spread": 0.3
                },
                "time_horizon": 30,
                "include_confidence": True,
                "include_historical": False
            }
        }


class RiskScoreResponse(BaseModel):
    """Response model for risk score calculation."""
    
    metrics: RiskMetrics = Field(..., description="Risk metrics and scores")
    indicators: List[RiskIndicator] = Field(..., description="Individual risk indicators")
    explanation: Dict[str, Any] = Field(..., description="Risk score explanation")
    metadata: Dict[str, Any] = Field(..., description="Calculation metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "overall_score": 68.5,
                    "risk_level": "high",
                    "confidence": 0.85,
                    "volatility": 12.3,
                    "trend": "increasing",
                    "percentile": 78.2,
                    "category_scores": {
                        "economic": 72.1,
                        "financial": 65.8
                    }
                },
                "indicators": [
                    {
                        "name": "Unemployment Rate",
                        "value": 3.8,
                        "normalized_value": 0.62,
                        "weight": 0.4,
                        "category": "economic"
                    }
                ],
                "explanation": {
                    "top_factors": ["unemployment_rate", "credit_spread"],
                    "methodology": "Weighted composite score"
                },
                "metadata": {
                    "calculation_time": "2024-01-15T10:30:00Z",
                    "data_freshness": 95.2,
                    "model_version": "v2.1.0"
                }
            }
        }


class RiskFactorsResponse(BaseModel):
    """Response model for risk factors analysis."""
    
    factors: List[Dict[str, Any]] = Field(..., description="Risk factors with impact scores")
    correlations: Dict[str, Dict[str, float]] = Field(..., description="Factor correlations")
    importance_rankings: List[str] = Field(..., description="Factors ranked by importance")
    time_series: Optional[Dict[str, List[Dict[str, Union[datetime, float]]]]] = Field(
        None, description="Time series data for factors"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "factors": [
                    {
                        "name": "Interest Rate Volatility",
                        "impact_score": 0.78,
                        "direction": "positive",
                        "category": "financial"
                    }
                ],
                "correlations": {
                    "unemployment_rate": {
                        "inflation_rate": -0.23,
                        "credit_spread": 0.65
                    }
                },
                "importance_rankings": [
                    "credit_spread",
                    "unemployment_rate",
                    "inflation_rate"
                ]
            }
        }


class RiskCategoriesResponse(BaseModel):
    """Response model for risk categories breakdown."""
    
    categories: Dict[RiskCategory, Dict[str, Any]] = Field(
        ..., description="Risk breakdown by category"
    )
    category_trends: Dict[RiskCategory, str] = Field(
        ..., description="Trend direction for each category"
    )
    cross_category_impacts: Dict[str, float] = Field(
        ..., description="Cross-category impact scores"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "categories": {
                    "economic": {
                        "score": 72.1,
                        "level": "high",
                        "indicators_count": 8,
                        "top_contributors": ["unemployment_rate", "gdp_growth"]
                    },
                    "financial": {
                        "score": 65.8,
                        "level": "moderate",
                        "indicators_count": 6,
                        "top_contributors": ["credit_spread", "bank_health"]
                    }
                },
                "category_trends": {
                    "economic": "increasing",
                    "financial": "stable"
                },
                "cross_category_impacts": {
                    "economic_to_financial": 0.72,
                    "financial_to_supply_chain": 0.45
                }
            }
        }


class RiskAlertRequest(BaseModel):
    """Request model for risk alert configuration."""
    
    alert_name: str = Field(..., description="Alert name")
    category: RiskCategory = Field(..., description="Risk category to monitor")
    threshold: float = Field(..., ge=0, le=100, description="Alert threshold")
    condition: str = Field(..., description="Alert condition", regex="^(above|below|change)$")
    notification_channels: List[str] = Field(..., description="Notification channels")
    enabled: bool = Field(True, description="Whether alert is enabled")
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_name": "High Financial Risk Alert",
                "category": "financial",
                "threshold": 80.0,
                "condition": "above",
                "notification_channels": ["email", "webhook"],
                "enabled": True
            }
        }


class RiskAlertResponse(BaseModel):
    """Response model for risk alert operations."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_name: str = Field(..., description="Alert name")
    status: str = Field(..., description="Alert status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_triggered: Optional[datetime] = Field(None, description="Last trigger timestamp")
    trigger_count: int = Field(0, description="Number of times triggered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_12345",
                "alert_name": "High Financial Risk Alert",
                "status": "active",
                "created_at": "2024-01-15T10:00:00Z",
                "last_triggered": "2024-01-15T14:30:00Z",
                "trigger_count": 3
            }
        }


# Common response models
class APIResponse(BaseModel):
    """Base API response model."""
    
    success: bool = Field(True, description="Request success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": True,
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid input data provided",
                "details": {
                    "field_errors": {
                        "indicators": ["This field is required"]
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    
    data: List[Any] = Field(..., description="Response data")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")
    
    @root_validator
    def calculate_pages(cls, values):
        """Calculate total pages from total and size."""
        total = values.get('total', 0)
        size = values.get('size', 20)
        values['pages'] = (total + size - 1) // size if size > 0 else 0
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [],
                "total": 150,
                "page": 1,
                "size": 20,
                "pages": 8
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    dependencies: Dict[str, Dict[str, Any]] = Field(..., description="Dependency status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "dependencies": {
                    "database": {"status": "healthy", "response_time_ms": 15},
                    "cache": {"status": "healthy", "response_time_ms": 8},
                    "external_apis": {"status": "degraded", "healthy_count": 8, "total_count": 10}
                }
            }
        }