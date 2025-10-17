"""
Risk assessment API endpoints for RiskX.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from src.ml.models.risk_scorer import BasicRiskScorer, RiskScore as RiskScoreModel, RiskFactor as RiskFactorModel
from src.cache.cache_manager import CacheManager

router = APIRouter()


class RiskFactorResponse(BaseModel):
    """Risk factor response model."""
    name: str
    category: str
    value: float
    weight: float
    normalized_value: float
    description: str
    confidence: float


class RiskScoreResponse(BaseModel):
    """Risk score response model."""
    overall_score: float = Field(..., description="Overall risk score (0-100)")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    confidence: float = Field(..., description="Confidence in the score (0-1)")
    factors: List[RiskFactorResponse] = Field(..., description="Contributing risk factors")
    timestamp: datetime = Field(..., description="When this score was calculated")
    methodology_version: str = Field(..., description="Risk scoring methodology version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 42.5,
                "risk_level": "medium",
                "confidence": 0.87,
                "factors": [
                    {
                        "name": "unemployment_rate",
                        "category": "employment",
                        "value": 4.2,
                        "weight": 0.15,
                        "normalized_value": 0.35,
                        "description": "Unemployment rate: 4.2%",
                        "confidence": 0.95
                    }
                ],
                "timestamp": "2024-01-15T10:30:00",
                "methodology_version": "1.0"
            }
        }


class FactorAnalysisResponse(BaseModel):
    """Individual risk factor analysis response."""
    factor_name: str
    current_value: float
    risk_contribution: float
    trend_analysis: Dict[str, Any]
    historical_context: Dict[str, Any]
    explanation: str


@router.get("/score", response_model=RiskScoreResponse)
async def get_current_risk_score(
    use_cache: bool = Query(True, description="Whether to use cached data"),
    force_refresh: bool = Query(False, description="Force refresh of all data")
):
    """
    Get current comprehensive risk score.
    
    This endpoint provides the overall risk assessment based on multiple
    economic and financial indicators with explainable AI components.
    
    Args:
        use_cache: Whether to use cached data for faster response
        force_refresh: Force refresh of all underlying data
        db: Database session
        
    Returns:
        RiskScoreResponse: Current risk score with detailed factors
    """
    try:
        cache_manager = CacheManager()
        risk_scorer = BasicRiskScorer(cache_manager)
        
        # Calculate current risk score
        risk_score = await risk_scorer.calculate_risk_score(
            use_cache=use_cache,
            force_refresh=force_refresh
        )
        
        if not risk_score:
            raise HTTPException(
                status_code=503,
                detail="Risk scoring service temporarily unavailable"
            )
        
        # Convert to response model
        factors = [
            RiskFactorResponse(
                name=factor.name,
                category=factor.category,
                value=factor.value,
                weight=factor.weight,
                normalized_value=factor.normalized_value,
                description=factor.description,
                confidence=factor.confidence
            )
            for factor in risk_score.factors
        ]
        
        return RiskScoreResponse(
            overall_score=risk_score.overall_score,
            risk_level=risk_score.risk_level,
            confidence=risk_score.confidence,
            factors=factors,
            timestamp=risk_score.timestamp,
            methodology_version=risk_score.methodology_version
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating risk score: {str(e)}"
        )


@router.get("/factors")
async def get_risk_factors(
    category: Optional[str] = Query(None, description="Filter by factor category"),
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """
    Get individual risk factors with detailed analysis.
    
    Args:
        category: Optional category filter (employment, inflation, etc.)
        use_cache: Whether to use cached data
        
    Returns:
        List of risk factors with detailed analysis
    """
    try:
        cache_manager = CacheManager()
        risk_scorer = BasicRiskScorer(cache_manager)
        
        # Get current risk score to access factors
        risk_score = await risk_scorer.calculate_risk_score(use_cache=use_cache)
        
        if not risk_score:
            raise HTTPException(
                status_code=503,
                detail="Risk factors service temporarily unavailable"
            )
        
        factors = risk_score.factors
        
        # Filter by category if specified
        if category:
            factors = [f for f in factors if f.category.lower() == category.lower()]
        
        # Convert to detailed analysis format
        detailed_factors = []
        for factor in factors:
            detailed_factors.append({
                "factor_name": factor.name,
                "category": factor.category,
                "current_value": factor.value,
                "normalized_risk": factor.normalized_value,
                "weight": factor.weight,
                "risk_contribution": factor.normalized_value * factor.weight,
                "description": factor.description,
                "confidence": factor.confidence,
                "risk_level": _determine_factor_risk_level(factor.normalized_value),
                "explanation": _generate_factor_explanation(factor)
            })
        
        return {
            "factors": detailed_factors,
            "total_factors": len(detailed_factors),
            "category_filter": category,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving risk factors: {str(e)}"
        )


@router.get("/analysis/{factor_name}")
async def get_factor_analysis(
    factor_name: str,
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """
    Get detailed analysis for a specific risk factor.
    
    Args:
        factor_name: Name of the risk factor to analyze
        use_cache: Whether to use cached data
        
    Returns:
        Detailed factor analysis including trends and context
    """
    try:
        cache_manager = CacheManager()
        risk_scorer = BasicRiskScorer(cache_manager)
        
        # Get current risk score
        risk_score = await risk_scorer.calculate_risk_score(use_cache=use_cache)
        
        if not risk_score:
            raise HTTPException(
                status_code=503,
                detail="Risk analysis service temporarily unavailable"
            )
        
        # Find the specific factor
        target_factor = None
        for factor in risk_score.factors:
            if factor.name == factor_name:
                target_factor = factor
                break
        
        if not target_factor:
            raise HTTPException(
                status_code=404,
                detail=f"Risk factor '{factor_name}' not found"
            )
        
        # Generate detailed analysis
        analysis = {
            "factor_name": target_factor.name,
            "category": target_factor.category,
            "current_value": target_factor.value,
            "normalized_risk": target_factor.normalized_value,
            "weight_in_overall_score": target_factor.weight,
            "absolute_contribution": target_factor.normalized_value * target_factor.weight,
            "confidence": target_factor.confidence,
            "description": target_factor.description,
            "risk_level": _determine_factor_risk_level(target_factor.normalized_value),
            "detailed_explanation": _generate_detailed_factor_explanation(target_factor),
            "risk_thresholds": _get_factor_thresholds(factor_name),
            "recommendations": _generate_factor_recommendations(target_factor),
            "timestamp": datetime.utcnow()
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing risk factor: {str(e)}"
        )


@router.get("/categories")
async def get_risk_categories():
    """
    Get all available risk factor categories.
    
    Returns:
        List of risk categories with descriptions
    """
    categories = {
        "employment": {
            "description": "Labor market and unemployment indicators",
            "key_factors": ["unemployment_rate", "job_growth", "labor_participation"]
        },
        "inflation": {
            "description": "Price stability and inflation measures",
            "key_factors": ["inflation_rate", "core_inflation", "price_expectations"]
        },
        "interest_rates": {
            "description": "Monetary policy and interest rate environment",
            "key_factors": ["federal_funds_rate", "yield_curve_spread", "real_rates"]
        },
        "economic_growth": {
            "description": "GDP growth and economic activity",
            "key_factors": ["gdp_growth", "productivity", "business_investment"]
        },
        "trade": {
            "description": "International trade and supply chain factors",
            "key_factors": ["trade_balance", "export_growth", "supply_disruptions"]
        },
        "financial_stress": {
            "description": "Financial market conditions and stress indicators",
            "key_factors": ["financial_stress_index", "credit_spreads", "market_volatility"]
        }
    }
    
    return {
        "categories": categories,
        "total_categories": len(categories),
        "timestamp": datetime.utcnow()
    }


@router.get("/methodology")
async def get_risk_methodology():
    """
    Get detailed information about the risk scoring methodology.
    
    Returns:
        Detailed methodology explanation and transparency information
    """
    methodology = {
        "version": "1.0",
        "description": "Multi-factor risk scoring using economic and financial indicators",
        "approach": "Weighted aggregation of normalized risk factors",
        "data_sources": [
            "Federal Reserve Economic Data (FRED)",
            "Bureau of Economic Analysis (BEA)",
            "Bureau of Labor Statistics (BLS)"
        ],
        "scoring_scale": {
            "range": "0-100",
            "interpretation": {
                "0-25": "Low Risk",
                "25-50": "Medium Risk", 
                "50-75": "High Risk",
                "75-100": "Critical Risk"
            }
        },
        "factor_weights": {
            "unemployment_rate": 0.15,
            "inflation_rate": 0.12,
            "federal_funds_rate": 0.10,
            "yield_curve_spread": 0.18,
            "gdp_growth": 0.15,
            "trade_balance": 0.08,
            "financial_stress_index": 0.12,
            "personal_savings_rate": 0.10
        },
        "update_frequency": "Every 30 minutes with cached fallbacks",
        "confidence_calculation": "Weighted average of individual factor confidences",
        "bias_mitigation": [
            "Multiple independent data sources",
            "Transparent weighting scheme",
            "Historical backtesting validation",
            "Regular methodology review"
        ],
        "limitations": [
            "Based on publicly available data with reporting lags",
            "Weights derived from historical relationships",
            "May not capture unprecedented events",
            "Requires human interpretation for policy decisions"
        ],
        "timestamp": datetime.utcnow()
    }
    
    return methodology


def _determine_factor_risk_level(normalized_value: float) -> str:
    """Determine risk level for a normalized factor value."""
    if normalized_value < 0.25:
        return "low"
    elif normalized_value < 0.5:
        return "medium"
    elif normalized_value < 0.75:
        return "high"
    else:
        return "critical"


def _generate_factor_explanation(factor: RiskFactorModel) -> str:
    """Generate explanation for a risk factor."""
    risk_level = _determine_factor_risk_level(factor.normalized_value)
    contribution_pct = round(factor.normalized_value * factor.weight * 100, 1)
    
    return (f"{factor.description}. Current risk level: {risk_level}. "
            f"This factor contributes {contribution_pct}% to the overall risk score.")


def _generate_detailed_factor_explanation(factor: RiskFactorModel) -> str:
    """Generate detailed explanation for a specific factor."""
    risk_level = _determine_factor_risk_level(factor.normalized_value)
    
    explanations = {
        "unemployment_rate": f"Unemployment at {factor.value}% indicates {risk_level} labor market stress. Higher unemployment typically signals economic weakness and reduced consumer spending power.",
        "inflation_rate": f"Inflation at {factor.value}% represents {risk_level} price stability risk. Inflation outside the 2-3% target range can indicate economic imbalances.",
        "federal_funds_rate": f"Federal funds rate at {factor.value}% shows {risk_level} monetary policy tightness. Higher rates can slow economic growth and increase borrowing costs.",
        "yield_curve_spread": f"Yield curve spread at {factor.value}% indicates {risk_level} recession risk. Inverted curves (negative spreads) historically precede recessions.",
        "gdp_growth": f"GDP growth at {factor.value}% represents {risk_level} economic expansion. Negative growth indicates recession while very high growth may be unsustainable.",
        "trade_balance": f"Trade balance of ${factor.value}B shows {risk_level} external sector risk. Large deficits can indicate competitiveness issues.",
        "financial_stress_index": f"Financial stress index at {factor.value} indicates {risk_level} financial market conditions. Higher values suggest increased market stress."
    }
    
    return explanations.get(factor.name, f"Current value of {factor.value} indicates {risk_level} risk level.")


def _get_factor_thresholds(factor_name: str) -> Dict[str, float]:
    """Get risk thresholds for a specific factor."""
    thresholds = {
        "unemployment_rate": {"low": 3.5, "high": 7.0},
        "inflation_rate": {"low": 1.5, "high": 4.0, "target": 2.0},
        "federal_funds_rate": {"low": 1.0, "high": 5.0},
        "yield_curve_spread": {"low": 0.5, "high": -0.5},
        "gdp_growth": {"low": 2.0, "high": -1.0},
        "trade_balance": {"low": -50.0, "high": -100.0},
        "financial_stress_index": {"low": 0.0, "high": 1.0}
    }
    
    return thresholds.get(factor_name, {"low": 0, "high": 1})


def _generate_factor_recommendations(factor: RiskFactorModel) -> List[str]:
    """Generate recommendations based on factor risk level."""
    risk_level = _determine_factor_risk_level(factor.normalized_value)
    
    if risk_level == "low":
        return [f"{factor.description} is at healthy levels", "Continue monitoring for changes"]
    elif risk_level == "medium":
        return [f"{factor.description} shows moderate risk", "Monitor closely for deterioration", "Consider defensive positioning"]
    elif risk_level == "high":
        return [f"{factor.description} indicates elevated risk", "Implement risk mitigation strategies", "Prepare for potential disruptions"]
    else:  # critical
        return [f"{factor.description} shows critical risk levels", "Immediate attention required", "Activate contingency plans"]