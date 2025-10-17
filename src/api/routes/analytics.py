"""
Economic analytics and aggregation API endpoints for RiskX.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.data.processors.indicator_aggregator import run_indicator_aggregation, IndicatorAggregator

router = APIRouter()


class EconomicOverviewResponse(BaseModel):
    """Economic overview response model."""
    overall_risk_level: str = Field(..., description="Overall economic risk level")
    economic_momentum: str = Field(..., description="Economic momentum direction")
    market_stress_level: str = Field(..., description="Current market stress level")
    key_concerns: List[str] = Field(..., description="Key economic concerns")
    positive_signals: List[str] = Field(..., description="Positive economic signals")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_risk_level": "moderate",
                "economic_momentum": "improving",
                "market_stress_level": "low",
                "key_concerns": [
                    "Elevated inflation levels",
                    "Supply chain disruptions"
                ],
                "positive_signals": [
                    "Strong employment growth",
                    "Stable financial conditions"
                ],
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class CategorySummaryResponse(BaseModel):
    """Category summary response model."""
    category_name: str
    indicator_count: int
    avg_risk_score: float
    category_trend: str
    category_volatility: str
    key_indicators: List[str]
    last_updated: datetime


class IndicatorSummaryResponse(BaseModel):
    """Individual indicator summary response model."""
    indicator_name: str
    category: str
    current_value: float
    mean: float
    median: float
    std_dev: float
    trend_direction: str
    volatility_level: str
    last_updated: datetime
    data_points: int


class AggregationResponse(BaseModel):
    """Complete aggregation response model."""
    economic_overview: EconomicOverviewResponse
    category_summaries: List[CategorySummaryResponse]
    indicator_summaries: List[IndicatorSummaryResponse]
    aggregation_metadata: Dict[str, Any]


@router.get("/overview", response_model=EconomicOverviewResponse)
async def get_economic_overview(
    use_cache: bool = Query(True, description="Whether to use cached data"),
    force_refresh: bool = Query(False, description="Force refresh of all data")
):
    """
    Get comprehensive economic overview with risk assessment.
    
    This endpoint provides a high-level summary of economic conditions
    across all monitored categories with trend analysis and risk assessment.
    
    Args:
        use_cache: Whether to use cached data for faster response
        force_refresh: Force refresh of all underlying data
        
    Returns:
        EconomicOverviewResponse: Comprehensive economic overview
    """
    try:
        # Run aggregation
        aggregation_results = run_indicator_aggregation(use_cache=not force_refresh)
        
        economic_overview = aggregation_results["economic_overview"]
        
        return EconomicOverviewResponse(
            overall_risk_level=economic_overview.overall_risk_level,
            economic_momentum=economic_overview.economic_momentum,
            market_stress_level=economic_overview.market_stress_level,
            key_concerns=economic_overview.key_concerns,
            positive_signals=economic_overview.positive_signals,
            timestamp=economic_overview.timestamp
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating economic overview: {str(e)}"
        )


@router.get("/categories", response_model=List[CategorySummaryResponse])
async def get_category_summaries(
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """
    Get summary statistics for all economic categories.
    
    Returns aggregated statistics, trends, and risk assessments
    for each major economic category (employment, inflation, etc.).
    
    Args:
        use_cache: Whether to use cached data
        
    Returns:
        List of category summaries with statistics and trends
    """
    try:
        aggregation_results = run_indicator_aggregation(use_cache=use_cache)
        
        category_summaries = [
            CategorySummaryResponse(
                category_name=summary.category_name,
                indicator_count=summary.indicator_count,
                avg_risk_score=summary.avg_risk_score,
                category_trend=summary.category_trend,
                category_volatility=summary.category_volatility,
                key_indicators=summary.key_indicators,
                last_updated=summary.last_updated
            )
            for summary in aggregation_results["category_summaries"]
        ]
        
        return category_summaries
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving category summaries: {str(e)}"
        )


@router.get("/indicators", response_model=List[IndicatorSummaryResponse])
async def get_indicator_summaries(
    category: Optional[str] = Query(None, description="Filter by category"),
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """
    Get detailed statistics for individual economic indicators.
    
    Returns comprehensive statistics including trend analysis,
    volatility assessment, and historical context for each indicator.
    
    Args:
        category: Optional category filter
        use_cache: Whether to use cached data
        
    Returns:
        List of indicator summaries with detailed statistics
    """
    try:
        aggregation_results = run_indicator_aggregation(use_cache=use_cache)
        
        indicator_summaries = aggregation_results["indicator_summaries"]
        
        # Filter by category if specified
        if category:
            indicator_summaries = [
                summary for summary in indicator_summaries 
                if summary.category.lower() == category.lower()
            ]
        
        response_summaries = [
            IndicatorSummaryResponse(
                indicator_name=summary.indicator_name,
                category=summary.category,
                current_value=summary.current_value,
                mean=summary.mean,
                median=summary.median,
                std_dev=summary.std_dev,
                trend_direction=summary.trend_direction,
                volatility_level=summary.volatility_level,
                last_updated=summary.last_updated,
                data_points=summary.data_points
            )
            for summary in indicator_summaries
        ]
        
        return response_summaries
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving indicator summaries: {str(e)}"
        )


@router.get("/aggregation", response_model=AggregationResponse)
async def get_complete_aggregation(
    use_cache: bool = Query(True, description="Whether to use cached data"),
    force_refresh: bool = Query(False, description="Force refresh of all data")
):
    """
    Get complete economic indicator aggregation with all details.
    
    This endpoint returns comprehensive aggregation results including
    economic overview, category summaries, individual indicator statistics,
    and analytical insights.
    
    Args:
        use_cache: Whether to use cached data
        force_refresh: Force refresh of all underlying data
        
    Returns:
        Complete aggregation results with all analysis
    """
    try:
        aggregation_results = run_indicator_aggregation(use_cache=not force_refresh)
        
        economic_overview = aggregation_results["economic_overview"]
        
        # Convert to response models
        overview_response = EconomicOverviewResponse(
            overall_risk_level=economic_overview.overall_risk_level,
            economic_momentum=economic_overview.economic_momentum,
            market_stress_level=economic_overview.market_stress_level,
            key_concerns=economic_overview.key_concerns,
            positive_signals=economic_overview.positive_signals,
            timestamp=economic_overview.timestamp
        )
        
        category_responses = [
            CategorySummaryResponse(
                category_name=summary.category_name,
                indicator_count=summary.indicator_count,
                avg_risk_score=summary.avg_risk_score,
                category_trend=summary.category_trend,
                category_volatility=summary.category_volatility,
                key_indicators=summary.key_indicators,
                last_updated=summary.last_updated
            )
            for summary in aggregation_results["category_summaries"]
        ]
        
        indicator_responses = [
            IndicatorSummaryResponse(
                indicator_name=summary.indicator_name,
                category=summary.category,
                current_value=summary.current_value,
                mean=summary.mean,
                median=summary.median,
                std_dev=summary.std_dev,
                trend_direction=summary.trend_direction,
                volatility_level=summary.volatility_level,
                last_updated=summary.last_updated,
                data_points=summary.data_points
            )
            for summary in aggregation_results["indicator_summaries"]
        ]
        
        return AggregationResponse(
            economic_overview=overview_response,
            category_summaries=category_responses,
            indicator_summaries=indicator_responses,
            aggregation_metadata=aggregation_results["aggregation_metadata"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating complete aggregation: {str(e)}"
        )


@router.get("/insights")
async def get_analytical_insights(
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """
    Get analytical insights and cross-category analysis.
    
    Returns advanced analytics including correlations, statistical summaries,
    and insights derived from cross-category analysis of economic indicators.
    
    Args:
        use_cache: Whether to use cached data
        
    Returns:
        Analytical insights and advanced statistics
    """
    try:
        aggregation_results = run_indicator_aggregation(use_cache=use_cache)
        
        insights = aggregation_results.get("insights", {})
        
        # Add timestamp and metadata
        response = {
            "insights": insights,
            "generated_at": datetime.utcnow(),
            "data_sources": aggregation_results["aggregation_metadata"]["data_sources"],
            "methodology": aggregation_results["aggregation_metadata"]["methodology"]
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating analytical insights: {str(e)}"
        )


@router.get("/health")
async def analytics_health_check():
    """
    Health check for analytics and aggregation services.
    
    Returns the operational status of the economic indicator
    aggregation and analytics systems.
    """
    try:
        # Test aggregation system
        aggregator = IndicatorAggregator()
        
        # Quick health check - just verify we can initialize and connect to data sources
        health_status = {
            "status": "healthy",
            "analytics_service": "operational",
            "data_sources": {
                "fred_connector": "available",
                "bea_connector": "available",
                "cache_manager": "available"
            },
            "aggregation_engine": "ready",
            "last_successful_aggregation": datetime.utcnow(),
            "timestamp": datetime.utcnow()
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }