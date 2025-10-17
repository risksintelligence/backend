"""
Data access API endpoints for economic, financial, and supply chain data.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...data.storage.cache import get_cache_instance
from ...data.sources.fred import FREDConnector
from ...data.sources.census import CensusTradeDataFetcher
from ...data.sources.bea import BEAConnector
from ...data.sources.bls import BlsDataFetcher

logger = logging.getLogger('riskx.api.routes.data')

router = APIRouter()


class DataRequest(BaseModel):
    """Request model for data queries."""
    source: str = Field(..., description="Data source identifier")
    series: Optional[List[str]] = Field(default=None, description="Specific data series to fetch")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    frequency: Optional[str] = Field(default=None, description="Data frequency (daily, weekly, monthly, quarterly)")
    include_metadata: bool = Field(default=False, description="Include series metadata")


class DataResponse(BaseModel):
    """Response model for data queries."""
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict] = None
    count: int
    cache_status: str


@router.get("/sources")
async def get_data_sources():
    """
    Get list of available data sources and their capabilities.
    
    Returns information about all integrated data sources including
    available series, update frequencies, and access status.
    """
    try:
        sources = {
            "fred": {
                "name": "Federal Reserve Economic Data",
                "description": "Economic time series from the Federal Reserve Bank of St. Louis",
                "categories": ["monetary_policy", "economic_indicators", "financial_markets"],
                "update_frequency": "daily",
                "total_series": 760000,
                "key_series": [
                    "GDPC1", "UNRATE", "FEDFUNDS", "DGS10", "CPIAUCSL", "PAYEMS"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            },
            "census": {
                "name": "U.S. Census Bureau Trade Data",
                "description": "International trade statistics and supply chain data",
                "categories": ["trade_flows", "imports", "exports", "supply_chain"],
                "update_frequency": "monthly",
                "total_series": 25000,
                "key_series": [
                    "total_trade", "trade_balance", "imports_by_country", "exports_by_commodity"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            },
            "bea": {
                "name": "Bureau of Economic Analysis",
                "description": "GDP, national accounts, and economic analysis data",
                "categories": ["gdp", "national_accounts", "regional_data"],
                "update_frequency": "quarterly",
                "total_series": 15000,
                "key_series": [
                    "GDP", "PCE", "government_spending", "business_investment"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            },
            "bls": {
                "name": "Bureau of Labor Statistics",
                "description": "Employment, wages, and labor market data",
                "categories": ["employment", "wages", "labor_force", "productivity"],
                "update_frequency": "monthly",
                "total_series": 30000,
                "key_series": [
                    "unemployment_rate", "nonfarm_payrolls", "average_hourly_earnings"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            },
            "fdic": {
                "name": "Federal Deposit Insurance Corporation",
                "description": "Banking sector health and stability indicators",
                "categories": ["banking", "credit_risk", "financial_stability"],
                "update_frequency": "quarterly",
                "total_series": 5000,
                "key_series": [
                    "bank_capital_ratios", "credit_losses", "deposit_growth"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            },
            "noaa": {
                "name": "National Oceanic and Atmospheric Administration",
                "description": "Weather, climate, and natural disaster data",
                "categories": ["weather", "climate", "natural_disasters"],
                "update_frequency": "daily",
                "total_series": 10000,
                "key_series": [
                    "severe_weather_events", "temperature_anomalies", "precipitation"
                ],
                "api_status": "active",
                "last_update": "2024-01-01T00:00:00Z"
            }
        }
        
        return {
            "sources": sources,
            "total_sources": len(sources),
            "active_sources": len([s for s in sources.values() if s["api_status"] == "active"]),
            "last_updated": datetime.utcnow().isoformat(),
            "data_coverage": {
                "economic_indicators": 4,
                "financial_data": 2,
                "trade_data": 1,
                "environmental_data": 1
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data sources: {str(e)}")


@router.get("/series/{source}")
async def get_source_series(
    source: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search term for series names"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of series to return")
):
    """
    Get available data series for a specific source.
    
    Returns list of data series with metadata including descriptions,
    frequencies, and availability dates.
    """
    try:
        source = source.lower()
        
        # Define series catalogs for each source
        series_catalogs = {
            "fred": {
                "monetary_policy": [
                    {"id": "FEDFUNDS", "name": "Federal Funds Rate", "frequency": "monthly", "start": "1954-07-01"},
                    {"id": "DGS10", "name": "10-Year Treasury Rate", "frequency": "daily", "start": "1962-01-02"},
                    {"id": "TB3MS", "name": "3-Month Treasury Rate", "frequency": "monthly", "start": "1934-01-01"}
                ],
                "economic_indicators": [
                    {"id": "GDPC1", "name": "Real GDP", "frequency": "quarterly", "start": "1947-01-01"},
                    {"id": "UNRATE", "name": "Unemployment Rate", "frequency": "monthly", "start": "1948-01-01"},
                    {"id": "CPIAUCSL", "name": "Consumer Price Index", "frequency": "monthly", "start": "1947-01-01"},
                    {"id": "PAYEMS", "name": "Nonfarm Payrolls", "frequency": "monthly", "start": "1939-01-01"}
                ],
                "financial_markets": [
                    {"id": "SP500", "name": "S&P 500", "frequency": "daily", "start": "1957-03-04"},
                    {"id": "DEXUSEU", "name": "USD/EUR Exchange Rate", "frequency": "daily", "start": "1999-01-04"},
                    {"id": "BAMLH0A0HYM2", "name": "High Yield Credit Spread", "frequency": "daily", "start": "1996-12-31"}
                ]
            },
            "census": {
                "trade_flows": [
                    {"id": "total_trade", "name": "Total Trade Value", "frequency": "monthly", "start": "1989-01-01"},
                    {"id": "trade_balance", "name": "Trade Balance", "frequency": "monthly", "start": "1989-01-01"}
                ],
                "imports": [
                    {"id": "total_imports", "name": "Total Imports", "frequency": "monthly", "start": "1989-01-01"},
                    {"id": "imports_china", "name": "Imports from China", "frequency": "monthly", "start": "1989-01-01"}
                ],
                "exports": [
                    {"id": "total_exports", "name": "Total Exports", "frequency": "monthly", "start": "1989-01-01"},
                    {"id": "exports_manufactured", "name": "Manufactured Exports", "frequency": "monthly", "start": "1989-01-01"}
                ]
            },
            "bea": {
                "gdp": [
                    {"id": "GDP", "name": "Gross Domestic Product", "frequency": "quarterly", "start": "1947-01-01"},
                    {"id": "GDPDEF", "name": "GDP Deflator", "frequency": "quarterly", "start": "1947-01-01"}
                ],
                "national_accounts": [
                    {"id": "PCE", "name": "Personal Consumption Expenditures", "frequency": "quarterly", "start": "1947-01-01"},
                    {"id": "GPDI", "name": "Gross Private Domestic Investment", "frequency": "quarterly", "start": "1947-01-01"}
                ]
            },
            "bls": {
                "employment": [
                    {"id": "LNS14000000", "name": "Unemployment Rate", "frequency": "monthly", "start": "1948-01-01"},
                    {"id": "CES0000000001", "name": "Total Nonfarm Payrolls", "frequency": "monthly", "start": "1939-01-01"}
                ],
                "wages": [
                    {"id": "CES0500000003", "name": "Average Hourly Earnings", "frequency": "monthly", "start": "2006-03-01"},
                    {"id": "ECIWAG", "name": "Employment Cost Index", "frequency": "quarterly", "start": "1980-12-01"}
                ]
            }
        }
        
        if source not in series_catalogs:
            raise HTTPException(status_code=404, detail=f"Unknown data source: {source}")
        
        # Get series for the source
        source_series = series_catalogs[source]
        
        # Filter by category if specified
        if category:
            if category not in source_series:
                raise HTTPException(status_code=404, detail=f"Category {category} not found for source {source}")
            filtered_series = {category: source_series[category]}
        else:
            filtered_series = source_series
        
        # Flatten series list
        all_series = []
        for cat, series_list in filtered_series.items():
            for series in series_list:
                series["category"] = cat
                all_series.append(series)
        
        # Apply search filter if specified
        if search:
            search_lower = search.lower()
            all_series = [
                s for s in all_series 
                if search_lower in s["name"].lower() or search_lower in s["id"].lower()
            ]
        
        # Apply limit
        limited_series = all_series[:limit]
        
        return {
            "source": source,
            "category": category,
            "search_term": search,
            "total_found": len(all_series),
            "returned": len(limited_series),
            "series": limited_series,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching series for source {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get series: {str(e)}")


@router.get("/fetch/{source}")
async def fetch_data(
    source: str,
    series_id: str = Query(..., description="Data series identifier"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    frequency: Optional[str] = Query(None, description="Data frequency"),
    use_cache: bool = Query(True, description="Use cached data if available")
):
    """
    Fetch specific data series from a source.
    
    Retrieves time series data with optional date filtering and frequency
    conversion. Uses cache for performance optimization.
    """
    try:
        source = source.lower()
        cache_manager = get_cache_instance()
        
        # Check cache first if enabled
        cache_key = f"data:{source}:{series_id}:{start_date}:{end_date}:{frequency}"
        cached_data = None
        
        if use_cache and cache_manager:
            cached_data = cache_manager.get(cache_key)
            if cached_data:
                return DataResponse(
                    source=source,
                    timestamp=datetime.utcnow(),
                    data=cached_data,
                    count=len(cached_data.get("observations", [])),
                    cache_status="hit"
                )
        
        # Initialize appropriate connector
        connector = None
        if source == "fred":
            connector = FREDConnector()
        elif source == "census":
            connector = CensusTradeDataFetcher()
        elif source == "bea":
            connector = BEAConnector()
        elif source == "bls":
            connector = BlsDataFetcher()
        else:
            raise HTTPException(status_code=404, detail=f"Unsupported data source: {source}")
        
        # Fetch data from source
        if source == "fred":
            data = await connector.get_series_data(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
        elif source == "census":
            # Census API has different parameters
            data = await connector.fetch_data()
            # Filter for specific series if needed
            if series_id != "all":
                data = {k: v for k, v in data.items() if series_id in k.lower()}
        elif source == "bea":
            data = await connector.get_gdp_data()
        elif source == "bls":
            data = await connector.fetch_employment_data()
        else:
            data = {}
        
        # Cache the result
        if cache_manager and data:
            cache_manager.set(cache_key, data, ttl=3600)  # Cache for 1 hour
        
        return DataResponse(
            source=source,
            timestamp=datetime.utcnow(),
            data=data,
            count=len(data.get("observations", [])) if isinstance(data, dict) else len(data),
            cache_status="miss" if not cached_data else "refresh"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data from {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")


@router.get("/quality/{source}")
async def get_data_quality_metrics(source: str):
    """
    Get data quality metrics for a specific source.
    
    Returns statistics on data completeness, timeliness, accuracy,
    and other quality indicators for the specified data source.
    """
    try:
        source = source.lower()
        
        # Simulate data quality metrics (in production, these would be calculated from actual data)
        current_time = datetime.utcnow()
        
        quality_metrics = {
            "fred": {
                "completeness": 0.98,
                "timeliness": 0.95,
                "accuracy": 0.99,
                "consistency": 0.97,
                "last_quality_check": current_time.isoformat(),
                "issues": {
                    "missing_values": 0.02,
                    "delayed_updates": 0.05,
                    "revision_frequency": 0.03
                },
                "update_statistics": {
                    "expected_frequency": "daily",
                    "actual_frequency": "daily",
                    "last_update": "2024-01-01T09:00:00Z",
                    "average_delay_hours": 2.5
                }
            },
            "census": {
                "completeness": 0.92,
                "timeliness": 0.88,
                "accuracy": 0.95,
                "consistency": 0.93,
                "last_quality_check": current_time.isoformat(),
                "issues": {
                    "missing_values": 0.08,
                    "delayed_updates": 0.12,
                    "revision_frequency": 0.05
                },
                "update_statistics": {
                    "expected_frequency": "monthly",
                    "actual_frequency": "monthly",
                    "last_update": "2024-01-01T00:00:00Z",
                    "average_delay_days": 45
                }
            },
            "bea": {
                "completeness": 0.95,
                "timeliness": 0.90,
                "accuracy": 0.97,
                "consistency": 0.94,
                "last_quality_check": current_time.isoformat(),
                "issues": {
                    "missing_values": 0.05,
                    "delayed_updates": 0.10,
                    "revision_frequency": 0.08
                },
                "update_statistics": {
                    "expected_frequency": "quarterly",
                    "actual_frequency": "quarterly",
                    "last_update": "2024-01-01T00:00:00Z",
                    "average_delay_days": 30
                }
            },
            "bls": {
                "completeness": 0.94,
                "timeliness": 0.91,
                "accuracy": 0.96,
                "consistency": 0.95,
                "last_quality_check": current_time.isoformat(),
                "issues": {
                    "missing_values": 0.06,
                    "delayed_updates": 0.09,
                    "revision_frequency": 0.04
                },
                "update_statistics": {
                    "expected_frequency": "monthly",
                    "actual_frequency": "monthly",
                    "last_update": "2024-01-01T08:30:00Z",
                    "average_delay_days": 7
                }
            }
        }
        
        if source not in quality_metrics:
            raise HTTPException(status_code=404, detail=f"Quality metrics not available for source: {source}")
        
        metrics = quality_metrics[source]
        
        # Calculate overall quality score
        weights = {"completeness": 0.3, "timeliness": 0.25, "accuracy": 0.3, "consistency": 0.15}
        overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        
        return {
            "source": source,
            "overall_quality_score": round(overall_score, 3),
            "quality_grade": "A" if overall_score >= 0.95 else "B" if overall_score >= 0.90 else "C" if overall_score >= 0.80 else "D",
            "metrics": metrics,
            "recommendations": [
                "Monitor delayed updates for potential data pipeline issues" if metrics["issues"]["delayed_updates"] > 0.1 else None,
                "Investigate missing values in critical series" if metrics["issues"]["missing_values"] > 0.05 else None,
                "Review revision patterns for potential methodology changes" if metrics["issues"]["revision_frequency"] > 0.05 else None
            ],
            "timestamp": current_time.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching quality metrics for {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")


@router.get("/export/{source}")
async def export_data(
    source: str,
    format: str = Query("json", description="Export format (json, csv, excel)"),
    series: Optional[str] = Query(None, description="Comma-separated list of series IDs"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Export data in various formats for external use.
    
    Provides data export capabilities in JSON, CSV, or Excel formats
    for integration with external tools and analysis platforms.
    """
    try:
        from fastapi.responses import StreamingResponse
        import io
        import json
        import csv
        
        source = source.lower()
        
        # For this implementation, we'll simulate data export
        # In production, this would fetch real data using the appropriate connector
        
        sample_data = {
            "metadata": {
                "source": source,
                "export_date": datetime.utcnow().isoformat(),
                "series_included": series.split(",") if series else ["all"],
                "date_range": f"{start_date} to {end_date}" if start_date and end_date else "all available"
            },
            "data": [
                {
                    "date": "2024-01-01",
                    "series_id": "GDPC1",
                    "value": 21427.7,
                    "unit": "billions_chained_2012_dollars"
                },
                {
                    "date": "2024-01-01", 
                    "series_id": "UNRATE",
                    "value": 3.9,
                    "unit": "percent"
                },
                {
                    "date": "2024-01-01",
                    "series_id": "FEDFUNDS",
                    "value": 4.75,
                    "unit": "percent"
                }
            ]
        }
        
        if format.lower() == "json":
            content = json.dumps(sample_data, indent=2)
            media_type = "application/json"
            filename = f"{source}_data_export_{datetime.utcnow().strftime('%Y%m%d')}.json"
            
        elif format.lower() == "csv":
            # Convert to CSV format
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["date", "series_id", "value", "unit"])
            
            # Write data rows
            for row in sample_data["data"]:
                writer.writerow([row["date"], row["series_id"], row["value"], row["unit"]])
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"{source}_data_export_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")
        
        # Create response
        response = StreamingResponse(
            io.BytesIO(content.encode()),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
        logger.info(f"Exported {source} data in {format} format")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data from {source}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/status")
async def get_data_pipeline_status():
    """
    Get overall status of the data pipeline and all sources.
    
    Returns comprehensive status information including source health,
    update schedules, and system performance metrics.
    """
    try:
        current_time = datetime.utcnow()
        
        # Simulate pipeline status (in production, this would check actual systems)
        pipeline_status = {
            "overall_status": "healthy",
            "last_health_check": current_time.isoformat(),
            "data_sources": {
                "fred": {"status": "active", "last_update": "2024-01-01T09:00:00Z", "next_update": "2024-01-02T09:00:00Z"},
                "census": {"status": "active", "last_update": "2024-01-01T00:00:00Z", "next_update": "2024-02-01T00:00:00Z"},
                "bea": {"status": "active", "last_update": "2024-01-01T00:00:00Z", "next_update": "2024-04-01T00:00:00Z"},
                "bls": {"status": "active", "last_update": "2024-01-01T08:30:00Z", "next_update": "2024-02-01T08:30:00Z"}
            },
            "system_metrics": {
                "total_series_tracked": 850000,
                "active_sources": 4,
                "cache_hit_rate": 0.92,
                "average_api_response_time_ms": 250,
                "daily_api_calls": 15000,
                "storage_used_gb": 2.5
            },
            "recent_updates": [
                {
                    "source": "fred",
                    "series_updated": 1250,
                    "timestamp": "2024-01-01T09:15:00Z",
                    "status": "completed"
                },
                {
                    "source": "bls",
                    "series_updated": 150,
                    "timestamp": "2024-01-01T08:45:00Z", 
                    "status": "completed"
                }
            ],
            "upcoming_updates": [
                {
                    "source": "census",
                    "scheduled_time": "2024-02-01T00:00:00Z",
                    "expected_series": 25000,
                    "priority": "high"
                },
                {
                    "source": "bea",
                    "scheduled_time": "2024-04-01T00:00:00Z",
                    "expected_series": 15000,
                    "priority": "medium"
                }
            ],
            "alerts": [],
            "performance_trends": {
                "data_quality_score": 0.96,
                "uptime_percentage": 99.8,
                "error_rate": 0.002
            }
        }
        
        return pipeline_status
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")