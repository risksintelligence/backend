#!/usr/bin/env python3
"""
Seed cache with sample economic data to provide fallback when APIs are unavailable.
This ensures the system has data to display even when external APIs fail.
"""
import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cache.cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_economic_data():
    """Create realistic sample economic data for fallback."""
    return {
        # Interest rates (%)
        "FEDFUNDS": {"value": 5.25, "date": "2024-01-01", "description": "Federal Funds Rate"},
        "DGS10": {"value": 4.35, "date": "2024-01-01", "description": "10-Year Treasury Rate"},
        "DGS2": {"value": 4.88, "date": "2024-01-01", "description": "2-Year Treasury Rate"},
        "TB3MS": {"value": 5.12, "date": "2024-01-01", "description": "3-Month Treasury"},
        
        # Inflation (%)
        "CPIAUCSL": {"value": 3.2, "date": "2024-01-01", "description": "Consumer Price Index"},
        "CPILFESL": {"value": 3.9, "date": "2024-01-01", "description": "Core CPI"},
        "PCEPI": {"value": 2.8, "date": "2024-01-01", "description": "PCE Price Index"},
        
        # Employment (%)
        "UNRATE": {"value": 3.8, "date": "2024-01-01", "description": "Unemployment Rate"},
        "CIVPART": {"value": 62.5, "date": "2024-01-01", "description": "Labor Force Participation"},
        "EMRATIO": {"value": 60.2, "date": "2024-01-01", "description": "Employment-Population Ratio"},
        
        # GDP (Trillions $)
        "GDP": {"value": 27.36, "date": "2024-01-01", "description": "Gross Domestic Product"},
        "GDPC1": {"value": 22.11, "date": "2024-01-01", "description": "Real GDP"},
        "GDPPOT": {"value": 22.75, "date": "2024-01-01", "description": "Potential GDP"},
        
        # Financial Conditions
        "NFCI": {"value": -0.15, "date": "2024-01-01", "description": "Financial Conditions Index"},
        "ANFCI": {"value": -0.08, "date": "2024-01-01", "description": "Adjusted Financial Conditions"},
        "STLFSI4": {"value": 0.23, "date": "2024-01-01", "description": "Financial Stress Index"},
        
        # Credit and Banking
        "TOTALSL": {"value": 4850.2, "date": "2024-01-01", "description": "Total Consumer Credit"},
        "DRBLACBS": {"value": 2.1, "date": "2024-01-01", "description": "Bank Credit Standards"},
        "DRCCLACBS": {"value": 1.8, "date": "2024-01-01", "description": "Consumer Credit Standards"},
        "MORTGAGE30US": {"value": 6.95, "date": "2024-01-01", "description": "30-Year Mortgage Rate"}
    }


def create_sample_bea_data():
    """Create sample BEA data."""
    return {
        "NIPA_T10101": {"value": 27360.0, "date": "2024-Q1", "description": "GDP Current Dollars"},
        "NIPA_T20100": {"value": 18250.0, "date": "2024-Q1", "description": "Personal Consumption"},
        "ITA_U70205S": {"value": -887.5, "date": "2024-01", "description": "Trade Balance"}
    }


def create_sample_risk_metrics():
    """Create sample risk assessment metrics."""
    return {
        "comprehensive": {
            "overall_score": 65.8,
            "confidence": 0.82,
            "category_scores": {
                "monetary_policy": 58.5,
                "inflation": 72.3,
                "employment": 45.2,
                "economic_growth": 68.9,
                "financial_conditions": 70.1,
                "credit_risk": 62.7
            },
            "factors": [
                {
                    "name": "Federal Funds Rate",
                    "value": 5.25,
                    "weight": 0.15,
                    "contribution": 8.8,
                    "trend": "stable"
                },
                {
                    "name": "Unemployment Rate", 
                    "value": 3.8,
                    "weight": 0.12,
                    "contribution": 4.6,
                    "trend": "improving"
                },
                {
                    "name": "Inflation Rate",
                    "value": 3.2,
                    "weight": 0.14,
                    "contribution": 10.1,
                    "trend": "declining"
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "methodology": "v1.0"
        }
    }


def create_analytics_data():
    """Create sample analytics aggregation data."""
    return {
        "total_indicators": 23,
        "available_indicators": 15,
        "data_freshness": 0.65,
        "last_update": datetime.utcnow().isoformat(),
        "categories": {
            "monetary_policy": {
                "indicators": 4,
                "available": 3,
                "avg_value": 4.85,
                "status": "healthy"
            },
            "inflation": {
                "indicators": 3,
                "available": 3,
                "avg_value": 3.3,
                "status": "healthy"
            },
            "employment": {
                "indicators": 3,
                "available": 2,
                "avg_value": 42.2,
                "status": "degraded"
            },
            "economic_growth": {
                "indicators": 4,
                "available": 3,
                "avg_value": 24.1,
                "status": "healthy"
            },
            "financial_conditions": {
                "indicators": 3,
                "available": 2,
                "avg_value": 0.05,
                "status": "degraded"
            },
            "credit_risk": {
                "indicators": 6,
                "available": 2,
                "avg_value": 1925.0,
                "status": "degraded"
            }
        },
        "system_status": {
            "fred_api": "rate_limited",
            "bea_api": "error",
            "cache_utilization": 0.85,
            "fallback_mode": True
        }
    }


def seed_cache():
    """Seed cache with sample data."""
    logger.info("Starting cache seeding process...")
    
    cache_manager = CacheManager()
    
    # Seed FRED data
    logger.info("Seeding FRED economic indicators...")
    fred_data = create_sample_economic_data()
    for series_id, data in fred_data.items():
        key = f"fred:{series_id}:{hash(series_id) % 100000000:08x}"
        cache_manager.set(key, data, ttl=7*24*3600)  # 1 week TTL
    
    # Seed BEA data
    logger.info("Seeding BEA economic data...")
    bea_data = create_sample_bea_data()
    for series_id, data in bea_data.items():
        key = f"bea:{series_id}:{hash(series_id) % 100000000:08x}"
        cache_manager.set(key, data, ttl=7*24*3600)
    
    # Seed risk metrics
    logger.info("Seeding risk assessment data...")
    risk_data = create_sample_risk_metrics()
    for metric_type, data in risk_data.items():
        key = f"risk_score:{metric_type}"
        cache_manager.set(key, data, ttl=7*24*3600)
    
    # Seed analytics aggregation
    logger.info("Seeding analytics data...")
    analytics_data = create_analytics_data()
    key = "analytics:aggregation"
    cache_manager.set(key, analytics_data, ttl=7*24*3600)
    
    # Register as fallback data
    logger.info("Registering fallback data...")
    cache_manager.register_fallback_data("fred", fred_data)
    cache_manager.register_fallback_data("bea", bea_data)
    cache_manager.register_fallback_data("risk_score", risk_data)
    cache_manager.register_fallback_data("analytics", analytics_data)
    
    logger.info("Cache seeding completed successfully!")
    
    # Print cache statistics
    stats = cache_manager.get_cache_stats()
    logger.info(f"Cache statistics: {stats}")
    
    health = cache_manager.health_check()
    logger.info(f"Cache health: {health}")


if __name__ == "__main__":
    seed_cache()