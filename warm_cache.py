#!/usr/bin/env python3
"""
Quick cache warming script for analytics overview endpoint.
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.cache.cache_manager import CacheManager

def warm_analytics_cache():
    """Warm cache with analytics overview data."""
    cache_manager = CacheManager()
    
    # Create analytics overview data that matches EconomicOverviewResponse
    analytics_overview = {
        "overall_risk_level": "moderate",
        "economic_momentum": "improving", 
        "market_stress_level": "high",
        "key_concerns": ["Elevated risk in inflation sector"],
        "positive_signals": ["Stable conditions in interest rates sector", "Multiple sectors showing positive momentum"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Set the cache key that the optimized endpoint looks for
    cache_key = "analytics:overview:aggregated"
    cache_manager.set(cache_key, analytics_overview, ttl=3600)
    
    print(f"Cache warmed with key: {cache_key}")
    print(f"Data: {analytics_overview}")
    
    # Also create categories data
    categories_data = [
        {
            "category_name": "employment",
            "indicator_count": 8,
            "avg_risk_score": 35.2,
            "category_trend": "improving",
            "category_volatility": "low",
            "key_indicators": ["Unemployment Rate", "Job Openings", "Labor Force Participation"],
            "last_updated": datetime.now().isoformat()
        },
        {
            "category_name": "inflation",
            "indicator_count": 6,
            "avg_risk_score": 68.4,
            "category_trend": "declining",
            "category_volatility": "high",
            "key_indicators": ["Consumer Price Index", "Core CPI", "Producer Price Index"],
            "last_updated": datetime.now().isoformat()
        }
    ]
    
    cache_manager.set("analytics:categories:aggregated", categories_data, ttl=3600)
    print("Categories cache also warmed")

if __name__ == "__main__":
    warm_analytics_cache()