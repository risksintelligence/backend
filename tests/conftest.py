"""
Pytest Configuration and Fixtures

Provides shared fixtures and configuration for the RiskX test suite.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Test imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Provide test-specific settings"""
    settings = get_settings()
    # Override with test values
    settings.DATABASE_URL = "sqlite:///test_riskx.db"
    settings.REDIS_URL = "redis://localhost:6379/1"  # Use test database
    settings.LOG_LEVEL = "DEBUG"
    settings.ENVIRONMENT = "test"
    return settings


@pytest.fixture
async def cache_manager():
    """Provide a test cache manager"""
    cache = CacheManager()
    yield cache
    # Cleanup
    try:
        await cache.clear_all()
    except:
        pass


@pytest.fixture
def mock_fred_data():
    """Mock FRED API response data"""
    return {
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01", 
                "date": "2023-12-01",
                "value": "3.5"
            },
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": "2023-11-01", 
                "value": "3.7"
            }
        ]
    }


@pytest.fixture
def mock_bea_data():
    """Mock BEA API response data"""
    return {
        "BEAAPI": {
            "Results": {
                "Data": [
                    {
                        "TableName": "T20305",
                        "LineCode": "1",
                        "LineDescription": "GDP",
                        "TimePeriod": "2023Q4",
                        "DataValue": "27000.0"
                    }
                ]
            }
        }
    }


@pytest.fixture
def mock_trade_data():
    """Mock Census trade data"""
    return [
        ["ENDUSE", "ENDUSE_LDESC", "CTY_CODE", "CTY_NAME", "time", "GEN_VAL_MO", "GEN_VAL_YR"],
        ["40000", "Total Exports", "5700", "China", "2023-12", "15000000", "180000000"],
        ["40000", "Total Exports", "1220", "Canada", "2023-12", "25000000", "300000000"]
    ]


@pytest.fixture
def mock_weather_data():
    """Mock NOAA weather alerts data"""
    return {
        "features": [
            {
                "properties": {
                    "id": "alert123",
                    "event": "Winter Storm Warning",
                    "severity": "Moderate",
                    "certainty": "Likely",
                    "urgency": "Expected",
                    "headline": "Winter Storm Warning issued",
                    "description": "Heavy snow expected",
                    "effective": "2024-01-15T12:00:00Z",
                    "expires": "2024-01-16T12:00:00Z",
                    "areaDesc": "New York, NY"
                }
            }
        ]
    }


@pytest.fixture
def mock_cyber_data():
    """Mock CISA KEV data"""
    return {
        "vulnerabilities": [
            {
                "cveID": "CVE-2023-1234",
                "vendorProject": "Microsoft",
                "product": "Windows",
                "vulnerabilityName": "Remote Code Execution",
                "dateAdded": "2023-12-01",
                "shortDescription": "Critical RCE vulnerability",
                "requiredAction": "Apply patch",
                "dueDate": "2024-01-01"
            }
        ]
    }


@pytest.fixture
def sample_risk_data():
    """Sample risk assessment data"""
    return {
        "economic_indicators": {
            "gdp_growth": 2.1,
            "unemployment_rate": 3.8,
            "inflation_rate": 2.4,
            "interest_rate": 5.25
        },
        "trade_metrics": {
            "trade_balance": -75000000000,
            "import_dependency": 0.15,
            "export_concentration": 0.25
        },
        "financial_health": {
            "bank_stress_index": 0.3,
            "credit_default_rate": 0.02,
            "market_volatility": 0.18
        },
        "disruption_indicators": {
            "weather_severity": "moderate",
            "cyber_threat_level": "medium",
            "supply_chain_risk": "low"
        }
    }


@pytest.fixture
def mock_model_predictions():
    """Mock ML model prediction results"""
    return {
        "risk_score": 0.65,
        "confidence": 0.87,
        "feature_importance": {
            "unemployment_rate": 0.15,
            "trade_balance": 0.12,
            "inflation_rate": 0.10,
            "bank_stress": 0.08,
            "weather_events": 0.05
        },
        "explanation": {
            "primary_drivers": ["unemployment_rate", "trade_balance"],
            "risk_level": "moderate",
            "recommendation": "Monitor closely"
        }
    }


@pytest.fixture
def mock_database_engine():
    """Mock database engine for testing"""
    engine = Mock()
    connection = AsyncMock()
    engine.acquire.return_value.__aenter__.return_value = connection
    engine.acquire.return_value.__aexit__.return_value = None
    return engine


@pytest.fixture
def temporary_file():
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('{"test": "data"}')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def mock_http_session():
    """Mock HTTP session for API testing"""
    session = AsyncMock()
    
    # Configure default successful response
    response = AsyncMock()
    response.status = 200
    response.json.return_value = {"status": "success", "data": []}
    response.text.return_value = "OK"
    
    session.get.return_value.__aenter__.return_value = response
    session.post.return_value.__aenter__.return_value = response
    
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for cache testing"""
    redis_client = AsyncMock()
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    redis_client.delete.return_value = 1
    redis_client.exists.return_value = False
    redis_client.expire.return_value = True
    return redis_client


@pytest.fixture
def test_data_files():
    """Create test data files"""
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Sample CSV data
    csv_file = test_data_dir / "sample_economic_data.csv"
    csv_file.write_text("""date,series_id,value
2023-12-01,GDPC1,21000.5
2023-11-01,GDPC1,20950.2
2023-10-01,GDPC1,20900.8""")
    
    # Sample JSON data
    json_file = test_data_dir / "sample_api_response.json"
    json_file.write_text('{"status": "success", "data": {"test": true}}')
    
    yield {
        "csv_file": csv_file,
        "json_file": json_file,
        "data_dir": test_data_dir
    }
    
    # Cleanup
    try:
        csv_file.unlink()
        json_file.unlink()
        test_data_dir.rmdir()
    except:
        pass


@pytest.fixture
def mock_notification_manager():
    """Mock notification manager"""
    manager = Mock()
    manager.notify = AsyncMock(return_value={"email": True, "webhook": True})
    manager.notify_pipeline_start = AsyncMock(return_value={"status": "sent"})
    manager.notify_pipeline_success = AsyncMock(return_value={"status": "sent"})
    manager.notify_pipeline_failure = AsyncMock(return_value={"status": "sent"})
    return manager


@pytest.fixture
def sample_validation_config():
    """Sample data validation configuration"""
    return {
        "source_name": "test_source",
        "schema_name": "fred_data",
        "validate_schema": True,
        "validate_quality": True,
        "completeness_threshold": 80,
        "max_age_days": 7,
        "date_column": "date",
        "outlier_threshold": 5
    }


class AsyncContextManager:
    """Helper class for async context manager testing"""
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers in tests"""
    return AsyncContextManager


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external_api: mark test as requiring external API"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Mark external API tests
        if "api" in str(item.fspath) and "mock" not in item.name.lower():
            item.add_marker(pytest.mark.external_api)