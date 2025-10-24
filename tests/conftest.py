"""
Test configuration and fixtures for the RiskX backend test suite.
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock
import tempfile
import os

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Database testing
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Application imports
from src.api.main import app
from src.core.database import get_db, Base
from src.core.dependencies import get_cache_manager
from src.cache.cache_manager import IntelligentCacheManager
from src.ml.models.network_analyzer import NetworkAnalyzer
from src.data.models.risk_models import RiskScore, RiskFactor


# Test Database Configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def override_get_db(db_session: AsyncSession):
    """Override database dependency for testing."""
    async def _override_get_db():
        yield db_session
    return _override_get_db


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager for testing."""
    mock_cache = AsyncMock(spec=IntelligentCacheManager)
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_cache.delete = AsyncMock()
    mock_cache.keys = AsyncMock(return_value=[])
    mock_cache.get_metrics = MagicMock(return_value={
        "redis_hits": 100,
        "postgres_hits": 50,
        "file_hits": 10,
        "cache_misses": 5,
        "total_requests": 165,
        "hit_rate_percent": 96.97
    })
    return mock_cache


@pytest.fixture
def override_get_cache_manager(mock_cache_manager):
    """Override cache manager dependency for testing."""
    def _override_get_cache_manager():
        return mock_cache_manager
    return _override_get_cache_manager


@pytest.fixture
def test_client(override_get_db, override_get_cache_manager):
    """Create a test client with overridden dependencies."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cache_manager] = override_get_cache_manager
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(override_get_db, override_get_cache_manager):
    """Create an async test client with overridden dependencies."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cache_manager] = override_get_cache_manager
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_risk_score_data():
    """Sample risk score data for testing."""
    return {
        "overall_score": 75.5,
        "confidence": 0.85,
        "trend": "rising",
        "economic_score": 78.2,
        "market_score": 72.1,
        "geopolitical_score": 68.9,
        "technical_score": 81.3,
        "data_sources": ["fred", "bea", "bls"],
        "calculation_method": "weighted_average"
    }


@pytest.fixture
def sample_risk_factor_data():
    """Sample risk factor data for testing."""
    return {
        "name": "GDP Growth Rate",
        "category": "economic",
        "description": "Quarterly GDP growth rate",
        "current_value": 2.1,
        "current_score": 75.0,
        "impact_level": "high",
        "weight": 1.5,
        "threshold_low": 0.0,
        "threshold_high": 4.0,
        "data_source": "fred",
        "series_id": "GDP",
        "update_frequency": "quarterly"
    }


@pytest.fixture
def sample_network_data():
    """Sample network data for testing."""
    nodes = [
        {
            "node_id": "NODE_1",
            "name": "Test Node 1",
            "node_type": "financial",
            "risk_level": 25.0,
            "weight": 1.0,
            "x_position": 0.1,
            "y_position": 0.5
        },
        {
            "node_id": "NODE_2", 
            "name": "Test Node 2",
            "node_type": "technology",
            "risk_level": 35.0,
            "weight": 1.2,
            "x_position": 0.3,
            "y_position": 0.3
        },
        {
            "node_id": "NODE_3",
            "name": "Test Node 3", 
            "node_type": "infrastructure",
            "risk_level": 45.0,
            "weight": 1.8,
            "x_position": 0.7,
            "y_position": 0.6
        }
    ]
    
    edges = [
        {
            "edge_id": "EDGE_1",
            "source_node_id": "NODE_1",
            "target_node_id": "NODE_2",
            "edge_type": "financial",
            "weight": 1.5,
            "propagation_probability": 0.7,
            "amplification_factor": 1.2,
            "direction": "directed"
        },
        {
            "edge_id": "EDGE_2",
            "source_node_id": "NODE_2",
            "target_node_id": "NODE_3",
            "edge_type": "dependency",
            "weight": 2.0,
            "propagation_probability": 0.9,
            "amplification_factor": 1.5,
            "direction": "directed"
        }
    ]
    
    return {"nodes": nodes, "edges": edges}


@pytest.fixture
def network_analyzer():
    """Create a network analyzer instance for testing."""
    return NetworkAnalyzer()


@pytest.fixture
def sample_economic_data():
    """Sample economic data for testing."""
    return {
        "series_id": "GDP",
        "source": "fred",
        "name": "Gross Domestic Product",
        "value": 27000000.0,
        "units": "millions_of_dollars",
        "frequency": "quarterly",
        "observation_date": "2024-01-01T00:00:00Z",
        "period": "2024-Q1",
        "seasonal_adjustment": "seasonally_adjusted",
        "revision_status": "final"
    }


@pytest.fixture
def sample_fred_response():
    """Sample FRED API response for testing."""
    return {
        "observations": [
            {
                "date": "2024-01-01",
                "value": "27000000.0"
            },
            {
                "date": "2024-04-01", 
                "value": "27100000.0"
            }
        ]
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.keys = AsyncMock(return_value=[])
    return mock_redis


@pytest.fixture
def temporary_cache_directory():
    """Create a temporary directory for file cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for external API testing."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_client.post = AsyncMock()
    return mock_client


@pytest.fixture
def sample_websocket_message():
    """Sample WebSocket message for testing."""
    return {
        "type": "subscribe",
        "topic": "risk_alerts"
    }


@pytest.fixture
def mock_external_apis():
    """Mock external API responses."""
    return {
        "fred": {
            "GDP": {
                "observations": [
                    {"date": "2024-01-01", "value": "27000000.0"}
                ]
            }
        },
        "bea": {
            "economic_accounts": {
                "data": {"gdp_by_industry": []}
            }
        },
        "bls": {
            "labor_statistics": {
                "series": [{"seriesID": "LNS14000000", "data": []}]
            }
        },
        "census": {
            "population_data": {
                "data": [{"population": 331000000}]
            }
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


# Test data cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    # Cleanup logic can be added here if needed


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    monkeypatch.setenv("BEA_API_KEY", "test-bea-key")
    monkeypatch.setenv("BLS_API_KEY", "test-bls-key")
    monkeypatch.setenv("CENSUS_API_KEY", "test-census-key")


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Data for benchmark testing."""
    return {
        "large_dataset": list(range(10000)),
        "network_nodes": [f"NODE_{i}" for i in range(1000)],
        "time_series": [{"date": f"2024-{i:02d}-01", "value": i * 100} for i in range(1, 13)]
    }