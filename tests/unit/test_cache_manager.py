"""
Unit tests for the IntelligentCacheManager.
"""
import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.cache.cache_manager import IntelligentCacheManager


@pytest.mark.unit
class TestIntelligentCacheManager:
    """Test cases for IntelligentCacheManager."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.keys = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        db_mock = AsyncMock()
        db_mock.execute = AsyncMock()
        db_mock.commit = AsyncMock()
        db_mock.rollback = AsyncMock()
        return db_mock
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def cache_manager(self, mock_redis, mock_db_session, temp_cache_dir):
        """Create cache manager with mocked dependencies."""
        with patch('src.cache.cache_manager.os.makedirs'):
            manager = IntelligentCacheManager(mock_redis, mock_db_session)
            manager.cache_dir = temp_cache_dir
            return manager
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, mock_redis, temp_cache_dir):
        """Test cache manager initialization."""
        with patch('src.cache.cache_manager.os.makedirs') as mock_makedirs:
            manager = IntelligentCacheManager(mock_redis)
            
            assert manager.redis_client == mock_redis
            assert manager.db_session is None
            assert manager.cache_dir == "data/cache"
            assert "redis_hits" in manager.metrics
            mock_makedirs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_from_redis_hit(self, cache_manager, mock_redis):
        """Test successful cache hit from Redis."""
        # Setup
        cache_key = "test:key"
        cached_data = {
            "value": {"test": "data"},
            "cached_at": datetime.utcnow().isoformat()
        }
        mock_redis.get.return_value = json.dumps(cached_data)
        
        # Execute
        result = await cache_manager.get(cache_key)
        
        # Verify
        assert result == {"test": "data"}
        assert cache_manager.metrics["redis_hits"] == 1
        mock_redis.get.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_get_from_redis_miss_postgres_hit(self, cache_manager, mock_redis, mock_db_session):
        """Test Redis miss but PostgreSQL hit."""
        # Setup
        cache_key = "test:key"
        mock_redis.get.return_value = None
        
        # Mock PostgreSQL response
        mock_result = MagicMock()
        mock_result.first.return_value = (
            json.dumps({"test": "data"}),
            datetime.utcnow()
        )
        mock_db_session.execute.return_value = mock_result
        
        # Execute
        result = await cache_manager.get(cache_key)
        
        # Verify
        assert result == {"test": "data"}
        assert cache_manager.metrics["postgres_hits"] == 1
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_with_max_age_expired(self, cache_manager, mock_redis):
        """Test cache get with expired max_age."""
        # Setup
        cache_key = "test:key"
        old_time = datetime(2020, 1, 1).isoformat()
        cached_data = {
            "value": {"test": "data"},
            "cached_at": old_time
        }
        mock_redis.get.return_value = json.dumps(cached_data)
        
        # Execute
        result = await cache_manager.get(cache_key, max_age_seconds=60)
        
        # Verify
        assert result is None  # Should be expired
    
    @pytest.mark.asyncio
    async def test_set_to_all_tiers(self, cache_manager, mock_redis, mock_db_session):
        """Test setting data to all cache tiers."""
        # Setup
        cache_key = "test:key"
        test_data = {"test": "data"}
        ttl_seconds = 3600
        
        # Execute
        await cache_manager.set(cache_key, test_data, ttl_seconds)
        
        # Verify
        mock_redis.setex.assert_called_once()
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_cache_operations(self, cache_manager, temp_cache_dir):
        """Test file cache set and get operations."""
        # Setup
        cache_key = "test:file:key"
        test_data = {"test": "file_data"}
        
        # Set data to file cache
        await cache_manager._set_to_file(cache_key, test_data)
        
        # Verify file was created
        safe_key = cache_key.replace(":", "_").replace("/", "_")
        file_path = os.path.join(temp_cache_dir, f"{safe_key}.json")
        assert os.path.exists(file_path)
        
        # Get data from file cache
        result = await cache_manager._get_from_file(cache_key, max_age_seconds=None)
        
        # Verify
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_delete_from_all_tiers(self, cache_manager, mock_redis, mock_db_session):
        """Test deleting from all cache tiers."""
        # Setup
        cache_key = "test:key"
        
        # Execute
        await cache_manager.delete(cache_key)
        
        # Verify
        mock_redis.delete.assert_called_once_with(cache_key)
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_keys_operation(self, cache_manager, mock_redis):
        """Test keys operation."""
        # Setup
        pattern = "test:*"
        expected_keys = ["test:key1", "test:key2"]
        mock_redis.keys.return_value = expected_keys
        
        # Execute
        result = await cache_manager.keys(pattern)
        
        # Verify
        assert result == expected_keys
        mock_redis.keys.assert_called_once_with(pattern)
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, cache_manager):
        """Test metrics calculation."""
        # Setup
        cache_manager.metrics = {
            "redis_hits": 100,
            "postgres_hits": 50,
            "file_hits": 10,
            "cache_misses": 5
        }
        
        # Execute
        metrics = cache_manager.get_metrics()
        
        # Verify
        assert metrics["total_requests"] == 165
        assert metrics["hit_rate_percent"] == 96.97
        assert "redis_hits" in metrics
    
    @pytest.mark.asyncio
    async def test_cache_miss_all_tiers(self, cache_manager, mock_redis, mock_db_session):
        """Test cache miss across all tiers."""
        # Setup
        cache_key = "test:missing:key"
        
        # Mock all tiers returning None
        mock_redis.get.return_value = None
        
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        # Execute
        result = await cache_manager.get(cache_key)
        
        # Verify
        assert result is None
        assert cache_manager.metrics["cache_misses"] == 1
    
    @pytest.mark.asyncio
    async def test_redis_error_handling(self, cache_manager, mock_redis):
        """Test Redis error handling."""
        # Setup
        cache_key = "test:key"
        mock_redis.get.side_effect = Exception("Redis connection error")
        
        # Execute
        result = await cache_manager._get_from_redis(cache_key, None)
        
        # Verify
        assert result is None  # Should handle error gracefully
    
    @pytest.mark.asyncio
    async def test_postgres_error_handling(self, cache_manager, mock_db_session):
        """Test PostgreSQL error handling."""
        # Setup
        cache_key = "test:key"
        test_data = {"test": "data"}
        mock_db_session.execute.side_effect = Exception("DB connection error")
        
        # Execute - should not raise exception
        await cache_manager._set_to_postgres(cache_key, test_data)
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_cache_error_handling(self, cache_manager):
        """Test file cache error handling."""
        # Setup
        cache_key = "test:invalid:key"
        
        # Execute with invalid directory
        cache_manager.cache_dir = "/invalid/path"
        await cache_manager._set_to_file(cache_key, {"test": "data"})
        
        # Should not raise exception
        result = await cache_manager._get_from_file(cache_key, None)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_warming_scenario(self, cache_manager, mock_redis, mock_db_session):
        """Test cache warming scenario from file to Redis."""
        # Setup
        cache_key = "test:warm:key"
        test_data = {"test": "warm_data"}
        
        # Mock Redis miss, PostgreSQL miss, file hit
        mock_redis.get.return_value = None
        
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        # Set up file cache
        await cache_manager._set_to_file(cache_key, test_data)
        
        # Execute
        result = await cache_manager.get(cache_key)
        
        # Verify
        assert result == test_data
        assert cache_manager.metrics["file_hits"] == 1
        # Should warm up Redis and PostgreSQL
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, cache_manager, mock_redis):
        """Test concurrent cache operations."""
        # Setup
        cache_keys = [f"test:concurrent:{i}" for i in range(10)]
        test_data = {"test": "concurrent_data"}
        
        # Execute concurrent operations
        tasks = []
        for key in cache_keys:
            tasks.append(cache_manager.set(key, test_data))
        
        await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert mock_redis.setex.call_count == 10
    
    @pytest.mark.asyncio
    async def test_cache_performance_metrics(self, cache_manager):
        """Test cache performance metrics tracking."""
        # Setup initial state
        assert cache_manager.metrics["redis_hits"] == 0
        
        # Simulate cache operations
        cache_manager.metrics["redis_hits"] += 5
        cache_manager.metrics["postgres_hits"] += 3
        cache_manager.metrics["file_hits"] += 2
        cache_manager.metrics["cache_misses"] += 1
        
        # Get metrics
        metrics = cache_manager.get_metrics()
        
        # Verify calculations
        assert metrics["total_requests"] == 11
        assert metrics["hit_rate_percent"] == 90.91
        assert metrics["redis_hits"] == 5