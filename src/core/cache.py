import redis.asyncio as redis
import json
import os
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Redis client
redis_client = None
REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL:
    redis_client = redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )

async def check_redis_connection():
    """Check if Redis connection is working."""
    if not redis_client:
        return {"status": "not_configured", "error": "REDIS_URL not set"}
    
    try:
        # Test basic Redis operations
        test_key = "health_check_test"
        test_value = "test_value"
        
        # Set a test value
        await redis_client.set(test_key, test_value, ex=10)  # Expire in 10 seconds
        
        # Get the test value
        retrieved_value = await redis_client.get(test_key)
        
        # Clean up
        await redis_client.delete(test_key)
        
        if retrieved_value == test_value:
            return {"status": "connected", "test_operations": "success"}
        else:
            return {"status": "error", "test_operations": "failed"}
            
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        return {"status": "error", "message": str(e)}

async def get_redis_info():
    """Get Redis server information."""
    if not redis_client:
        return {"status": "not_configured"}
    
    try:
        # Get Redis server info
        info = await redis_client.info()
        
        # Get database stats
        db_info = await redis_client.info("keyspace")
        
        return {
            "status": "connected",
            "redis_version": info.get("redis_version", "unknown"),
            "memory_used": info.get("used_memory_human", "unknown"),
            "memory_peak": info.get("used_memory_peak_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "database_info": db_info,
            "uptime_seconds": info.get("uptime_in_seconds", 0)
        }
    except Exception as e:
        logger.error(f"Redis info error: {e}")
        return {"status": "error", "message": str(e)}

async def test_cache_operations():
    """Test comprehensive cache operations."""
    if not redis_client:
        return {"status": "not_configured"}
    
    try:
        test_results = {}
        
        # Test 1: Basic set/get
        await redis_client.set("test:basic", "basic_value", ex=60)
        basic_value = await redis_client.get("test:basic")
        test_results["basic_operations"] = basic_value == "basic_value"
        
        # Test 2: JSON data
        test_data = {
            "risk_score": 75.5,
            "factors": ["economic", "financial"],
            "timestamp": datetime.utcnow().isoformat()
        }
        await redis_client.set("test:json", json.dumps(test_data), ex=60)
        json_value = await redis_client.get("test:json")
        retrieved_data = json.loads(json_value) if json_value else None
        test_results["json_operations"] = retrieved_data == test_data
        
        # Test 3: Expiration
        await redis_client.set("test:expire", "expire_value", ex=1)
        ttl = await redis_client.ttl("test:expire")
        test_results["expiration_set"] = ttl > 0
        
        # Test 4: Key operations
        test_keys = ["test:key1", "test:key2", "test:key3"]
        for i, key in enumerate(test_keys):
            await redis_client.set(key, f"value_{i}", ex=60)
        
        existing_keys = await redis_client.keys("test:key*")
        test_results["multiple_keys"] = len(existing_keys) == 3
        
        # Test 5: Hash operations
        hash_data = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        await redis_client.hset("test:hash", mapping=hash_data)
        retrieved_hash = await redis_client.hgetall("test:hash")
        test_results["hash_operations"] = retrieved_hash == hash_data
        
        # Clean up test keys
        cleanup_keys = ["test:basic", "test:json", "test:expire", "test:hash"] + test_keys
        await redis_client.delete(*cleanup_keys)
        
        return {
            "status": "success",
            "test_results": test_results,
            "all_tests_passed": all(test_results.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache operations test error: {e}")
        return {"status": "error", "message": str(e)}

class BasicCacheManager:
    """Basic cache manager for testing purposes."""
    
    def __init__(self):
        self.redis_client = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return data.get("value")
            return None
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value in cache."""
        if not self.redis_client:
            return
        
        try:
            cache_data = {
                "value": value,
                "cached_at": datetime.utcnow().isoformat()
            }
            await self.redis_client.set(key, json.dumps(cache_data), ex=ttl_seconds)
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
    
    async def keys(self, pattern: str = "*"):
        """Get keys matching pattern."""
        if not self.redis_client:
            return []
        
        try:
            return await self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Cache keys error for {pattern}: {e}")
            return []