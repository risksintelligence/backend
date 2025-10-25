from typing import Optional, Any, List
import asyncio
import json
import os
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import aiofiles
import logging

logger = logging.getLogger(__name__)


class IntelligentCacheManager:
    """
    Three-tier cache with automatic fallback.
    Always returns data instantly, never blocks on API calls.
    """
    
    def __init__(self, redis_client, db_session: Optional[AsyncSession] = None):
        self.redis_client = redis_client
        self.db_session = db_session
        self.cache_dir = "data/cache"
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Performance metrics
        self.metrics = {
            "redis_hits": 0,
            "postgres_hits": 0,
            "file_hits": 0,
            "cache_misses": 0
        }
    
    async def get(
        self, 
        cache_key: str, 
        max_age_seconds: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get data from cache with automatic tier fallback.
        Returns None only if data has never been cached.
        """
        
        # L1: Check Redis (fastest)
        data = await self._get_from_redis(cache_key, max_age_seconds)
        if data is not None:
            self.metrics["redis_hits"] += 1
            logger.debug(f"Redis hit: {cache_key}")
            return data
        
        # L2: Check PostgreSQL
        if self.db_session:
            data = await self._get_from_postgres(cache_key, max_age_seconds)
            if data is not None:
                self.metrics["postgres_hits"] += 1
                logger.debug(f"PostgreSQL hit: {cache_key}")
                # Warm up Redis
                await self._set_to_redis(cache_key, data)
                return data
        
        # L3: Check File System
        data = await self._get_from_file(cache_key, max_age_seconds)
        if data is not None:
            self.metrics["file_hits"] += 1
            logger.debug(f"File hit: {cache_key}")
            # Warm up Redis and PostgreSQL
            await self._set_to_redis(cache_key, data)
            if self.db_session:
                await self._set_to_postgres(cache_key, data)
            return data
        
        # Cache miss - data never cached before
        self.metrics["cache_misses"] += 1
        logger.warning(f"Cache miss: {cache_key}")
        return None
    
    async def set(
        self, 
        cache_key: str, 
        data: Any,
        ttl_seconds: int = 3600
    ) -> None:
        """
        Set data in all cache tiers simultaneously.
        Fire-and-forget for speed.
        """
        
        # Set in all tiers asynchronously
        tasks = [
            self._set_to_redis(cache_key, data, ttl_seconds),
            self._set_to_file(cache_key, data)
        ]
        
        if self.db_session:
            tasks.append(self._set_to_postgres(cache_key, data))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Cached across all tiers: {cache_key}")
    
    async def delete(self, cache_key: str) -> None:
        """Delete from all cache tiers."""
        tasks = [
            self._delete_from_redis(cache_key),
            self._delete_from_file(cache_key)
        ]
        
        if self.db_session:
            tasks.append(self._delete_from_postgres(cache_key))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Deleted from all tiers: {cache_key}")
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern from Redis."""
        if not self.redis_client:
            return []
        
        try:
            return await self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis keys error for {pattern}: {e}")
            return []
    
    # === Redis Layer (L1) ===
    
    async def _get_from_redis(
        self, 
        cache_key: str,
        max_age_seconds: Optional[int]
    ) -> Optional[Any]:
        if not self.redis_client:
            return None
            
        try:
            cached = await self.redis_client.get(cache_key)
            if not cached:
                return None
            
            data = json.loads(cached)
            
            # Check age if max_age specified
            if max_age_seconds and data.get("cached_at"):
                try:
                    cached_at = datetime.fromisoformat(data.get("cached_at"))
                    age = (datetime.utcnow() - cached_at).total_seconds()
                    if age > max_age_seconds:
                        return None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid cached_at timestamp for {cache_key}: {e}")
                    return None
            
            return data.get("value")
        
        except Exception as e:
            logger.error(f"Redis error for {cache_key}: {e}")
            return None
    
    async def _set_to_redis(
        self, 
        cache_key: str, 
        data: Any,
        ttl_seconds: int = 3600
    ) -> None:
        if not self.redis_client:
            return
            
        try:
            cache_data = {
                "value": data,
                "cached_at": datetime.utcnow().isoformat()
            }
            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(cache_data, default=str, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Redis set error for {cache_key}: {e}")
    
    async def _delete_from_redis(self, cache_key: str) -> None:
        if not self.redis_client:
            return
            
        try:
            await self.redis_client.delete(cache_key)
        except Exception as e:
            logger.error(f"Redis delete error for {cache_key}: {e}")
    
    # === PostgreSQL Layer (L2) ===
    
    async def _get_from_postgres(
        self, 
        cache_key: str,
        max_age_seconds: Optional[int]
    ) -> Optional[Any]:
        if not self.db_session:
            return None
            
        try:
            result = await self.db_session.execute(
                text("""
                SELECT data, cached_at 
                FROM cache_entries 
                WHERE cache_key = :cache_key
                """),
                {"cache_key": cache_key}
            )
            row = result.first()
            
            if not row:
                return None
            
            data, cached_at = row
            
            # Check age
            if max_age_seconds:
                age = (datetime.utcnow() - cached_at).total_seconds()
                if age > max_age_seconds:
                    return None
            
            return json.loads(data)
        
        except Exception as e:
            logger.error(f"PostgreSQL error for {cache_key}: {e}")
            return None
    
    async def _set_to_postgres(
        self, 
        cache_key: str, 
        data: Any
    ) -> None:
        if not self.db_session:
            return
            
        try:
            await self.db_session.execute(
                text("""
                INSERT INTO cache_entries (cache_key, data, cached_at)
                VALUES (:cache_key, :data, NOW())
                ON CONFLICT (cache_key) 
                DO UPDATE SET data = EXCLUDED.data, cached_at = NOW()
                """),
                {
                    "cache_key": cache_key,
                    "data": json.dumps(data, default=str)
                }
            )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"PostgreSQL set error for {cache_key}: {e}")
            await self.db_session.rollback()
    
    async def _delete_from_postgres(self, cache_key: str) -> None:
        if not self.db_session:
            return
            
        try:
            await self.db_session.execute(
                text("DELETE FROM cache_entries WHERE cache_key = :cache_key"),
                {"cache_key": cache_key}
            )
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"PostgreSQL delete error for {cache_key}: {e}")
            await self.db_session.rollback()
    
    # === File System Layer (L3) ===
    
    async def _get_from_file(
        self, 
        cache_key: str,
        max_age_seconds: Optional[int]
    ) -> Optional[Any]:
        try:
            # Sanitize cache key for filename
            safe_key = cache_key.replace(":", "_").replace("/", "_")
            file_path = f"{self.cache_dir}/{safe_key}.json"
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                cache_data = json.loads(content)
            
            # Check age
            if max_age_seconds and cache_data.get("cached_at"):
                try:
                    cached_at = datetime.fromisoformat(cache_data.get("cached_at"))
                    age = (datetime.utcnow() - cached_at).total_seconds()
                    if age > max_age_seconds:
                        return None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid cached_at timestamp in file for {cache_key}: {e}")
                    return None
            
            return cache_data.get("value")
        
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"File cache error for {cache_key}: {e}")
            return None
    
    async def _set_to_file(
        self, 
        cache_key: str, 
        data: Any
    ) -> None:
        try:
            # Sanitize cache key for filename
            safe_key = cache_key.replace(":", "_").replace("/", "_")
            file_path = f"{self.cache_dir}/{safe_key}.json"
            
            cache_data = {
                "value": data,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(cache_data, default=str, ensure_ascii=False, indent=2))
        
        except Exception as e:
            logger.error(f"File cache set error for {cache_key}: {e}")
    
    async def _delete_from_file(self, cache_key: str) -> None:
        try:
            safe_key = cache_key.replace(":", "_").replace("/", "_")
            file_path = f"{self.cache_dir}/{safe_key}.json"
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"File cache delete error for {cache_key}: {e}")
    
    def get_metrics(self) -> dict:
        """Get cache performance metrics."""
        total_hits = sum([
            self.metrics["redis_hits"],
            self.metrics["postgres_hits"],
            self.metrics["file_hits"]
        ])
        total_requests = total_hits + self.metrics["cache_misses"]
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        }