from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.cache import redis_client
from src.cache.cache_manager import IntelligentCacheManager

# Global cache manager instance
_cache_manager: IntelligentCacheManager = None


def get_cache_manager() -> IntelligentCacheManager:
    """Dependency to get cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        # Initialize with Redis client only (no DB session for dependency)
        _cache_manager = IntelligentCacheManager(
            redis_client=redis_client,
            db_session=None  # DB session will be injected when needed
        )
    
    return _cache_manager


async def get_cache_manager_with_db(
    db: AsyncSession = Depends(get_db),
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> IntelligentCacheManager:
    """Get cache manager with database session for full three-tier caching."""
    
    # Temporarily set DB session for this request
    cache.db_session = db
    try:
        yield cache
    finally:
        cache.db_session = None