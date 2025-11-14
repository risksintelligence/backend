"""Database connection management."""
import asyncpg
from typing import Optional
from .config import get_settings

_pool: Optional[asyncpg.Pool] = None

async def get_database_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(settings.RIS_POSTGRES_DSN)
    return _pool

async def close_database_pool():
    """Close database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None