import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import logging
from src.core.config import get_settings

logger = logging.getLogger(__name__)

# Get database URL from settings configuration
settings = get_settings()
DATABASE_URL = settings.database_url
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Create async engine
engine = None
if DATABASE_URL:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,  # SQL logging disabled for production
        pool_pre_ping=True,
        pool_recycle=300,
    )

# Create session factory
AsyncSessionLocal = None
if engine:
    AsyncSessionLocal = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )

# Base class for models
Base = declarative_base()

async def get_db():
    """Dependency to get database session."""
    if not AsyncSessionLocal:
        raise Exception("Database not configured")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()

async def check_database_connection():
    """Check if database connection is working."""
    if not engine:
        return {"status": "not_configured", "error": "DATABASE_URL not set"}
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as connection_check"))
            row = result.fetchone()
            if row and row[0] == 1:
                return {"status": "connected", "query_result": "success"}
            else:
                return {"status": "error", "query_result": "failed"}
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return {"status": "error", "message": str(e)}

async def get_database_info():
    """Get database connection information."""
    if not engine:
        return {"status": "not_configured"}
    
    try:
        async with engine.begin() as conn:
            # Get database version
            version_result = await conn.execute(text("SELECT version()"))
            version = version_result.fetchone()[0] if version_result else "unknown"
            
            # Get current database name
            db_result = await conn.execute(text("SELECT current_database()"))
            database_name = db_result.fetchone()[0] if db_result else "unknown"
            
            # Get connection count
            conn_result = await conn.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            ))
            active_connections = conn_result.fetchone()[0] if conn_result else 0
            
            return {
                "status": "connected",
                "database_name": database_name,
                "version": version.split()[0] if version else "unknown",
                "active_connections": active_connections,
                "engine_pool_size": engine.pool.size(),
                "engine_checked_out": engine.pool.checkedout()
            }
    except Exception as e:
        logger.error(f"Database info error: {e}")
        return {"status": "error", "message": str(e)}