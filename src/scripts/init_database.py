#!/usr/bin/env python3
"""
Initialize database tables for RiskX Platform.
Creates all tables from SQLAlchemy models.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import engine, AsyncSessionLocal
from src.data.models.risk_models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_tables():
    """Create all database tables."""
    try:
        logger.info("Creating database tables...")
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
        # Verify table creation
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"Created tables: {tables}")
            
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def init_database():
    """Initialize the complete database."""
    try:
        logger.info("Initializing RiskX database...")
        
        await create_tables()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())