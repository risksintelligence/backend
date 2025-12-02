#!/usr/bin/env python3
"""
Database Connection Verification Script
Explicitly tests and logs which database is being used in production.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import get_settings

def verify_database_connection():
    """Test database connection and log details."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Get settings
        settings = get_settings()
        
        logger.info("=== DATABASE CONNECTION VERIFICATION ===")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Is Production: {settings.is_production}")
        logger.info(f"Database URL: {settings.database_url}")
        
        # Determine database type from URL
        if settings.database_url.startswith("postgresql"):
            db_type = "PostgreSQL"
            logger.info("🐘 Attempting PostgreSQL connection...")
        elif settings.database_url.startswith("sqlite"):
            db_type = "SQLite"
            logger.info("💾 Using SQLite database...")
        else:
            db_type = "Unknown"
            logger.warning(f"❓ Unknown database type: {settings.database_url[:20]}...")
        
        # Test connection
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT 1 as test")).fetchone()
            logger.info(f"✅ {db_type} connection successful!")
            logger.info(f"✅ Test query result: {result}")
            
            # Check if we have RRIO tables
            if db_type == "PostgreSQL":
                tables_query = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('observations', 'page_views', 'user_events')
                """)
            else:  # SQLite
                tables_query = text("""
                    SELECT name 
                    FROM sqlite_master 
                    WHERE type='table' 
                    AND name IN ('observations', 'page_views', 'user_events')
                """)
            
            tables_result = conn.execute(tables_query).fetchall()
            existing_tables = [row[0] for row in tables_result]
            
            logger.info(f"📊 RRIO tables found: {existing_tables}")
            
            if 'observations' in existing_tables:
                # Check observation count
                obs_count = conn.execute(text("SELECT COUNT(*) FROM observations")).fetchone()
                logger.info(f"📈 Observations in database: {obs_count[0]}")
            
            if 'page_views' in existing_tables:
                # Check page views count
                pv_count = conn.execute(text("SELECT COUNT(*) FROM page_views")).fetchone()
                logger.info(f"📄 Page views tracked: {pv_count[0]}")
        
        # Environment variable check
        logger.info("\n=== ENVIRONMENT VARIABLES ===")
        postgres_dsn = os.getenv('RIS_POSTGRES_DSN')
        redis_url = os.getenv('RIS_REDIS_URL')
        
        if postgres_dsn:
            logger.info(f"✅ RIS_POSTGRES_DSN: {postgres_dsn[:30]}...")
        else:
            logger.warning("❌ RIS_POSTGRES_DSN not set - using SQLite default")
        
        if redis_url:
            logger.info(f"✅ RIS_REDIS_URL: {redis_url[:30]}...")
        else:
            logger.warning("❌ RIS_REDIS_URL not set - Redis cache disabled")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error(f"❌ This explains the 500 errors in production!")
        return False

if __name__ == "__main__":
    success = verify_database_connection()
    sys.exit(0 if success else 1)