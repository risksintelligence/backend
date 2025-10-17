#!/usr/bin/env python3
"""
Database Initialization Script

Initializes the RiskX database with required tables, indexes, and initial data.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import get_database_engine, create_all_tables
from src.core.config import get_settings
from scripts.setup.create_tables import create_custom_tables
from scripts.setup.seed_data import seed_initial_data


async def initialize_database():
    """Initialize the database with all required components"""
    logger = logging.getLogger("init_db")
    settings = get_settings()
    
    try:
        logger.info("Starting database initialization...")
        
        # Step 1: Create database engine and basic tables
        logger.info("Creating database engine...")
        engine = get_database_engine()
        
        logger.info("Creating standard tables...")
        await create_all_tables(engine)
        
        # Step 2: Create custom tables
        logger.info("Creating custom tables...")
        await create_custom_tables(engine)
        
        # Step 3: Seed initial data
        logger.info("Seeding initial data...")
        await seed_initial_data(engine)
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


async def check_database_connection():
    """Check if database connection is working"""
    logger = logging.getLogger("init_db")
    
    try:
        engine = get_database_engine()
        
        # Test connection with a simple query
        async with engine.acquire() as conn:
            result = await conn.execute("SELECT 1 as test")
            test_result = await result.fetchone()
            
            if test_result and test_result[0] == 1:
                logger.info("Database connection successful")
                return True
            else:
                logger.error("Database connection test failed")
                return False
                
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False


async def reset_database():
    """Reset database (drop and recreate all tables)"""
    logger = logging.getLogger("init_db")
    
    try:
        logger.warning("Resetting database - all data will be lost!")
        
        engine = get_database_engine()
        
        # Drop all tables
        logger.info("Dropping existing tables...")
        async with engine.acquire() as conn:
            # Get all table names
            tables_query = """
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' AND tablename NOT LIKE 'pg_%'
            """
            result = await conn.execute(tables_query)
            tables = await result.fetchall()
            
            # Drop each table
            for table in tables:
                table_name = table[0]
                await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                logger.info(f"Dropped table: {table_name}")
        
        # Reinitialize
        await initialize_database()
        
        logger.info("Database reset completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {str(e)}")
        raise


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/db_init.log', mode='a')
        ]
    )


async def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger("init_db")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Initialize RiskX database')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset database (drop and recreate all tables)')
    parser.add_argument('--check', action='store_true',
                       help='Only check database connection')
    
    args = parser.parse_args()
    
    try:
        if args.check:
            logger.info("Checking database connection...")
            success = await check_database_connection()
            if success:
                logger.info("Database connection check passed")
                return 0
            else:
                logger.error("Database connection check failed")
                return 1
                
        elif args.reset:
            logger.warning("Database reset requested...")
            confirm = input("Are you sure you want to reset the database? (yes/no): ")
            if confirm.lower() == 'yes':
                await reset_database()
                logger.info("Database reset completed")
                return 0
            else:
                logger.info("Database reset cancelled")
                return 0
                
        else:
            logger.info("Initializing database...")
            await initialize_database()
            logger.info("Database initialization completed")
            return 0
            
    except Exception as e:
        logger.error(f"Database operation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)