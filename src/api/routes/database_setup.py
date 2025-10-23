from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime
import logging

from src.core.database import get_db, engine
from src.data.models.risk_models import Base
from src.scripts.populate_sample_data import populate_all_sample_data

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/database", tags=["database_setup"])


@router.post("/init")
async def initialize_database():
    """Initialize database tables."""
    try:
        logger.info("Initializing database tables...")
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
        return {
            "status": "success",
            "message": "Database tables initialized successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")


@router.post("/populate")
async def populate_sample_data():
    """Populate database with sample data for testing."""
    try:
        logger.info("Populating sample data...")
        
        await populate_all_sample_data()
        
        return {
            "status": "success",
            "message": "Sample data populated successfully",
            "data_created": {
                "risk_scores": "30 days of historical data",
                "risk_factors": "8 economic and market factors",
                "alerts": "4 sample alerts"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sample data population error: {e}")
        raise HTTPException(status_code=500, detail=f"Sample data population failed: {str(e)}")


@router.post("/setup")
async def setup_complete_database():
    """Initialize database and populate with sample data."""
    try:
        logger.info("Setting up complete database...")
        
        # Step 1: Initialize tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables initialized")
        
        # Step 2: Populate sample data  
        await populate_all_sample_data()
        
        logger.info("Sample data populated")
        
        return {
            "status": "success",
            "message": "Database setup completed successfully",
            "steps_completed": [
                "Database tables created",
                "Sample risk scores populated (30 days)",
                "Sample risk factors created (8 factors)", 
                "Sample alerts generated (4 alerts)"
            ],
            "ready_for_testing": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Database setup failed: {str(e)}")


@router.delete("/reset")
async def reset_database():
    """Reset database by dropping and recreating all tables."""
    try:
        logger.info("Resetting database...")
        
        # Drop all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("All tables dropped")
            
            # Recreate tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("All tables recreated")
        
        return {
            "status": "success",
            "message": "Database reset completed successfully",
            "action": "All tables dropped and recreated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")


@router.get("/schema")
async def get_database_schema(db: AsyncSession = Depends(get_db)):
    """Get database schema information."""
    try:
        # Get table information
        result = await db.execute(text("""
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """))
        
        schema_info = {}
        for row in result.fetchall():
            table_name = row[0]
            if table_name not in schema_info:
                schema_info[table_name] = []
            
            schema_info[table_name].append({
                "column": row[1],
                "type": row[2],
                "nullable": row[3] == 'YES',
                "default": row[4]
            })
        
        return {
            "status": "success",
            "schema": schema_info,
            "table_count": len(schema_info),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Schema retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")


@router.get("/data/summary")
async def get_data_summary(db: AsyncSession = Depends(get_db)):
    """Get summary of data in database."""
    try:
        # Count records in each table
        tables = ['risk_scores', 'risk_factors', 'alerts']
        summary = {}
        
        for table in tables:
            try:
                result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                summary[table] = count
            except Exception as table_error:
                summary[table] = f"Error: {str(table_error)}"
        
        return {
            "status": "success",
            "data_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Data summary failed: {str(e)}")