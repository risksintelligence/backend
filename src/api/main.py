from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db, check_database_connection, get_database_info
from src.core.cache import check_redis_connection, get_redis_info, BasicCacheManager
from src.cache.cache_manager import IntelligentCacheManager
from src.cache.refresh_worker import BackgroundRefreshWorker
from src.core.dependencies import get_cache_manager
from src.api.routes import risk, economic, cache_management
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Global worker reference
refresh_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Initialize cache, background workers, and real-time streaming.
    Shutdown: Cleanup gracefully.
    """
    global refresh_worker
    
    # Startup
    logger.info("Starting RiskX Backend with Intelligent Caching and Real-Time Streaming")
    
    # Initialize cache manager
    cache_manager = get_cache_manager()
    
    # Start background refresh workers
    refresh_worker = BackgroundRefreshWorker(cache_manager)
    
    # Start workers in background
    asyncio.create_task(refresh_worker.start())
    
    # Initialize real-time streaming
    from src.api.routes.websocket import initialize_real_time_streaming
    await initialize_real_time_streaming(cache_manager)
    
    logger.info("Intelligent caching system, background workers, and real-time streaming started")
    
    yield  # Application runs
    
    # Shutdown
    logger.info("Shutting down RiskX Backend")
    
    if refresh_worker:
        await refresh_worker.stop()
        await refresh_worker.cleanup()
    
    # Shutdown real-time streaming
    from src.api.routes.websocket import shutdown_real_time_streaming
    await shutdown_real_time_streaming()
    
    logger.info("Cleanup complete")


app = FastAPI(
    title="RiskX API",
    version="1.0.0",
    description="Risk Intelligence Platform - Sophisticated White Background Dashboard with Intelligent Caching",
    lifespan=lifespan
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "RiskX Risk Intelligence Platform",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": "Bloomberg Terminal Style Interface"
    }

@app.get("/api/v1/health")
async def health_check():
    # Check database connection
    db_status = await check_database_connection()
    
    # Check Redis connection
    cache_status = await check_redis_connection()
    
    return {
        "status": "healthy",
        "service": "riskx-backend",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "database": db_status.get("status", "unknown"),
            "cache": cache_status.get("status", "unknown"),
            "external_apis": "not_connected_yet"
        }
    }

@app.get("/api/v1/status")
async def api_status():
    # Check database connection for detailed status
    db_status = await check_database_connection()
    
    # Check Redis connection for detailed status
    cache_status = await check_redis_connection()
    
    return {
        "api": "operational",
        "version": "1.0.0",
        "deployment": "render",
        "database": db_status.get("status", "unknown"),
        "cache": cache_status.get("status", "unknown"),
        "external_apis": "pending_connection",
        "features": {
            "risk_intelligence": "pending",
            "analytics": "pending", 
            "predictions": "pending",
            "network_analysis": "pending",
            "real_time_data": "pending"
        }
    }


@app.get("/api/v1/platform/info")
async def platform_info():
    return {
        "platform": "RiskX Risk Intelligence",
        "style": "Sophisticated White Background Dashboard",
        "capabilities": [
            "Real-time risk assessment",
            "Economic intelligence analytics", 
            "ML-powered predictions",
            "Network vulnerability analysis",
            "Professional dashboard interface"
        ],
        "data_sources": [
            "Federal Reserve Economic Data (FRED)",
            "Bureau of Economic Analysis (BEA)",
            "Bureau of Labor Statistics (BLS)",
            "U.S. Census Bureau"
        ],
        "deployment": "Production Ready on Render"
    }


@app.get("/api/v1/database/info")
async def database_info():
    """Get detailed database information."""
    return await get_database_info()

@app.get("/api/v1/database/tables")
async def list_database_tables(db: AsyncSession = Depends(get_db)):
    """List all tables in the database."""
    try:
        from sqlalchemy import text
        result = await db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]
        
        return {
            "status": "success",
            "tables": tables,
            "table_count": len(tables),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/v1/cache/info")
async def cache_info():
    """Get detailed Redis cache information."""
    return await get_redis_info()



# Include intelligent caching API routes
app.include_router(risk.router)
app.include_router(economic.router)
app.include_router(cache_management.router)

# Include working API routes only
from src.api.routes import external_apis
app.include_router(external_apis.router)