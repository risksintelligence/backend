from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db, check_database_connection, get_database_info, engine
from src.data.models.risk_models import Base
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
    Startup: Initialize database with real data, cache, background workers, and real-time streaming.
    Shutdown: Cleanup gracefully.
    ABSOLUTE RULE: Only real API data allowed - no placeholders
    """
    global refresh_worker
    
    # Startup
    logger.info("Starting RiskX Backend with REAL DATA ONLY")
    
    # Step 1: Initialize database tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - let app handle gracefully
    
    # Step 2: Initialize cache manager
    cache_manager = get_cache_manager()
    
    # Step 3: Start background refresh workers (they will populate with real data)
    refresh_worker = BackgroundRefreshWorker(cache_manager)
    
    # Start workers in background - they will fetch real FRED data
    asyncio.create_task(refresh_worker.start())
    
    # Step 4: Initialize real-time streaming
    try:
        from src.api.routes.websocket import initialize_real_time_streaming
        await initialize_real_time_streaming(cache_manager)
        logger.info("Real-time streaming initialized")
    except ImportError:
        logger.warning("WebSocket module not found - continuing without real-time streaming")
    except Exception as e:
        logger.error(f"Real-time streaming initialization failed: {e}")
    
    logger.info("RiskX Backend started - background workers fetching real data from FRED API")
    logger.info("ABSOLUTE RULE ENFORCED: Zero placeholder data allowed")
    
    yield  # Application runs
    
    # Shutdown
    logger.info("Shutting down RiskX Backend")
    
    if refresh_worker:
        await refresh_worker.stop()
        await refresh_worker.cleanup()
    
    # Shutdown real-time streaming
    try:
        from src.api.routes.websocket import shutdown_real_time_streaming
        await shutdown_real_time_streaming()
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
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


@app.get("/api/v1/dashboard")
async def get_main_dashboard():
    """Get main dashboard data - delegating to analytics router."""
    from src.api.routes.risk_analytics import get_dashboard_data
    from src.core.dependencies import get_cache_manager, get_db
    
    # Get dependencies
    cache_manager = get_cache_manager()
    
    # Call analytics dashboard function
    try:
        # Since this endpoint doesn't take db as parameter directly,
        # we need to implement dashboard logic here or redirect
        cache_key = "dashboard:main"
        cached_data = await cache_manager.get(cache_key, max_age_seconds=300)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # If no cache, try to get data from the cache populated by background workers
        risk_cache = await cache_manager.get("risk:overview", max_age_seconds=900)
        
        if risk_cache:
            # Use only real cached risk data - no hardcoded values allowed
            return {
                "status": "success",
                "data": risk_cache,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # No fallback data allowed - must return unavailable if no real data
        return {
            "status": "unavailable",
            "message": "Dashboard data not yet available - background workers populating cache",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        # No fake data allowed - return error status only
        return {
            "status": "error",
            "message": f"Dashboard data unavailable: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }



# Include intelligent caching API routes
app.include_router(risk.router)
app.include_router(economic.router)
app.include_router(cache_management.router)

# Include analytics routes
from src.api.routes import risk_analytics, analytics
app.include_router(risk_analytics.router)
app.include_router(analytics.router)

# Include working API routes only
from src.api.routes import external_apis
app.include_router(external_apis.router)