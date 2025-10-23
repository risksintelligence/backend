from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db, check_database_connection, get_database_info
import os
from datetime import datetime

app = FastAPI(
    title="RiskX API",
    version="1.0.0",
    description="Risk Intelligence Platform - Bloomberg Terminal Style"
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
    
    return {
        "status": "healthy",
        "service": "riskx-backend",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "database": db_status.get("status", "unknown"),
            "cache": "not_connected_yet",
            "external_apis": "not_connected_yet"
        }
    }

@app.get("/api/v1/status")
async def api_status():
    # Check database connection for detailed status
    db_status = await check_database_connection()
    
    return {
        "api": "operational",
        "version": "1.0.0",
        "deployment": "render",
        "database": db_status.get("status", "unknown"),
        "cache": "pending_connection",
        "external_apis": "pending_connection",
        "features": {
            "risk_intelligence": "pending",
            "analytics": "pending", 
            "predictions": "pending",
            "network_analysis": "pending",
            "real_time_data": "pending"
        }
    }

@app.get("/api/v1/test")
async def test_endpoint():
    return {
        "message": "Test endpoint operational",
        "timestamp": datetime.utcnow().isoformat(),
        "test_data": {
            "sample_risk_score": 75.5,
            "sample_factors": ["economic", "financial", "geopolitical"],
            "sample_indicators": {
                "gdp_growth": 2.1,
                "unemployment": 3.7,
                "inflation": 3.2,
                "market_volatility": 0.18
            }
        },
        "platform_status": "initialization_complete"
    }

@app.get("/api/v1/platform/info")
async def platform_info():
    return {
        "platform": "RiskX Risk Intelligence",
        "style": "Bloomberg Terminal Inspired",
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

# Database testing endpoints
@app.get("/api/v1/database/test")
async def test_database():
    """Test database connection and basic operations."""
    connection_test = await check_database_connection()
    
    if connection_test.get("status") == "connected":
        database_info = await get_database_info()
        return {
            "database_connection": "success",
            "connection_test": connection_test,
            "database_info": database_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "database_connection": "failed",
            "connection_test": connection_test,
            "timestamp": datetime.utcnow().isoformat()
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