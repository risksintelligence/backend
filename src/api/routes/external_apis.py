from fastapi import APIRouter, HTTPException
from datetime import datetime
from src.data.sources import fred, bea, bls, census
import asyncio

router = APIRouter(prefix="/api/v1/external", tags=["external_apis"])


@router.get("/fred/indicators")
async def get_fred_indicators():
    """Get key FRED economic indicators."""
    try:
        indicators = await fred.get_key_indicators()
        return {
            "status": "success",
            "data": indicators,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/fred/{series_id}")
async def get_fred_series(series_id: str):
    """Get specific FRED series data."""
    try:
        async with fred.FREDClient() as client:
            data = await client.get_series(series_id.upper())
        
        if data:
            return {
                "status": "success",
                "series_id": series_id,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "not_found",
                "message": f"Series {series_id} not found or unavailable",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/bea/accounts")
async def get_bea_accounts():
    """Get BEA economic accounts data."""
    try:
        accounts = await bea.get_economic_accounts()
        return {
            "status": "success",
            "data": accounts,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/bls/labor")
async def get_bls_labor():
    """Get BLS labor statistics."""
    try:
        labor_stats = await bls.get_labor_statistics()
        return {
            "status": "success",
            "data": labor_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/census/population")
async def get_census_population():
    """Get Census population data."""
    try:
        population = await census.get_population_data()
        income = await census.get_household_income()
        
        return {
            "status": "success",
            "data": {
                "population": population,
                "household_income": income
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/health")
async def check_apis_health():
    """Check health of all external APIs."""
    try:
        # Check all APIs concurrently
        health_checks = await asyncio.gather(
            fred.health_check(),
            bea.health_check(), 
            bls.health_check(),
            census.health_check(),
            return_exceptions=True
        )
        
        api_names = ["fred", "bea", "bls", "census"]
        health_status = {}
        
        for i, health in enumerate(health_checks):
            api_name = api_names[i]
            if isinstance(health, Exception):
                health_status[api_name] = {"status": "error", "error": str(health)}
            else:
                health_status[api_name] = {"status": "healthy" if health else "unhealthy"}
        
        # Overall status
        healthy_count = sum(1 for status in health_status.values() 
                          if status.get("status") == "healthy")
        
        return {
            "status": "success",
            "overall_health": f"{healthy_count}/{len(api_names)} APIs healthy",
            "apis": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }