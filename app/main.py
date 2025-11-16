import os
import logging
from datetime import datetime
from typing import Dict
import asyncio
from functools import wraps

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Setup logging first
from app.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from app.services.ingestion import ingest_local_series
from app.services.geri import compute_griscore, compute_geri_score
from app.services.impact import load_snapshot
from app.ml.regime import classify_regime
from app.ml.forecast import forecast_delta
from app.ml.anomaly import detect_anomalies
from app.services.transparency import get_data_freshness, get_update_log
from app.api import submissions as submissions_router
from app.api import monitoring as monitoring_router
from app.api import analytics as analytics_router
from app.api import community as community_router
from app.api import communication as communication_router
from app.db import SessionLocal, Base, engine
from app.models import ObservationModel
from app.core.config import get_settings
from app.core.security import require_analytics_rate_limit, require_ai_rate_limit, require_system_rate_limit
from app.core.auth import require_observatory_read, require_ai_read, optional_auth
from sqlalchemy import desc

def with_timeout(seconds: int):
    """Decorator to add timeout to operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Operation timed out after {seconds} seconds. Using cached data."
                )
        return wrapper
    return decorator

settings = get_settings()
app = FastAPI(
    title="RRIO GRII API", 
    version="0.4.0",
    description="RiskSX Resilience Intelligence Observatory - Global Economic Resilience Index",
    # Production optimizations
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Security middleware
allowed_origins = [origin.strip() for origin in settings.allowed_origins.split(",")]
if settings.is_development:
    allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "Accept"],
)

# Response compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request size and timeout middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers and request limits."""
    # Check request size (16MB limit)
    if hasattr(request, 'headers') and 'content-length' in request.headers:
        content_length = int(request.headers['content-length'])
        if content_length > 16_000_000:  # 16MB
            raise HTTPException(status_code=413, detail="Request too large")
    
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Trusted host middleware for production
if settings.is_production:
    allowed_hosts = os.getenv('RIS_ALLOWED_HOSTS', '*.rrio.dev,*.risksx.com').split(',')
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )


@app.on_event("startup")
async def startup_event():
    """Initialize database and load data cache on startup."""
    # Ensure database tables exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as exc:
        logger.error(f"Database initialization failed: {exc}")
        # Continue startup even if DB init fails, as tables may already exist
    
    # Load initial data cache with timeout protection
    try:
        # Try to load from cache first, then ingest if needed
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("ingestion") 
        
        # Check if we have any cached data
        from app.data.registry import SERIES_REGISTRY
        cached_count = 0
        for series_id in SERIES_REGISTRY.keys():
            data, _ = cache.get(series_id)
            if data:
                cached_count += 1
        
        if cached_count > 0:
            logger.info(f"Using {cached_count} cached series for startup")
            app.state.observations = {}  # Will load from cache on demand
        else:
            logger.info("No cache found, attempting initial ingestion with timeout...")
            app.state.observations = await asyncio.wait_for(
                asyncio.to_thread(ingest_local_series),
                timeout=30  # 30 second timeout for startup
            )
            logger.info(f"Initial ingestion completed: {len(app.state.observations)} series")
            
    except asyncio.TimeoutError:
        logger.warning("Startup ingestion timed out, will load from cache on demand")
        app.state.observations = {}
    except Exception as e:
        logger.error(f"Startup ingestion failed: {e}, will load from cache on demand")
        app.state.observations = {}


def _get_observations() -> dict:
    """Get observations with cache fallback to avoid timeouts."""
    observations = getattr(app.state, "observations", None)
    
    if observations is None or len(observations) == 0:
        # Try to load from cache first to avoid blocking API calls
        from app.core.unified_cache import UnifiedCache
        from app.data.registry import SERIES_REGISTRY
        from app.services.ingestion import Observation
        from datetime import datetime
        
        cache = UnifiedCache("ingestion")
        observations = {}
        
        for series_id in SERIES_REGISTRY.keys():
            data, metadata = cache.get(series_id)
            if data and 'timestamp' in data and 'value' in data:
                # Convert cached data to observation format
                timestamp_str = data['timestamp']
                if 'T' not in timestamp_str:
                    timestamp_str += 'T00:00:00'
                
                obs = Observation(
                    series_id=series_id,
                    observed_at=datetime.fromisoformat(timestamp_str.replace('Z', '')),
                    value=float(data['value'])
                )
                observations[series_id] = [obs]
        
        app.state.observations = observations
    
    return observations


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "ok", 
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "version": "0.4.0",
        "environment": settings.environment
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependencies."""
    from sqlalchemy import text
    health_status = {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    checks = {}
    
    # Check database connection
    try:
        from app.db import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy"}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis connection
    try:
        from app.core.cache import RedisCache
        cache = RedisCache("health")
        if cache.available:
            cache.client.ping()
            checks["redis"] = {"status": "healthy"}
        else:
            checks["redis"] = {"status": "unavailable"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check data freshness
    try:
        from app.services.geri import compute_geri_score
        geri_data = compute_geri_score()
        if geri_data:
            checks["data"] = {"status": "healthy", "last_update": geri_data.get("last_updated")}
        else:
            checks["data"] = {"status": "stale"}
    except Exception as e:
        checks["data"] = {"status": "unhealthy", "error": str(e)}
    
    health_status["checks"] = checks
    return health_status


@app.get("/api/v1/analytics/geri")
def current_griscore(
    _rate_limit: bool = Depends(require_analytics_rate_limit),
    _auth: dict = Depends(optional_auth)
) -> Dict[str, object]:
    """Get current GERI score with full v1 methodology."""
    observations = _get_observations()
    
    # Get regime classification for potential weight override
    regime_probs = classify_regime(observations)
    regime_confidence = max(regime_probs.values()) if regime_probs else 0.0
    
    # Compute full GERI score
    result = compute_geri_score(observations, regime_confidence=regime_confidence)
    
    # Add API-specific formatting
    result["drivers"] = [
        {"component": comp, "contribution": round(value, 3)}
        for comp, value in result["contributions"].items()
    ]
    result["color"] = _band_color(result["band"])
    
    return result


@app.get("/api/v1/ai/regime/current")
def current_regime(
    _rate_limit: bool = Depends(require_ai_rate_limit),
    _auth: dict = Depends(optional_auth)
) -> Dict[str, object]:
    observations = _get_observations()
    probabilities = classify_regime(observations)
    return {
        "regime": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
    }


@app.get("/api/v1/ai/forecast/next-24h")
def next_day_forecast(
    _rate_limit: bool = Depends(require_ai_rate_limit),
    _auth: dict = Depends(optional_auth)
) -> Dict[str, float]:
    observations = _get_observations()
    return forecast_delta(observations)


@app.get("/api/v1/anomalies/latest")
def anomaly_feed() -> Dict[str, float]:
    observations = _get_observations()
    return detect_anomalies(observations)


@app.get("/api/v1/impact/ras")
def ras_snapshot() -> Dict[str, object]:
    snapshot = load_snapshot()
    return snapshot.to_dict()

@app.get("/api/v1/transparency/data-freshness")
def data_freshness() -> dict:
    return {"freshness": get_data_freshness()}


@app.get("/api/v1/transparency/update-log")
def update_log() -> dict:
    return {"entries": get_update_log()}


# Additional API endpoints per documentation

@app.get("/api/v1/analytics/geri/history")
def geri_history(limit: int = 50, offset: int = 0) -> Dict[str, object]:
    """Get historical GERI snapshots with pagination."""
    try:
        # Validate pagination parameters
        limit = min(max(1, limit), 1000)  # Between 1 and 1000
        offset = max(0, offset)
        
        db = SessionLocal()
        # Get observations with pagination for memory efficiency
        recent_obs = db.query(ObservationModel).order_by(
            desc(ObservationModel.observed_at)
        ).offset(offset).limit(limit).all()
        
        # Get total count for pagination metadata
        total_count = db.query(ObservationModel).count()
        db.close()
        
        # Group by timestamp and compute historical scores
        timestamps = {}
        for obs in recent_obs:
            ts_key = obs.observed_at.isoformat()
            if ts_key not in timestamps:
                timestamps[ts_key] = {}
            timestamps[ts_key][obs.series_id] = obs.value
        
        # Generate historical series with memory-efficient processing
        series = []
        for timestamp, values in timestamps.items():
            if len(values) >= 3:  # Need minimum data
                # Simplified historical score calculation
                score = 50 + sum(values.values()) / len(values) * 0.1
                score = max(0, min(100, score))
                series.append({
                    "timestamp": timestamp,
                    "score": round(score, 2)
                })
        
        # Sort by timestamp for consistent ordering
        series.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "series": series,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        return {"series": [], "error": str(e)}


@app.get("/api/v1/analytics/components")
def geri_components() -> Dict[str, object]:
    """Get component-level values and z-scores."""
    observations = _get_observations()
    
    components = []
    for series_id, obs_list in observations.items():
        if obs_list:
            latest_obs = obs_list[-1]
            # Simplified z-score calculation
            values = [o.value for o in obs_list[-30:]]  # Last 30 values
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                z_score = (latest_obs.value - mean_val) / (std_val or 1.0)
            else:
                z_score = 0.0
            
            components.append({
                "id": series_id,
                "value": round(latest_obs.value, 2),
                "z_score": round(z_score, 3),
                "timestamp": latest_obs.observed_at.isoformat() + "Z"
            })
    
    return {"components": components}


@app.get("/api/v1/system/data-freshness")
def system_data_freshness(_rate_limit: bool = Depends(require_system_rate_limit)) -> dict:
    """TTL status per component."""
    return {"freshness": get_data_freshness()}


@app.get("/api/v1/system/releases")
def system_releases() -> dict:
    """Release history referencing governance approvals."""
    # Placeholder implementation
    return {
        "releases": [
            {
                "version": "v1.0.0",
                "date": "2024-11-16",
                "description": "Phase A implementation with 8-component GERI",
                "governance_approval": "2024-11-15",
                "changes": ["Added 5-year rolling window", "Implemented all 8 GERI components"]
            }
        ]
    }


@app.get("/api/v1/impact/partners")
def impact_partners() -> dict:
    """Active Partner Labs, sectors served, deliverables."""
    # Placeholder implementation
    return {
        "partners": [
            {
                "lab_id": "fintech_lab_01",
                "sector": "financial_services", 
                "status": "active",
                "deliverables": ["risk_assessment", "scenario_analysis"],
                "showcase_date": "2024-12-01"
            }
        ]
    }


app.include_router(submissions_router.router)
app.include_router(monitoring_router.router)
app.include_router(analytics_router.router)
app.include_router(community_router.router)
app.include_router(communication_router.router)


def _band_color(band: str) -> str:
    mapping = {
        "minimal": "#00C853",
        "low": "#64DD17",
        "moderate": "#FFD600",
        "high": "#FFAB00",
        "critical": "#D50000",
    }
    return mapping.get(band, "#64DD17")
