import importlib
import logging
import os
from typing import Optional

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

logger = logging.getLogger(__name__)

def load_router(module_path: str, attr: str = "router") -> Optional[object]:
    try:
        module = importlib.import_module(module_path)
        router = getattr(module, attr)
        logger.info(f"Successfully loaded router: {module_path}")
        return router
    except ModuleNotFoundError as exc:
        logger.warning("Skipping router %s due to missing dependency: %s", module_path, exc)
    except AttributeError:
        logger.warning("Router attribute '%s' missing in %s", attr, module_path)
    except Exception as exc:
        logger.error("Failed to load router %s: %s", module_path, exc, exc_info=True)
    return None


router_modules = [
    "src.api.v1.geri",
    "src.api.v1.ai",
    "src.api.v1.transparency",
    "src.api.v1.research",
    "src.api.v1.scenario",
    "src.api.v1.scenario_alerts",
    "src.api.v1.subscription",
    "src.api.v1.admin",
    "src.api.v1.alerts",
    "src.api.v1.monitoring",
    "src.api.v1.auth",
    "src.api.v1.backup",
    "src.api.v1.ml_admin",
    "src.api.v1.scenario_sharing",
    "src.api.v1.cron_admin",
    "src.api.v1.advanced_exports",
]

app = FastAPI(title="RiskSX Intelligence System API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    
    # Don't expose internal errors in production
    if os.getenv("ENVIRONMENT", "production").lower() in ["production", "prod"]:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {str(exc)}",
                "traceback": traceback.format_exc()
            }
        )

raw_origins = os.getenv("RIS_ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

# Add production domains if not already included
production_domains = [
    "https://frontend-1-wvu7.onrender.com",
    "https://app.risksx.io",
    "http://localhost:3000",
    "http://localhost:3001"
]

for domain in production_domains:
    if domain not in allowed_origins and "*" not in allowed_origins:
        allowed_origins.append(domain)

wildcard = len(allowed_origins) == 1 and allowed_origins[0] == "*"
allow_credentials = not wildcard
origins_param = ["*"] if wildcard else allowed_origins

logger.info(f"CORS origins configured: {origins_param}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_param,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Set environment variables for test mode if not set
env = os.getenv("ENVIRONMENT", "development").lower()
if env in ["local", "dev", "development"]:
    os.environ.setdefault("RIS_TEST_MODE", "true")

# Ensure production has proper environment variable fallbacks
postgres_dsn = os.getenv("RIS_POSTGRES_DSN") or os.getenv("DATABASE_URL")
redis_url = os.getenv("RIS_REDIS_URL") or os.getenv("REDIS_URL")

if postgres_dsn and not os.getenv("RIS_POSTGRES_DSN"):
    os.environ["RIS_POSTGRES_DSN"] = postgres_dsn
    logger.info("Using DATABASE_URL for RIS_POSTGRES_DSN")

if redis_url and not os.getenv("RIS_REDIS_URL"):
    os.environ["RIS_REDIS_URL"] = redis_url  
    logger.info("Using REDIS_URL for RIS_REDIS_URL")

# Set API key defaults if not provided
os.environ.setdefault("RIS_FRED_API_KEY", "demo-key")
os.environ.setdefault("RIS_EIA_API_KEY", "demo-key")

logger.info(f"Starting RIS API in {env} environment")
logger.info(f"Test mode: {os.getenv('RIS_TEST_MODE', 'false')}")
logger.info(f"Database URL: {os.getenv('RIS_POSTGRES_DSN', 'not set')[:50]}...")

loaded_routers = 0
failed_routers = []

for module in router_modules:
    router = load_router(module)
    if router:
        try:
            app.include_router(router)
            loaded_routers += 1
        except Exception as e:
            logger.error(f"Failed to include router {module}: {e}")
            failed_routers.append(module)
    else:
        failed_routers.append(module)

logger.info(f"Loaded {loaded_routers}/{len(router_modules)} router modules")
if failed_routers:
    logger.warning(f"Failed to load routers: {', '.join(failed_routers)}")
    
# Add router loading status endpoint
@app.get("/debug/routers")
async def router_status():
    return {
        "loaded_routers": loaded_routers,
        "total_routers": len(router_modules),
        "failed_routers": failed_routers,
        "environment": env,
        "test_mode": os.getenv("RIS_TEST_MODE", "false")
    }


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "RIS Intelligence API",
        "docs": "/docs",
        "status": "healthy",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
