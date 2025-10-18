"""
FastAPI application entry point for RiskX.
"""
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import settings
from src.core.security import SecurityHeaders, InputValidator
from src.api.routes import health, risk, analytics, websocket, network, prediction, simulation, data, monitoring
from src.api.middleware.security import SecurityMiddleware, RateLimitMiddleware, InputValidationMiddleware
from src.api.middleware.monitoring import RequestMonitoringMiddleware, HealthCheckMiddleware
from src.monitoring import start_background_metrics_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    print(f"Starting {settings.app_name}")
    
    # Start background monitoring
    await start_background_metrics_collection()
    print("Production monitoring started")
    
    yield
    
    # Shutdown
    print(f"Shutting down {settings.app_name}")


# Create FastAPI application
app = FastAPI(
    title="RiskX API",
    description="AI Risk Intelligence Observatory - API for risk analysis and prediction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add monitoring middleware (first for accurate metrics)
app.add_middleware(RequestMonitoringMiddleware)  # Request monitoring and metrics
app.add_middleware(HealthCheckMiddleware)        # Health check optimization

# Add security middleware for open-source platform (order matters)
app.add_middleware(InputValidationMiddleware)  # Prevent malicious input
app.add_middleware(SecurityMiddleware)         # Security headers and basic protection
app.add_middleware(RateLimitMiddleware)        # Prevent abuse

# Add CORS middleware - temporarily permissive for debugging
# TODO: Restore to settings.cors_origins once connectivity confirmed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHost middleware - temporarily allow all hosts for Render deployment  
# TODO: Restore to proper host validation once connectivity confirmed
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Temporarily allow all hosts to fix 400 errors
)

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "path": str(request.url.path),
            "method": request.method
        }
    )

# Root endpoint for basic API info
@app.get("/")
async def root():
    """Basic API information endpoint."""
    return {
        "name": "RiskX API",
        "version": "1.0.0",
        "description": "AI Risk Intelligence Observatory API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/api/v1/health",
            "docs": "/docs",
            "api": "/api/v1"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests."""
    return JSONResponse(content={"message": "No favicon available"}, status_code=204)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["risk"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(network.router, prefix="/api/v1/network", tags=["network"])
app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["prediction"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["simulation"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Root endpoint - must be defined before catch-all route
@app.get("/")
async def root(request: Request):
    """Root endpoint - serves frontend if available, otherwise API info."""
    try:
        logger.info(f"Root endpoint accessed from {request.client.host if request.client else 'unknown'}")
        
        # Configure static file serving for frontend
        frontend_export_path = Path("../frontend/out")
        
        # Try to serve frontend index.html if available
        if frontend_export_path.exists():
            index_file = frontend_export_path / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
        
        # If no frontend available, return API information
        return {
            "message": f"Welcome to {settings.app_name} API",
            "version": "1.0.0", 
            "environment": settings.environment,
            "debug": settings.debug,
            "description": "AI Risk Intelligence Observatory",
            "endpoints": {
                "health": "/api/v1/health",
                "docs": "/docs" if settings.debug else "disabled",
                "redoc": "/redoc" if settings.debug else "disabled"
            },
            "host_info": {
                "host": request.url.hostname,
                "scheme": request.url.scheme,
                "port": request.url.port
            },
            "frontend_status": "Frontend not built - run 'cd frontend && npm run build && npm run export'"
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# Configure static file serving for frontend
frontend_export_path = Path("../frontend/out")

# Mount static files if frontend export exists
if frontend_export_path.exists():
    # Mount all static assets from the exported frontend
    static_path = frontend_export_path / "_next" / "static"
    if static_path.exists():
        app.mount("/_next/static", StaticFiles(directory=str(static_path)), name="next_static")
    
    # Mount any other static assets
    for static_dir in ["images", "icons", "assets"]:
        static_dir_path = frontend_export_path / static_dir
        if static_dir_path.exists():
            app.mount(f"/{static_dir}", StaticFiles(directory=str(static_dir_path)), name=f"{static_dir}_static")
    
    logger.info("Serving frontend static files from export")

# Serve frontend for all other non-API routes (SPA fallback)
@app.get("/{path:path}")
async def serve_frontend(path: str, request: Request):
    """Serve frontend for all non-API routes."""
    # Skip API routes
    if path.startswith("api/") or path.startswith("docs") or path.startswith("redoc") or path.startswith("ws/"):
        raise HTTPException(status_code=404, detail="API route not found")
    
    # Try to serve from Next.js export
    frontend_export_path = Path("../frontend/out")
    if frontend_export_path.exists():
        index_file = frontend_export_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    
    # If no frontend build found, return API info
    return {
        "message": "RiskX API - Frontend not built",
        "info": "This is the API endpoint. To see the frontend, build the frontend first.",
        "build_command": "cd frontend && npm run build && npm run export",
        "api_docs": "/docs" if settings.debug else "disabled"
    }