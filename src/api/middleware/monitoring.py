"""
Request Monitoring Middleware for RiskX API

Tracks all API requests, response times, and error rates
for comprehensive production monitoring and analytics.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)


class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor API requests and collect performance metrics.
    
    Tracks:
    - Request/response times
    - HTTP status codes
    - Error rates
    - Request patterns
    - User agents and IP addresses
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {
            "/health",
            "/metrics", 
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent")
        client_host = request.client.host if request.client else None
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_message = None
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            status_code = 500
            error_message = str(e)
            
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Record metrics (skip health checks and static assets)
        if not any(excluded in path for excluded in self.excluded_paths):
            try:
                await metrics_collector.record_api_request(
                    endpoint=path,
                    method=method,
                    status_code=status_code,
                    response_time=response_time,
                    user_agent=user_agent,
                    ip_address=client_host,
                    error_message=error_message
                )
            except Exception as e:
                logger.error(f"Error recording metrics: {e}")
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-ID"] = f"{int(start_time * 1000000)}"
        
        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle health checks and system status endpoints.
    
    Provides quick health status without full request processing.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle health check requests"""
        
        # Quick health check endpoint
        if request.url.path == "/health/quick":
            from fastapi.responses import JSONResponse
            return JSONResponse({
                "status": "healthy",
                "timestamp": time.time(),
                "service": "riskx-api"
            })
        
        # Detailed health check
        elif request.url.path == "/health/detailed":
            try:
                health_status = await metrics_collector.get_system_health_status()
                from fastapi.responses import JSONResponse
                return JSONResponse(health_status)
            except Exception as e:
                from fastapi.responses import JSONResponse
                return JSONResponse({
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }, status_code=500)
        
        # Continue with normal processing
        return await call_next(request)