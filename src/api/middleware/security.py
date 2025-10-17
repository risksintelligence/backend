"""
Security middleware for FastAPI application.
Provides authentication, rate limiting, input validation, and security headers.
"""
import time
import logging
from typing import Callable, Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED

from src.core.config import settings
from src.core.security import SecurityHeaders, InputValidator, SecurityMiddleware as CoreSecurityMiddleware
from src.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware for all API requests.
    
    Features:
    - Security headers injection
    - Request size validation
    - Input sanitization
    - Security logging
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.input_validator = InputValidator()
        self.security_headers = SecurityHeaders()
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process security checks for each request."""
        start_time = time.time()
        
        try:
            # Validate request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request entity too large"},
                    headers=self.security_headers.get_security_headers()
                )
            
            # Validate content type for POST/PUT requests
            if request.method in ['POST', 'PUT', 'PATCH'] and request.headers.get('content-type'):
                content_type = request.headers.get('content-type', '').lower()
                if not any(allowed in content_type for allowed in ['application/json', 'multipart/form-data', 'application/x-www-form-urlencoded']):
                    logger.warning(f"Invalid content type: {content_type} from {request.client.host}")
                    return JSONResponse(
                        status_code=415,
                        content={"error": "Unsupported media type"},
                        headers=self.security_headers.get_security_headers()
                    )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers to response
            security_headers = self.security_headers.get_security_headers()
            for header, value in security_headers.items():
                response.headers[header] = value
            
            # Log request metrics
            duration = time.time() - start_time
            logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers=self.security_headers.get_security_headers()
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for open-source platform.
    
    Features:
    - Per-IP rate limiting to prevent abuse
    - Generous limits for research and educational use
    - Different limits for different endpoints
    - Configurable time windows
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.cache_manager = CacheManager()
        self.core_security = CoreSecurityMiddleware()
        
        # Rate limit configurations - generous for open-source research platform
        self.rate_limits = {
            '/api/v1/prediction': {'requests': 500, 'window': 3600},  # 500 requests per hour
            '/api/v1/risk': {'requests': 1000, 'window': 3600},       # 1000 requests per hour
            '/api/v1/analytics': {'requests': 1000, 'window': 3600},  # 1000 requests per hour
            '/api/v1/data': {'requests': 2000, 'window': 3600},       # 2000 requests per hour
            'default': {'requests': 5000, 'window': 3600}            # 5000 requests per hour default
        }
    
    def get_rate_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for a path."""
        for pattern, config in self.rate_limits.items():
            if pattern != 'default' and path.startswith(pattern):
                return config
        return self.rate_limits['default']
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Check for forwarded IP
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Fall back to direct IP
        return request.client.host if request.client else 'unknown'
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        try:
            client_ip = self.get_client_identifier(request)
            path = request.url.path
            
            # Skip rate limiting for health checks and docs
            if path in ['/health', '/docs', '/redoc', '/openapi.json']:
                return await call_next(request)
            
            # Get rate limit configuration
            config = self.get_rate_limit_config(path)
            
            # Check rate limit using core security middleware
            if not self.core_security.check_rate_limit(client_ip, path):
                logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
                
                return JSONResponse(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": config['window']
                    },
                    headers={
                        "Retry-After": str(config['window']),
                        "X-RateLimit-Limit": str(config['requests']),
                        "X-RateLimit-Window": str(config['window'])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            response.headers["X-RateLimit-Limit"] = str(config['requests'])
            response.headers["X-RateLimit-Window"] = str(config['window'])
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            # Continue with request on middleware error
            return await call_next(request)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for protected endpoints.
    
    Features:
    - JWT token validation
    - Optional authentication for public endpoints
    - User context injection
    - Authentication logging
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.core_security = CoreSecurityMiddleware()
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            '/health',
            '/docs',
            '/redoc',
            '/openapi.json',
            '/api/v1/health',
            '/api/v1/health/ready',
            '/api/v1/health/live'
        }
        
        # Optional authentication endpoints (work with or without auth)
        self.optional_auth_endpoints = {
            '/api/v1/risk/score',
            '/api/v1/analytics/overview',
            '/api/v1/data/sources'
        }
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        return path in self.public_endpoints
    
    def is_optional_auth_endpoint(self, path: str) -> bool:
        """Check if endpoint has optional authentication."""
        return path in self.optional_auth_endpoints
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply authentication checks to requests."""
        try:
            path = request.url.path
            
            # Skip authentication for public endpoints
            if self.is_public_endpoint(path):
                return await call_next(request)
            
            # Get authorization header
            auth_header = request.headers.get('Authorization')
            
            # Handle optional authentication endpoints
            if self.is_optional_auth_endpoint(path):
                if auth_header:
                    # Validate token if provided
                    token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else None
                    if token:
                        try:
                            payload = self.core_security.validate_authentication(token)
                            if payload:
                                # Add user context to request state
                                request.state.user = payload
                        except Exception as e:
                            logger.warning(f"Optional auth validation failed: {e}")
                
                return await call_next(request)
            
            # Require authentication for protected endpoints
            if not auth_header or not auth_header.startswith('Bearer '):
                logger.warning(f"Missing or invalid authorization header for {path}")
                return JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Authentication required",
                        "message": "Valid authorization token required"
                    },
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Extract and validate token
            token = auth_header.replace('Bearer ', '')
            try:
                payload = self.core_security.validate_authentication(token)
                if not payload:
                    raise ValueError("Invalid token")
                
                # Add user context to request state
                request.state.user = payload
                
                # Process request with authenticated context
                return await call_next(request)
                
            except Exception as e:
                logger.warning(f"Authentication failed for {path}: {e}")
                return JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Authentication failed",
                        "message": "Invalid or expired token"
                    },
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers={"WWW-Authenticate": "Bearer"}
            )


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Input validation and sanitization middleware.
    
    Features:
    - SQL injection prevention
    - XSS protection
    - Input length validation
    - Malicious pattern detection
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.input_validator = InputValidator()
        self.max_json_size = 10 * 1024 * 1024  # 10MB
        
        # Suspicious patterns to detect
        self.suspicious_patterns = [
            r'(?i)(union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|drop\s+table)',
            r'(?i)(<script|javascript:|data:text/html|eval\(|expression\()',
            r'(?i)(\.\.\/|\.\.\\|\.\.\%2f|\.\.\%5c)',  # Path traversal
            r'(?i)(exec\s*\(|system\s*\(|passthru\s*\(|shell_exec\s*\()'  # Command injection
        ]
    
    def scan_for_threats(self, content: str) -> Optional[str]:
        """Scan content for security threats."""
        import re
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content):
                return pattern
        return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate and sanitize request input."""
        try:
            # Skip validation for certain endpoints
            if request.url.path in ['/health', '/docs', '/redoc', '/openapi.json']:
                return await call_next(request)
            
            # Validate query parameters
            query_string = str(request.url.query)
            if query_string:
                threat = self.scan_for_threats(query_string)
                if threat:
                    logger.warning(f"Suspicious query pattern detected: {threat} from {request.client.host}")
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid request parameters"}
                    )
            
            # For POST/PUT requests, validate body content
            if request.method in ['POST', 'PUT', 'PATCH']:
                # Get content type
                content_type = request.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    # Read and validate JSON body
                    body = await request.body()
                    
                    if len(body) > self.max_json_size:
                        logger.warning(f"JSON body too large: {len(body)} bytes from {request.client.host}")
                        return JSONResponse(
                            status_code=413,
                            content={"error": "Request entity too large"}
                        )
                    
                    # Scan body for threats
                    if body:
                        body_str = body.decode('utf-8', errors='ignore')
                        threat = self.scan_for_threats(body_str)
                        if threat:
                            logger.warning(f"Suspicious body pattern detected: {threat} from {request.client.host}")
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Invalid request content"}
                            )
            
            # Process request
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Input validation middleware error: {e}")
            # Continue with request on middleware error to avoid blocking legitimate requests
            return await call_next(request)