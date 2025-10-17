"""
API dependencies for RiskX platform.
Provides dependency injection for FastAPI endpoints with authentication, validation, and monitoring.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, Generator
import time
import logging

from ..core.config import get_settings
from ..core.security import security_middleware, SecurityMiddleware, input_validator
from ..core.database import get_db_session
from ..core.exceptions import (
    AuthenticationError, AuthorizationError, RateLimitExceededError,
    ValidationError, handle_exception
)
from ..core.logging import performance_logger, security_logger

logger = logging.getLogger('riskx.api.dependencies')

# Security scheme
security = HTTPBearer(auto_error=False)


class RequestContext:
    """Context object for current request."""
    
    def __init__(self, request: Request):
        self.request = request
        self.start_time = time.time()
        self.user_id: Optional[str] = None
        self.permissions: list = []
        self.request_id: str = self._generate_request_id()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())
    
    def get_client_ip(self) -> str:
        """Get client IP address."""
        forwarded = self.request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.request.client.host
    
    def get_user_agent(self) -> str:
        """Get user agent string."""
        return self.request.headers.get("User-Agent", "Unknown")
    
    def get_duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


async def get_request_context(request: Request) -> RequestContext:
    """Get request context for current request."""
    return RequestContext(request)


async def get_database_session():
    """Get database session dependency."""
    async with get_db_session() as session:
        yield session


async def verify_rate_limit(
    context: RequestContext = Depends(get_request_context)
) -> RequestContext:
    """Verify rate limiting for request."""
    client_ip = context.get_client_ip()
    endpoint = context.request.url.path
    
    # Check rate limit
    if not security_middleware.check_rate_limit(client_ip, endpoint):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )
    
    return context


async def get_current_user(
    context: RequestContext = Depends(verify_rate_limit),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from token."""
    if not credentials:
        return None
    
    try:
        # Verify token
        payload = security_middleware.validate_authentication(credentials.credentials)
        
        if not payload:
            raise AuthenticationError("Invalid authentication token")
        
        # Update context with user info
        context.user_id = payload.get('user_id')
        context.permissions = payload.get('permissions', [])
        
        return {
            'user_id': payload.get('user_id'),
            'permissions': payload.get('permissions', []),
            'token_jti': payload.get('jti')
        }
        
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def require_authentication(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require valid authentication."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


def require_permission(permission: str):
    """Dependency factory for permission checking."""
    async def check_permission(
        user: Dict[str, Any] = Depends(require_authentication)
    ) -> Dict[str, Any]:
        if permission not in user.get('permissions', []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return check_permission


async def validate_json_input(
    request: Request,
    max_size_mb: int = 10
) -> Dict[str, Any]:
    """Validate JSON input from request."""
    try:
        # Check content length
        content_length = request.headers.get('content-length')
        if content_length:
            content_length = int(content_length)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if content_length > max_size_bytes:
                raise ValidationError(
                    f"Request body too large. Maximum size: {max_size_mb}MB"
                )
        
        # Parse JSON
        body = await request.json()
        
        # Basic validation
        if not isinstance(body, dict):
            raise ValidationError("Request body must be a JSON object")
        
        return body
        
    except ValueError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValidationError(f"Failed to parse request body: {str(e)}")


async def validate_query_params(
    request: Request,
    allowed_params: Optional[list] = None
) -> Dict[str, Any]:
    """Validate query parameters."""
    params = dict(request.query_params)
    
    # Check for allowed parameters
    if allowed_params:
        invalid_params = set(params.keys()) - set(allowed_params)
        if invalid_params:
            raise ValidationError(
                f"Invalid query parameters: {', '.join(invalid_params)}"
            )
    
    # Sanitize parameter values
    sanitized_params = {}
    for key, value in params.items():
        sanitized_params[key] = input_validator.sanitize_input(value)
    
    return sanitized_params


class PaginationParams:
    """Pagination parameters with validation."""
    
    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        max_size: int = 100
    ):
        # Validate page
        if page < 1:
            raise ValidationError("Page number must be >= 1")
        
        # Validate size
        if size < 1:
            raise ValidationError("Page size must be >= 1")
        if size > max_size:
            raise ValidationError(f"Page size must be <= {max_size}")
        
        self.page = page
        self.size = size
        self.offset = (page - 1) * size
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            'page': self.page,
            'size': self.size,
            'offset': self.offset
        }


async def get_pagination_params(
    page: int = 1,
    size: int = 20
) -> PaginationParams:
    """Get validated pagination parameters."""
    return PaginationParams(page=page, size=size)


class CacheControl:
    """Cache control parameters."""
    
    def __init__(self, no_cache: bool = False, max_age: int = 300):
        self.no_cache = no_cache
        self.max_age = max_age
    
    def get_headers(self) -> Dict[str, str]:
        """Get cache control headers."""
        if self.no_cache:
            return {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        else:
            return {
                'Cache-Control': f'public, max-age={self.max_age}'
            }


async def get_cache_control(
    no_cache: bool = False,
    max_age: int = 300
) -> CacheControl:
    """Get cache control settings."""
    return CacheControl(no_cache=no_cache, max_age=max_age)


class ResponseLogger:
    """Middleware for logging API responses."""
    
    def __init__(self, context: RequestContext):
        self.context = context
    
    def log_response(self, status_code: int, response_size: Optional[int] = None):
        """Log API response metrics."""
        duration_ms = self.context.get_duration_ms()
        
        # Log performance metrics
        performance_logger.log_api_request(
            method=self.context.request.method,
            path=self.context.request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=self.context.user_id
        )
        
        # Log security events for failed requests
        if status_code >= 400:
            security_logger.log_suspicious_activity(
                self.context.user_id or self.context.get_client_ip(),
                'api_error',
                {
                    'method': self.context.request.method,
                    'path': self.context.request.url.path,
                    'status_code': status_code,
                    'user_agent': self.context.get_user_agent()
                },
                'medium' if status_code < 500 else 'high'
            )


async def get_response_logger(
    context: RequestContext = Depends(get_request_context)
) -> ResponseLogger:
    """Get response logger for current request."""
    return ResponseLogger(context)


class RequestValidator:
    """Request validation utilities."""
    
    @staticmethod
    def validate_risk_score_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk score calculation request."""
        required_fields = ['indicators', 'weights']
        
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate indicators
        indicators = data['indicators']
        if not isinstance(indicators, dict):
            raise ValidationError("Indicators must be a dictionary")
        
        # Validate weights
        weights = data['weights']
        if not isinstance(weights, dict):
            raise ValidationError("Weights must be a dictionary")
        
        # Check weight sum
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValidationError("Weights must sum to 1.0")
        
        return data
    
    @staticmethod
    def validate_simulation_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario simulation request."""
        required_fields = ['scenario_type', 'parameters']
        
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate scenario type
        valid_scenario_types = ['economic_shock', 'supply_disruption', 'policy_change']
        if data['scenario_type'] not in valid_scenario_types:
            raise ValidationError(f"Invalid scenario type. Must be one of: {valid_scenario_types}")
        
        return data


# Create global validator instance
request_validator = RequestValidator()


# Common dependency combinations
async def get_authenticated_context(
    context: RequestContext = Depends(verify_rate_limit),
    user: Dict[str, Any] = Depends(require_authentication)
) -> tuple[RequestContext, Dict[str, Any]]:
    """Get authenticated request context."""
    return context, user


async def get_admin_context(
    context: RequestContext = Depends(verify_rate_limit),
    user: Dict[str, Any] = Depends(require_permission('admin'))
) -> tuple[RequestContext, Dict[str, Any]]:
    """Get admin request context."""
    return context, user


# Error handler for dependency injection
def handle_dependency_error(exc: Exception) -> HTTPException:
    """Handle errors from dependencies."""
    try:
        # Convert to RiskX exception
        riskx_exc = handle_exception(exc)
        
        # Map to HTTP status codes
        if isinstance(riskx_exc, AuthenticationError):
            status_code = status.HTTP_401_UNAUTHORIZED
            headers = {"WWW-Authenticate": "Bearer"}
        elif isinstance(riskx_exc, AuthorizationError):
            status_code = status.HTTP_403_FORBIDDEN
            headers = None
        elif isinstance(riskx_exc, RateLimitExceededError):
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            headers = {"Retry-After": "60"}
        elif isinstance(riskx_exc, ValidationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            headers = None
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            headers = None
        
        return HTTPException(
            status_code=status_code,
            detail=riskx_exc.user_message,
            headers=headers
        )
        
    except Exception:
        # Fallback for unexpected errors
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )