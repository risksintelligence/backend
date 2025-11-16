import time
from typing import Dict
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name='X-RRIO-REVIEWER', auto_error=False)
settings = get_settings()

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, Dict[str, any]] = {}

class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        if client_id not in rate_limit_storage:
            rate_limit_storage[client_id] = {'requests': [], 'reset_time': current_time + self.window_size}
        
        client_data = rate_limit_storage[client_id]
        
        # Clean old requests outside window
        client_data['requests'] = [req_time for req_time in client_data['requests'] if req_time > window_start]
        
        # Check if under limit
        if len(client_data['requests']) >= self.requests_per_minute:
            return False
        
        # Add current request
        client_data['requests'].append(current_time)
        client_data['reset_time'] = current_time + self.window_size
        
        return True
    
    def get_reset_time(self, client_id: str) -> int:
        """Get when the rate limit resets."""
        return int(rate_limit_storage.get(client_id, {}).get('reset_time', time.time()))

# Rate limiters for different endpoint types
analytics_limiter = RateLimiter(60)  # 60 req/min for analytics
ai_limiter = RateLimiter(30)         # 30 req/min for AI endpoints
system_limiter = RateLimiter(120)    # 120 req/min for system endpoints

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use API key if available, otherwise IP address
    api_key = request.headers.get('X-RRIO-API-KEY')
    if api_key:
        return f"api:{api_key[:10]}"  # First 10 chars of API key
    
    # Use X-Forwarded-For if behind proxy, otherwise direct IP
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    client_host = getattr(request.client, 'host', 'unknown')
    return f"ip:{client_host}"

def require_rate_limit(limiter: RateLimiter, endpoint_type: str = "general"):
    """Rate limiting dependency."""
    def dependency(request: Request):
        client_id = get_client_id(request)
        
        if not limiter.is_allowed(client_id):
            reset_time = limiter.get_reset_time(client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {endpoint_type} endpoints",
                headers={
                    "Retry-After": str(reset_time - int(time.time())),
                    "X-RateLimit-Limit": str(limiter.requests_per_minute),
                    "X-RateLimit-Reset": str(reset_time)
                }
            )
        
        return True
    
    return dependency

# Specific rate limit dependencies
require_analytics_rate_limit = require_rate_limit(analytics_limiter, "analytics")
require_ai_rate_limit = require_rate_limit(ai_limiter, "AI")
require_system_rate_limit = require_rate_limit(system_limiter, "system")

def require_reviewer(api_key: str = Depends(api_key_header)) -> None:
    """Require reviewer API key authentication."""
    if not api_key or api_key != settings.reviewer_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Reviewer authentication failed'
        )

def require_api_key(request: Request) -> str:
    """Require API key from header or query parameter."""
    # Check header first
    api_key = request.headers.get('X-RRIO-API-KEY')
    
    # Check query parameter as fallback
    if not api_key:
        api_key = request.query_params.get('api_key')
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via X-RRIO-API-KEY header or api_key query parameter."
        )
    
    # In production, validate against database of valid API keys
    # For now, just ensure it's present
    if len(api_key) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )
    
    return api_key
