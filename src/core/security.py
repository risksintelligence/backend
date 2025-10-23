from datetime import datetime, timedelta
from typing import Optional, Union
import jwt
import bcrypt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
import os
from src.core.config import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# HTTP Bearer security scheme
security = HTTPBearer()


class SecurityConfig:
    """Security configuration and constants."""
    
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    TOKEN_BLACKLIST_CLEANUP_HOURS = 24
    
    # Rate limiting
    API_RATE_LIMIT_PER_MINUTE = 100
    AUTH_RATE_LIMIT_PER_MINUTE = 10
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def validate_password_strength(password: str) -> bool:
    """Validate password meets security requirements."""
    if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    return has_upper and has_lower and has_digit and has_special


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    secret_key = settings.secret_key
    if not secret_key:
        raise ValueError("SECRET_KEY not configured")
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    secret_key = settings.secret_key
    if not secret_key:
        raise ValueError("SECRET_KEY not configured")
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> dict:
    """Verify and decode JWT token."""
    try:
        secret_key = settings.secret_key
        if not secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error"
            )
        
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        
        # Verify token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}"
            )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from JWT token."""
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    return user_id


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return hash_api_key(api_key) == hashed_key


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return secrets.token_hex(16)


def sanitize_input(input_string: str, max_length: int = 255) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not isinstance(input_string, str):
        return ""
    
    # Remove potentially dangerous characters
    sanitized = input_string.strip()
    sanitized = sanitized[:max_length]  # Truncate to max length
    
    # Remove control characters except normal whitespace
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
    
    return sanitized


def validate_email(email: str) -> bool:
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Clean up old entries
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > window_start
            ]
        else:
            self.requests[key] = []
        
        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False


# Global rate limiter instance
rate_limiter = RateLimiter()


def check_rate_limit(key: str, limit: int = SecurityConfig.API_RATE_LIMIT_PER_MINUTE):
    """Dependency for rate limiting."""
    if not rate_limiter.is_allowed(key, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


def secure_headers_middleware(request, call_next):
    """Middleware to add security headers."""
    response = call_next(request)
    
    for header, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    
    return response


def mask_sensitive_data(data: dict, sensitive_fields: list = None) -> dict:
    """Mask sensitive data in dictionaries for logging."""
    if sensitive_fields is None:
        sensitive_fields = ['password', 'token', 'key', 'secret', 'auth']
    
    masked_data = data.copy()
    
    for key, value in masked_data.items():
        if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
            if isinstance(value, str) and len(value) > 4:
                masked_data[key] = value[:2] + "*" * (len(value) - 4) + value[-2:]
            else:
                masked_data[key] = "***"
    
    return masked_data


class SecurityException(Exception):
    """Custom exception for security-related errors."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)