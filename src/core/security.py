"""
Security utilities for RiskX platform.
Provides authentication, authorization, input validation, and security monitoring.
"""

import os
import hashlib
import secrets
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from functools import wraps
import re
import html
import bleach
from urllib.parse import urlparse
import logging

from .config import get_settings
from .logging import security_logger

logger = logging.getLogger('riskx.security')


class SecurityConfig:
    """Security configuration and constants."""
    
    # Password requirements
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    
    # JWT settings
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Rate limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_LOCKOUT_MINUTES = 15
    API_RATE_LIMIT_PER_MINUTE = 100
    
    # Input validation
    MAX_INPUT_LENGTH = 10000
    ALLOWED_HTML_TAGS = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
    
    # File upload security
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_EXTENSIONS = {'.json', '.csv', '.txt', '.pdf', '.png', '.jpg'}


class InputValidator:
    """Input validation and sanitization utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password strength."""
        validation = {
            'length': len(password) >= SecurityConfig.MIN_PASSWORD_LENGTH,
            'uppercase': bool(re.search(r'[A-Z]', password)) if SecurityConfig.REQUIRE_UPPERCASE else True,
            'lowercase': bool(re.search(r'[a-z]', password)) if SecurityConfig.REQUIRE_LOWERCASE else True,
            'digits': bool(re.search(r'\d', password)) if SecurityConfig.REQUIRE_DIGITS else True,
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)) if SecurityConfig.REQUIRE_SPECIAL_CHARS else True,
        }
        validation['valid'] = all(validation.values())
        return validation
    
    @staticmethod
    def sanitize_input(data: str, max_length: Optional[int] = None) -> str:
        """Sanitize user input to prevent XSS and injection attacks."""
        if max_length is None:
            max_length = SecurityConfig.MAX_INPUT_LENGTH
        
        # Limit length
        data = data[:max_length]
        
        # HTML escape
        data = html.escape(data)
        
        # Remove potentially dangerous characters
        data = re.sub(r'[<>"\']', '', data)
        
        return data.strip()
    
    @staticmethod
    def sanitize_html(html_content: str) -> str:
        """Sanitize HTML content using whitelist approach."""
        return bleach.clean(
            html_content,
            tags=SecurityConfig.ALLOWED_HTML_TAGS,
            strip=True
        )
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format and safety."""
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                not any(malicious in url.lower() for malicious in ['javascript:', 'data:', 'vbscript:'])
            ])
        except Exception:
            return False
    
    @staticmethod
    def validate_file_upload(filename: str, file_size: int) -> Dict[str, Any]:
        """Validate file upload safety."""
        validation = {
            'valid': True,
            'errors': []
        }
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SecurityConfig.ALLOWED_FILE_EXTENSIONS:
            validation['valid'] = False
            validation['errors'].append(f"File extension {file_ext} not allowed")
        
        # Check file size
        if file_size > SecurityConfig.MAX_FILE_SIZE:
            validation['valid'] = False
            validation['errors'].append(f"File size {file_size} exceeds maximum allowed size")
        
        # Check filename for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            validation['valid'] = False
            validation['errors'].append("Invalid filename contains path traversal characters")
        
        return validation


class PasswordManager:
    """Password hashing and verification utilities."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using secure methods."""
        import bcrypt
        
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)


class JWTManager:
    """JWT token management for authentication."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.settings = get_settings()
        self.secret_key = secret_key or self.settings.secret_key
    
    def create_access_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Create JWT access token."""
        if permissions is None:
            permissions = []
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'type': 'access',
            'exp': datetime.utcnow() + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for token invalidation
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        payload = {
            'user_id': user_id,
            'type': 'refresh',
            'exp': datetime.utcnow() + timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def verify_token(self, token: str, token_type: str = 'access') -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[SecurityConfig.JWT_ALGORITHM])
            
            if payload.get('type') != token_type:
                logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None


class RateLimiter:
    """Rate limiting implementation for API endpoints."""
    
    def __init__(self):
        self.attempts = {}  # In production, use Redis or database
    
    def is_rate_limited(self, identifier: str, max_attempts: int, window_minutes: int) -> bool:
        """Check if identifier is rate limited."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Clean old attempts
        if identifier in self.attempts:
            self.attempts[identifier] = [
                timestamp for timestamp in self.attempts[identifier]
                if timestamp > window_start
            ]
        else:
            self.attempts[identifier] = []
        
        # Check if rate limited
        if len(self.attempts[identifier]) >= max_attempts:
            return True
        
        # Record current attempt
        self.attempts[identifier].append(current_time)
        return False
    
    def get_reset_time(self, identifier: str, window_minutes: int) -> Optional[datetime]:
        """Get when rate limit will reset."""
        if identifier not in self.attempts or not self.attempts[identifier]:
            return None
        
        oldest_attempt = min(self.attempts[identifier])
        reset_time = datetime.fromtimestamp(oldest_attempt + (window_minutes * 60))
        return reset_time


class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.jwt_manager = JWTManager()
    
    def check_rate_limit(self, identifier: str, endpoint: str) -> bool:
        """Check rate limit for request."""
        # Different limits for different endpoints
        if endpoint.startswith('/api/auth'):
            max_attempts = SecurityConfig.MAX_LOGIN_ATTEMPTS
            window_minutes = SecurityConfig.LOGIN_LOCKOUT_MINUTES
        else:
            max_attempts = SecurityConfig.API_RATE_LIMIT_PER_MINUTE
            window_minutes = 1
        
        is_limited = self.rate_limiter.is_rate_limited(identifier, max_attempts, window_minutes)
        
        if is_limited:
            security_logger.log_suspicious_activity(
                identifier, 
                'rate_limit_exceeded',
                {'endpoint': endpoint, 'max_attempts': max_attempts}
            )
        
        return not is_limited
    
    def validate_authentication(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token."""
        payload = self.jwt_manager.verify_token(token)
        
        if payload:
            security_logger.log_data_access(
                payload['user_id'],
                'api_access',
                1,
                'authenticated_request'
            )
        
        return payload


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }


def require_authentication(f):
    """Decorator to require authentication for API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # This would be implemented with your web framework
        # Example implementation for FastAPI or Flask
        pass
    return decorated_function


def require_permission(permission: str):
    """Decorator to require specific permission for API endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Permission check implementation
            pass
        return decorated_function
    return decorator


def audit_log(operation: str):
    """Decorator to log operations for audit trail."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                security_logger.logger.info(
                    f"Operation {operation} completed successfully",
                    extra={
                        'operation': operation,
                        'duration': duration,
                        'success': True
                    }
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                security_logger.logger.error(
                    f"Operation {operation} failed: {e}",
                    extra={
                        'operation': operation,
                        'duration': duration,
                        'success': False,
                        'error': str(e)
                    }
                )
                raise
        return decorated_function
    return decorator


class SecurityMonitor:
    """Security monitoring and anomaly detection."""
    
    def __init__(self):
        self.failed_attempts = {}  # In production, use Redis or database
    
    def monitor_failed_login(self, identifier: str, ip_address: str):
        """Monitor failed login attempts."""
        current_time = time.time()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append({
            'timestamp': current_time,
            'ip_address': ip_address
        })
        
        # Check for suspicious patterns
        recent_attempts = [
            attempt for attempt in self.failed_attempts[identifier]
            if current_time - attempt['timestamp'] < 300  # Last 5 minutes
        ]
        
        if len(recent_attempts) >= 3:
            security_logger.log_suspicious_activity(
                identifier,
                'multiple_failed_logins',
                {
                    'attempts_count': len(recent_attempts),
                    'ip_addresses': list(set(attempt['ip_address'] for attempt in recent_attempts))
                },
                'high'
            )
    
    def detect_unusual_access_pattern(self, user_id: str, resource: str, 
                                    access_count: int, time_window: int):
        """Detect unusual access patterns."""
        # Simple anomaly detection - in production, use more sophisticated methods
        if access_count > 50 and time_window < 60:  # 50+ accesses in under 1 minute
            security_logger.log_suspicious_activity(
                user_id,
                'unusual_access_pattern',
                {
                    'resource': resource,
                    'access_count': access_count,
                    'time_window_seconds': time_window
                },
                'medium'
            )


# Global instances
input_validator = InputValidator()
password_manager = PasswordManager()
security_middleware = SecurityMiddleware()
security_monitor = SecurityMonitor()