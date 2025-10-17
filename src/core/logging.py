"""
Logging configuration for RiskX platform.
Provides structured logging with security and performance monitoring.
"""

import os
import sys
import logging
import logging.handlers
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from contextlib import contextmanager

from .config import get_settings


class SecurityFilter(logging.Filter):
    """Filter to prevent logging of sensitive information."""
    
    SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'api_key', 'access_key', 
        'private_key', 'credential', 'auth', 'session_id'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from log records."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Mask sensitive data in log messages
            record.msg = self._mask_sensitive_data(record.msg)
        
        if hasattr(record, 'args') and record.args:
            # Mask sensitive data in log arguments
            record.args = tuple(
                self._mask_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        return True
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text."""
        import re
        
        # Mask potential API keys, tokens, etc.
        patterns = [
            (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_-]{10,})', r'\1***masked***'),
            (r'(token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_-]{10,})', r'\1***masked***'),
            (r'(password["\']?\s*[:=]\s*["\']?)([^\s"\']{3,})', r'\1***masked***'),
            (r'(secret["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_-]{10,})', r'\1***masked***'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger('riskx.performance')
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       duration_ms: float, user_id: Optional[str] = None):
        """Log API request performance metrics."""
        self.logger.info(
            "API request completed",
            extra={
                'operation': 'api_request',
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration': duration_ms,
                'user_id': user_id
            }
        )
    
    def log_data_fetch(self, source: str, duration_ms: float, 
                      record_count: int, cache_hit: bool):
        """Log data fetch performance metrics."""
        self.logger.info(
            "Data fetch completed",
            extra={
                'operation': 'data_fetch',
                'source': source,
                'duration': duration_ms,
                'record_count': record_count,
                'cache_hit': cache_hit
            }
        )
    
    def log_model_inference(self, model_name: str, duration_ms: float, 
                           input_size: int, prediction_count: int):
        """Log ML model inference performance."""
        self.logger.info(
            "Model inference completed",
            extra={
                'operation': 'model_inference',
                'model_name': model_name,
                'duration': duration_ms,
                'input_size': input_size,
                'prediction_count': prediction_count
            }
        )


class SecurityLogger:
    """Logger for security events and audit trails."""
    
    def __init__(self):
        self.logger = logging.getLogger('riskx.security')
    
    def log_authentication_attempt(self, user_id: str, success: bool, 
                                 ip_address: str, user_agent: str):
        """Log authentication attempts."""
        level = logging.INFO if success else logging.WARNING
        message = "Authentication successful" if success else "Authentication failed"
        
        self.logger.log(
            level,
            message,
            extra={
                'operation': 'authentication',
                'user_id': user_id,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent
            }
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, 
                                action: str, ip_address: str):
        """Log authorization failures."""
        self.logger.warning(
            "Authorization denied",
            extra={
                'operation': 'authorization_denied',
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'ip_address': ip_address
            }
        )
    
    def log_suspicious_activity(self, user_id: str, activity: str, 
                              details: Dict[str, Any], severity: str = 'medium'):
        """Log suspicious activities."""
        level = logging.ERROR if severity == 'high' else logging.WARNING
        
        self.logger.log(
            level,
            f"Suspicious activity detected: {activity}",
            extra={
                'operation': 'suspicious_activity',
                'user_id': user_id,
                'activity': activity,
                'severity': severity,
                'details': details
            }
        )
    
    def log_data_access(self, user_id: str, data_type: str, 
                       record_count: int, operation: str):
        """Log data access for audit trails."""
        self.logger.info(
            "Data access logged",
            extra={
                'operation': 'data_access',
                'user_id': user_id,
                'data_type': data_type,
                'record_count': record_count,
                'access_operation': operation
            }
        )


def setup_logging(settings=None) -> Dict[str, logging.Logger]:
    """Set up logging configuration for RiskX platform."""
    if settings is None:
        settings = get_settings()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different log types
    for subdir in ["app", "api", "etl", "ml", "security"]:
        (log_dir / subdir).mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if settings.LOG_FORMAT == "json":
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)
    
    # File handlers for different log types
    loggers = {}
    
    # Application logger
    app_logger = logging.getLogger('riskx.app')
    app_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app" / "application.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    app_handler.setFormatter(StructuredFormatter())
    app_handler.addFilter(SecurityFilter())
    app_logger.addHandler(app_handler)
    loggers['app'] = app_logger
    
    # API logger
    api_logger = logging.getLogger('riskx.api')
    api_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api" / "api.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    api_handler.setFormatter(StructuredFormatter())
    api_handler.addFilter(SecurityFilter())
    api_logger.addHandler(api_handler)
    loggers['api'] = api_logger
    
    # ETL logger
    etl_logger = logging.getLogger('riskx.etl')
    etl_handler = logging.handlers.RotatingFileHandler(
        log_dir / "etl" / "etl.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    etl_handler.setFormatter(StructuredFormatter())
    etl_logger.addHandler(etl_handler)
    loggers['etl'] = etl_logger
    
    # ML logger
    ml_logger = logging.getLogger('riskx.ml')
    ml_handler = logging.handlers.RotatingFileHandler(
        log_dir / "ml" / "ml.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    ml_handler.setFormatter(StructuredFormatter())
    ml_logger.addHandler(ml_handler)
    loggers['ml'] = ml_logger
    
    # Security logger with special handling
    security_logger = logging.getLogger('riskx.security')
    security_handler = logging.handlers.RotatingFileHandler(
        log_dir / "security" / "security.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=20  # Keep more security logs
    )
    security_handler.setFormatter(StructuredFormatter())
    # Note: No SecurityFilter for security logger to preserve all details
    security_logger.addHandler(security_handler)
    loggers['security'] = security_logger
    
    # Performance logger
    performance_logger = logging.getLogger('riskx.performance')
    performance_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app" / "performance.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    performance_handler.setFormatter(StructuredFormatter())
    performance_logger.addHandler(performance_handler)
    loggers['performance'] = performance_logger
    
    return loggers


@contextmanager
def log_performance(operation: str, logger: Optional[logging.Logger] = None):
    """Context manager for logging operation performance."""
    import time
    
    if logger is None:
        logger = logging.getLogger('riskx.performance')
    
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Operation '{operation}' completed",
            extra={
                'operation': operation,
                'duration': duration_ms
            }
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration."""
    return logging.getLogger(f'riskx.{name}')


def log_exception(logger: logging.Logger, message: str, **kwargs):
    """Log an exception with full context."""
    logger.exception(message, extra=kwargs)


# Initialize logging on module import
_loggers = None

def init_logging():
    """Initialize logging system."""
    global _loggers
    if _loggers is None:
        _loggers = setup_logging()
    return _loggers


# Convenience instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()