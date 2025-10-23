import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import structlog

# Context variable for correlation ID tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to log entries."""
    corr_id = correlation_id.get()
    if corr_id:
        event_dict['correlation_id'] = corr_id
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to log entries."""
    event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    return event_dict


def add_service_info(logger, method_name, event_dict):
    """Add service information to log entries."""
    event_dict['service'] = 'riskx-backend'
    event_dict['version'] = '1.0.0'
    return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': 'riskx-backend',
            'version': '1.0.0'
        }
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_entry['correlation_id'] = corr_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(message)s',
        stream=sys.stdout,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set up JSON formatter
    json_formatter = JSONFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(json_formatter)
    
    # Configure structlog
    structlog.configure(
        processors=[
            add_correlation_id,
            add_timestamp,
            add_service_info,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggingContext:
    """Context manager for adding correlation ID to logs."""
    
    def __init__(self, corr_id: str):
        self.corr_id = corr_id
        self.token = None
    
    def __enter__(self):
        self.token = correlation_id.set(self.corr_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id.reset(self.token)


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """Log API request with structured data."""
    logger = get_logger("api")
    logger.info(
        "API request completed",
        method=method,
        path=path,
        status_code=status_code,
        response_time_ms=response_time_ms,
        user_id=user_id,
        ip_address=ip_address
    )


def log_cache_operation(
    operation: str,
    cache_key: str,
    hit: bool,
    response_time_ms: float,
    cache_tier: str = "unknown"
) -> None:
    """Log cache operation with structured data."""
    logger = get_logger("cache")
    logger.info(
        "Cache operation",
        operation=operation,
        cache_key=cache_key,
        hit=hit,
        response_time_ms=response_time_ms,
        cache_tier=cache_tier
    )


def log_external_api_call(
    api_name: str,
    endpoint: str,
    status_code: Optional[int],
    response_time_ms: float,
    success: bool,
    error_message: Optional[str] = None
) -> None:
    """Log external API call with structured data."""
    logger = get_logger("external_api")
    
    if success:
        logger.info(
            "External API call successful",
            api_name=api_name,
            endpoint=endpoint,
            status_code=status_code,
            response_time_ms=response_time_ms
        )
    else:
        logger.error(
            "External API call failed",
            api_name=api_name,
            endpoint=endpoint,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error_message=error_message
        )


def log_business_event(
    event_type: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None
) -> None:
    """Log business events with structured data."""
    logger = get_logger("business")
    logger.info(
        "Business event",
        event_type=event_type,
        event_data=event_data,
        user_id=user_id
    )


def log_security_event(
    event_type: str,
    severity: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """Log security events with structured data."""
    logger = get_logger("security")
    
    log_method = getattr(logger, severity.lower(), logger.info)
    log_method(
        "Security event",
        event_type=event_type,
        details=details,
        user_id=user_id,
        ip_address=ip_address
    )


def log_database_operation(
    operation: str,
    table: str,
    records_affected: int,
    execution_time_ms: float,
    success: bool,
    error_message: Optional[str] = None
) -> None:
    """Log database operations with structured data."""
    logger = get_logger("database")
    
    if success:
        logger.info(
            "Database operation completed",
            operation=operation,
            table=table,
            records_affected=records_affected,
            execution_time_ms=execution_time_ms
        )
    else:
        logger.error(
            "Database operation failed",
            operation=operation,
            table=table,
            execution_time_ms=execution_time_ms,
            error_message=error_message
        )