"""
Comprehensive Error Logging System for External API Failures

Provides structured logging, error tracking, and failure analytics for all
external API integrations to improve system observability and debugging.
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from app.core.config import get_settings
from app.core.unified_cache import UnifiedCache

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    DATA_FORMAT = "data_format"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

@dataclass
class APIErrorRecord:
    """Structured record for API errors."""
    timestamp: datetime
    service: str
    endpoint: str
    method: str
    status_code: Optional[int]
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    response_time_ms: Optional[float]
    request_id: Optional[str] = None
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    context: Optional[Dict[str, Any]] = None

class APIErrorLogger:
    """Comprehensive logging system for external API failures."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = UnifiedCache("error_logs")
        
        # In-memory error tracking for real-time analytics
        self.error_buffer = defaultdict(lambda: deque(maxlen=100))
        self.error_stats = defaultdict(lambda: {
            "total_errors": 0,
            "last_error": None,
            "error_rate": 0.0,
            "categories": defaultdict(int),
            "severity_counts": defaultdict(int)
        })
        
        # Configure structured logging
        self.error_logger = logging.getLogger("api_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        
        # Add file handler if logs directory exists
        try:
            from pathlib import Path
            logs_dir = Path("logs")
            if logs_dir.exists():
                handler = logging.FileHandler("logs/api_errors.log")
                handler.setFormatter(formatter)
                self.error_logger.addHandler(handler)
        except Exception as e:
            logger.warning(f"Could not set up error log file: {e}")
    
    def log_api_error(
        self,
        service: str,
        endpoint: str,
        method: str = "GET",
        status_code: Optional[int] = None,
        error_message: str = "",
        response_time_ms: Optional[float] = None,
        response_body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        retry_count: int = 0
    ) -> str:
        """Log an API error with comprehensive details."""
        
        # Categorize and assess severity
        error_category = self._categorize_error(status_code, error_message, exception)
        severity = self._assess_severity(error_category, status_code, retry_count)
        
        # Create error record
        error_record = APIErrorRecord(
            timestamp=datetime.utcnow(),
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            error_message=error_message,
            error_category=error_category,
            severity=severity,
            response_time_ms=response_time_ms,
            headers=headers,
            response_body=response_body[:500] if response_body else None,  # Truncate long responses
            stack_trace=traceback.format_exc() if exception else None,
            retry_count=retry_count,
            context=context
        )
        
        # Generate error ID
        error_id = f"{service}_{int(datetime.utcnow().timestamp())}"
        
        # Log to structured logger
        self._log_structured_error(error_record, error_id)
        
        # Update in-memory tracking
        self._update_error_tracking(service, error_record)
        
        # Cache error for retrieval
        self._cache_error_record(error_id, error_record)
        
        # Check for critical patterns
        self._check_critical_patterns(service, error_record)
        
        return error_id
    
    def _categorize_error(
        self, 
        status_code: Optional[int], 
        error_message: str, 
        exception: Optional[Exception]
    ) -> ErrorCategory:
        """Categorize the error based on available information."""
        
        if exception:
            if "timeout" in str(exception).lower():
                return ErrorCategory.TIMEOUT
            elif "connection" in str(exception).lower():
                return ErrorCategory.NETWORK
            elif "ssl" in str(exception).lower() or "certificate" in str(exception).lower():
                return ErrorCategory.NETWORK
        
        if status_code:
            if status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif status_code == 403:
                return ErrorCategory.AUTHENTICATION
            elif status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif 400 <= status_code < 500:
                return ErrorCategory.CONFIGURATION
            elif 500 <= status_code < 600:
                return ErrorCategory.SERVER_ERROR
        
        # Check error message for keywords
        error_lower = error_message.lower()
        if any(word in error_lower for word in ["timeout", "timed out"]):
            return ErrorCategory.TIMEOUT
        elif any(word in error_lower for word in ["auth", "key", "token", "unauthorized"]):
            return ErrorCategory.AUTHENTICATION
        elif "rate limit" in error_lower or "too many requests" in error_lower:
            return ErrorCategory.RATE_LIMIT
        elif any(word in error_lower for word in ["json", "xml", "format", "parse"]):
            return ErrorCategory.DATA_FORMAT
        elif any(word in error_lower for word in ["network", "connection", "dns"]):
            return ErrorCategory.NETWORK
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(
        self, 
        category: ErrorCategory, 
        status_code: Optional[int], 
        retry_count: int
    ) -> ErrorSeverity:
        """Assess the severity of the error."""
        
        # Critical conditions
        if retry_count > 3:
            return ErrorSeverity.CRITICAL
        if category == ErrorCategory.AUTHENTICATION and status_code == 401:
            return ErrorSeverity.HIGH
        if category == ErrorCategory.SERVER_ERROR and status_code and status_code >= 500:
            return ErrorSeverity.HIGH
        
        # High severity
        if category == ErrorCategory.CONFIGURATION:
            return ErrorSeverity.HIGH
        if category == ErrorCategory.RATE_LIMIT:
            return ErrorSeverity.MEDIUM
        
        # Medium severity
        if category in [ErrorCategory.TIMEOUT, ErrorCategory.NETWORK]:
            return ErrorSeverity.MEDIUM
        if category == ErrorCategory.DATA_FORMAT:
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _log_structured_error(self, error_record: APIErrorRecord, error_id: str):
        """Log error with structured format."""
        
        log_data = {
            "error_id": error_id,
            "service": error_record.service,
            "endpoint": error_record.endpoint,
            "method": error_record.method,
            "status_code": error_record.status_code,
            "error_message": error_record.error_message,
            "category": error_record.error_category.value,
            "severity": error_record.severity.value,
            "response_time_ms": error_record.response_time_ms,
            "retry_count": error_record.retry_count,
            "timestamp": error_record.timestamp.isoformat(),
            "context": error_record.context
        }
        
        # Log at appropriate level based on severity
        if error_record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.error_logger.error(json.dumps(log_data))
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(json.dumps(log_data))
        else:
            self.error_logger.info(json.dumps(log_data))
        
        # Also log to main logger for visibility
        logger.error(
            f"API Error [{error_record.severity.value.upper()}] "
            f"{error_record.service}: {error_record.error_message} "
            f"(Status: {error_record.status_code}, Category: {error_record.error_category.value})"
        )
    
    def _update_error_tracking(self, service: str, error_record: APIErrorRecord):
        """Update in-memory error tracking for analytics."""
        
        # Add to error buffer
        self.error_buffer[service].append(error_record)
        
        # Update statistics
        stats = self.error_stats[service]
        stats["total_errors"] += 1
        stats["last_error"] = error_record.timestamp
        stats["categories"][error_record.error_category.value] += 1
        stats["severity_counts"][error_record.severity.value] += 1
        
        # Calculate error rate (errors per hour)
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        recent_errors = [
            err for err in self.error_buffer[service]
            if err.timestamp > one_hour_ago
        ]
        stats["error_rate"] = len(recent_errors)
    
    def _cache_error_record(self, error_id: str, error_record: APIErrorRecord):
        """Cache error record for later retrieval."""
        try:
            self.cache.set(
                key=f"error_{error_id}",
                value=asdict(error_record),
                source="error_logger",
                derivation_flag="logged",
                soft_ttl=3600,  # 1 hour
                hard_ttl=86400  # 24 hours
            )
        except Exception as e:
            logger.warning(f"Could not cache error record: {e}")
    
    def _check_critical_patterns(self, service: str, error_record: APIErrorRecord):
        """Check for critical error patterns that need immediate attention."""
        
        recent_errors = list(self.error_buffer[service])
        if len(recent_errors) < 3:
            return
        
        # Check for repeated failures
        if len(recent_errors) >= 5:
            last_5_errors = recent_errors[-5:]
            if all(err.error_category == error_record.error_category for err in last_5_errors):
                logger.critical(
                    f"CRITICAL PATTERN DETECTED: {service} has 5 consecutive {error_record.error_category.value} errors. "
                    f"Last error: {error_record.error_message}"
                )
        
        # Check for authentication failures
        if error_record.error_category == ErrorCategory.AUTHENTICATION:
            auth_failures = [
                err for err in recent_errors[-10:]
                if err.error_category == ErrorCategory.AUTHENTICATION
            ]
            if len(auth_failures) >= 3:
                logger.critical(
                    f"CRITICAL: {service} has multiple authentication failures. "
                    f"Check API credentials immediately."
                )
    
    def get_error_analytics(self, service: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive error analytics."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        analytics = {
            "period_hours": hours,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "overall_stats": {
                "total_errors": 0,
                "unique_services": 0,
                "error_categories": defaultdict(int),
                "severity_distribution": defaultdict(int),
                "top_error_services": []
            }
        }
        
        # Analyze specific service or all services
        services_to_analyze = [service] if service else list(self.error_stats.keys())
        
        for svc in services_to_analyze:
            if svc not in self.error_buffer:
                continue
            
            # Filter errors by time period
            relevant_errors = [
                err for err in self.error_buffer[svc]
                if err.timestamp > cutoff_time
            ]
            
            if not relevant_errors:
                continue
            
            # Calculate service-specific analytics
            service_analytics = {
                "error_count": len(relevant_errors),
                "error_rate_per_hour": len(relevant_errors) / hours,
                "last_error": max(err.timestamp for err in relevant_errors).isoformat(),
                "categories": defaultdict(int),
                "severities": defaultdict(int),
                "common_errors": {},
                "average_response_time": None,
                "retry_patterns": defaultdict(int)
            }
            
            response_times = []
            error_messages = defaultdict(int)
            
            for err in relevant_errors:
                service_analytics["categories"][err.error_category.value] += 1
                service_analytics["severities"][err.severity.value] += 1
                service_analytics["retry_patterns"][err.retry_count] += 1
                
                if err.response_time_ms:
                    response_times.append(err.response_time_ms)
                
                # Track common error messages
                error_key = f"{err.status_code}_{err.error_message[:50]}"
                error_messages[error_key] += 1
                
                # Update overall stats
                analytics["overall_stats"]["total_errors"] += 1
                analytics["overall_stats"]["error_categories"][err.error_category.value] += 1
                analytics["overall_stats"]["severity_distribution"][err.severity.value] += 1
            
            if response_times:
                service_analytics["average_response_time"] = sum(response_times) / len(response_times)
            
            # Top 3 most common errors
            service_analytics["common_errors"] = dict(
                sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:3]
            )
            
            analytics["services"][svc] = service_analytics
        
        # Overall statistics
        analytics["overall_stats"]["unique_services"] = len(analytics["services"])
        
        # Top error services
        service_error_counts = [
            (svc, data["error_count"]) 
            for svc, data in analytics["services"].items()
        ]
        analytics["overall_stats"]["top_error_services"] = sorted(
            service_error_counts, key=lambda x: x[1], reverse=True
        )[:5]
        
        return analytics
    
    def get_service_health_score(self, service: str, hours: int = 24) -> float:
        """Calculate a health score (0-100) for a service based on recent errors."""
        
        if service not in self.error_buffer:
            return 100.0  # No errors recorded
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            err for err in self.error_buffer[service]
            if err.timestamp > cutoff_time
        ]
        
        if not recent_errors:
            return 100.0
        
        # Score calculation based on:
        # - Number of errors
        # - Severity of errors
        # - Error categories
        
        base_score = 100.0
        
        # Deduct points for each error (more points for severe errors)
        for err in recent_errors:
            if err.severity == ErrorSeverity.CRITICAL:
                base_score -= 15
            elif err.severity == ErrorSeverity.HIGH:
                base_score -= 10
            elif err.severity == ErrorSeverity.MEDIUM:
                base_score -= 5
            else:
                base_score -= 2
        
        # Additional deductions for problematic patterns
        auth_errors = sum(1 for err in recent_errors if err.error_category == ErrorCategory.AUTHENTICATION)
        if auth_errors > 0:
            base_score -= auth_errors * 5  # Authentication issues are serious
        
        rate_limit_errors = sum(1 for err in recent_errors if err.error_category == ErrorCategory.RATE_LIMIT)
        if rate_limit_errors > 3:
            base_score -= 10  # Repeated rate limiting
        
        return max(0.0, base_score)
    
    def clear_old_errors(self, days: int = 7):
        """Clear old error records to prevent memory bloat."""
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        for service in list(self.error_buffer.keys()):
            # Filter out old errors
            recent_errors = [
                err for err in self.error_buffer[service]
                if err.timestamp > cutoff_time
            ]
            
            # Update buffer
            self.error_buffer[service] = deque(recent_errors, maxlen=100)
            
            # Reset stats if no recent errors
            if not recent_errors:
                if service in self.error_stats:
                    del self.error_stats[service]
        
        logger.info(f"Cleared error records older than {days} days")

# Global instance
error_logger = APIErrorLogger()