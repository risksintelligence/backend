"""
Custom exceptions for RiskX platform.
Provides structured error handling with proper logging and user-friendly messages.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATA_ACCESS = "data_access"
    EXTERNAL_API = "external_api"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskXBaseException(Exception):
    """Base exception class for all RiskX-specific exceptions."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or self._generate_error_code()
        self.details = details or {}
        self.user_message = user_message or self._get_user_friendly_message()
        
        # Log the exception
        self._log_exception()
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception type and category."""
        class_name = self.__class__.__name__
        return f"{self.category.value.upper()}_{class_name.upper()}"
    
    def _get_user_friendly_message(self) -> str:
        """Get user-friendly error message."""
        return "An error occurred while processing your request. Please try again later."
    
    def _log_exception(self):
        """Log the exception with appropriate level based on severity."""
        logger = logging.getLogger('riskx.exceptions')
        
        log_data = {
            'error_code': self.error_code,
            'category': self.category.value,
            'severity': self.severity.value,
            'details': self.details
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(self.message, extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(self.message, extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(self.message, extra=log_data)
        else:
            logger.info(self.message, extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error_code': self.error_code,
            'message': self.user_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'details': self.details
        }


# Authentication and Authorization Exceptions

class AuthenticationError(RiskXBaseException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            user_message="Invalid credentials. Please check your username and password.",
            **kwargs
        )


class TokenExpiredError(AuthenticationError):
    """Raised when authentication token has expired."""
    
    def __init__(self, message: str = "Authentication token has expired", **kwargs):
        super().__init__(
            message,
            user_message="Your session has expired. Please log in again.",
            **kwargs
        )


class AuthorizationError(RiskXBaseException):
    """Raised when user lacks sufficient permissions."""
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            user_message="You don't have permission to access this resource.",
            **kwargs
        )


class RateLimitExceededError(RiskXBaseException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after_seconds'] = retry_after
        
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message=f"Too many requests. Please try again in {retry_after or 'a few'} seconds.",
            **kwargs
        )


# Validation Exceptions

class ValidationError(RiskXBaseException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, List[str]]] = None, **kwargs):
        details = kwargs.get('details', {})
        if field_errors:
            details['field_errors'] = field_errors
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            user_message="Please check your input and try again.",
            **kwargs
        )


class InvalidInputError(ValidationError):
    """Raised when input data is invalid."""
    
    def __init__(self, field_name: str, value: Any, expected_format: str, **kwargs):
        message = f"Invalid value for field '{field_name}': {value}"
        super().__init__(
            message,
            field_errors={field_name: [f"Expected {expected_format}"]},
            user_message=f"Invalid value for {field_name}. Expected {expected_format}.",
            **kwargs
        )


class MissingRequiredFieldError(ValidationError):
    """Raised when required field is missing."""
    
    def __init__(self, field_name: str, **kwargs):
        message = f"Missing required field: {field_name}"
        super().__init__(
            message,
            field_errors={field_name: ["This field is required"]},
            user_message=f"The field '{field_name}' is required.",
            **kwargs
        )


# Data Access Exceptions

class DataAccessError(RiskXBaseException):
    """Raised when data access fails."""
    
    def __init__(self, message: str = "Data access failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_ACCESS,
            severity=ErrorSeverity.HIGH,
            user_message="Unable to access data. Please try again later.",
            **kwargs
        )


class DatabaseError(DataAccessError):
    """Raised when database operation fails."""
    
    def __init__(self, message: str = "Database operation failed", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            user_message="Database operation failed. Please try again later.",
            **kwargs
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            user_message="Database is temporarily unavailable. Please try again later.",
            **kwargs
        )


class CacheError(DataAccessError):
    """Raised when cache operation fails."""
    
    def __init__(self, message: str = "Cache operation failed", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            user_message="Temporary data access issue. Please try again.",
            **kwargs
        )


class DataNotFoundError(DataAccessError):
    """Raised when requested data is not found."""
    
    def __init__(self, resource: str, identifier: str = None, **kwargs):
        message = f"{resource} not found"
        if identifier:
            message += f" with identifier: {identifier}"
        
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            details={'resource': resource, 'identifier': identifier},
            user_message=f"The requested {resource.lower()} was not found.",
            **kwargs
        )


class StorageError(DataAccessError):
    """Raised when storage operation fails."""
    
    def __init__(self, message: str = "Storage operation failed", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            user_message="Storage operation failed. Please try again later.",
            **kwargs
        )


# External API Exceptions

class ExternalAPIError(RiskXBaseException):
    """Raised when external API call fails."""
    
    def __init__(self, service_name: str, message: str = None, status_code: int = None, **kwargs):
        message = message or f"External API call to {service_name} failed"
        details = kwargs.get('details', {})
        details.update({'service_name': service_name})
        if status_code:
            details['status_code'] = status_code
        
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message=f"Unable to retrieve data from {service_name}. Using cached data if available.",
            **kwargs
        )


class APIRateLimitError(ExternalAPIError):
    """Raised when external API rate limit is exceeded."""
    
    def __init__(self, service_name: str, retry_after: Optional[int] = None, **kwargs):
        message = f"Rate limit exceeded for {service_name}"
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after_seconds'] = retry_after
        
        super().__init__(
            service_name,
            message,
            details=details,
            user_message=f"Data service is temporarily busy. Retrying automatically.",
            **kwargs
        )


class APITimeoutError(ExternalAPIError):
    """Raised when external API call times out."""
    
    def __init__(self, service_name: str, timeout_seconds: float, **kwargs):
        message = f"Timeout after {timeout_seconds}s calling {service_name}"
        super().__init__(
            service_name,
            message,
            details={'timeout_seconds': timeout_seconds},
            user_message=f"Data service is responding slowly. Using cached data.",
            **kwargs
        )


# Data Source Specific Exceptions

class DataSourceError(ExternalAPIError):
    """Raised when data source operation fails."""
    
    def __init__(self, source_name: str, operation: str, message: str = None, **kwargs):
        message = message or f"Data source {source_name} failed during {operation}"
        super().__init__(
            source_name,
            message,
            details={'operation': operation},
            user_message=f"Unable to retrieve data from {source_name}. Using cached data if available.",
            **kwargs
        )


class APIError(ExternalAPIError):
    """Generic API error - alias for ExternalAPIError."""
    pass


# Processing Exceptions

class ProcessingError(RiskXBaseException):
    """Raised when data processing fails."""
    
    def __init__(self, message: str = "Data processing failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            user_message="Unable to process data. Please try again later.",
            **kwargs
        )


class ModelError(ProcessingError):
    """Raised when ML model operation fails."""
    
    def __init__(self, model_name: str, operation: str, message: str = None, **kwargs):
        message = message or f"Model {model_name} failed during {operation}"
        super().__init__(
            message,
            details={'model_name': model_name, 'operation': operation},
            user_message="Unable to generate predictions. Please try again later.",
            **kwargs
        )


class TransformationError(ProcessingError):
    """Raised when data transformation fails."""
    
    def __init__(self, transformation: str, input_data_type: str, **kwargs):
        message = f"Failed to transform {input_data_type} using {transformation}"
        super().__init__(
            message,
            details={'transformation': transformation, 'input_data_type': input_data_type},
            user_message="Unable to process data format. Please check your input.",
            **kwargs
        )


# System Exceptions

class SystemError(RiskXBaseException):
    """Raised when system-level error occurs."""
    
    def __init__(self, message: str = "System error occurred", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_message="A system error occurred. Please contact support if this persists.",
            **kwargs
        )


# Configuration Exceptions

class ConfigurationError(RiskXBaseException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            user_message="System configuration issue. Please contact support.",
            **kwargs
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, **kwargs):
        message = f"Missing required configuration: {config_key}"
        super().__init__(
            message,
            details={'config_key': config_key},
            **kwargs
        )


# Business Logic Exceptions

class BusinessLogicError(RiskXBaseException):
    """Raised when business logic validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            user_message=message,  # Business logic errors are usually user-facing
            **kwargs
        )


class InvalidRiskScoreError(BusinessLogicError):
    """Raised when risk score calculation is invalid."""
    
    def __init__(self, score: float, valid_range: tuple, **kwargs):
        message = f"Invalid risk score {score}. Must be between {valid_range[0]} and {valid_range[1]}"
        super().__init__(
            message,
            details={'score': score, 'valid_range': valid_range},
            **kwargs
        )


class InsufficientDataError(BusinessLogicError):
    """Raised when insufficient data for reliable calculation."""
    
    def __init__(self, required_data_points: int, available_data_points: int, **kwargs):
        message = f"Insufficient data: need {required_data_points}, have {available_data_points}"
        super().__init__(
            message,
            details={
                'required_data_points': required_data_points,
                'available_data_points': available_data_points
            },
            user_message="Insufficient data available for reliable calculation.",
            **kwargs
        )


# Utility Functions

def handle_exception(exc: Exception) -> RiskXBaseException:
    """Convert standard exceptions to RiskX exceptions."""
    if isinstance(exc, RiskXBaseException):
        return exc
    
    # Map common exceptions to RiskX exceptions
    exception_mappings = {
        ConnectionError: DatabaseConnectionError,
        TimeoutError: APITimeoutError,
        ValueError: ValidationError,
        KeyError: DataNotFoundError,
        FileNotFoundError: DataNotFoundError,
        PermissionError: AuthorizationError,
    }
    
    for exc_type, riskx_exc_type in exception_mappings.items():
        if isinstance(exc, exc_type):
            return riskx_exc_type(str(exc))
    
    # Default to generic system error
    return RiskXBaseException(
        str(exc),
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.HIGH,
        details={'original_exception': exc.__class__.__name__}
    )


def create_error_response(exc: RiskXBaseException, include_details: bool = False) -> Dict[str, Any]:
    """Create standardized error response for API endpoints."""
    response = {
        'error': True,
        'error_code': exc.error_code,
        'message': exc.user_message,
        'category': exc.category.value,
        'severity': exc.severity.value
    }
    
    if include_details and exc.details:
        response['details'] = exc.details
    
    return response