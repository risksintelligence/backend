"""
Standardized error handling for GRII Intelligence Core APIs.
Implements Bloomberg-grade error responses with consistent schema.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import uuid

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Individual error detail following RFC 7807 Problem Details specification."""
    type: str = Field(description="Error type identifier")
    title: str = Field(description="Human-readable summary")
    detail: str = Field(description="Specific error message")
    instance: Optional[str] = Field(default=None, description="Request instance identifier")


class RRIOError(BaseModel):
    """Standardized RRIO API error response schema."""
    error: bool = Field(default=True, description="Always true for error responses")
    status: int = Field(description="HTTP status code")
    code: str = Field(description="Internal error code")
    message: str = Field(description="Human-readable error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    documentation: Optional[str] = Field(default=None, description="Link to documentation")
    support: Dict[str, str] = Field(
        default_factory=lambda: {
            "contact": "support@riskxobservatory.com",
            "docs": "https://docs.riskxobservatory.com/api/errors"
        }
    )


class RRIOAPIError(HTTPException):
    """Custom exception class for standardized RRIO API errors."""
    
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        documentation: Optional[str] = None
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        self.documentation = documentation
        super().__init__(status_code=status_code, detail=message)


# Standard error codes following internal conventions
class ErrorCodes:
    # Data and Validation Errors (400-499)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DATA_FORMAT = "INVALID_DATA_FORMAT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    
    # Authentication and Authorization (401, 403)
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    ACCESS_DENIED = "ACCESS_DENIED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Resource Errors (404, 409)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    COMPONENT_NOT_FOUND = "COMPONENT_NOT_FOUND"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    
    # Rate Limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Server Errors (500-599)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    ML_MODEL_ERROR = "ML_MODEL_ERROR"
    DATA_SOURCE_ERROR = "DATA_SOURCE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # GRII-specific errors
    GERI_COMPUTATION_ERROR = "GERI_COMPUTATION_ERROR"
    STALE_ML_MODEL = "STALE_ML_MODEL"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    REGIME_CLASSIFICATION_ERROR = "REGIME_CLASSIFICATION_ERROR"
    FORECAST_ERROR = "FORECAST_ERROR"
    ANOMALY_DETECTION_ERROR = "ANOMALY_DETECTION_ERROR"


def create_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[List[ErrorDetail]] = None,
    documentation: Optional[str] = None,
    request: Optional[Request] = None
) -> JSONResponse:
    """Create standardized error response."""
    
    error_response = RRIOError(
        status=status_code,
        code=code,
        message=message,
        details=details or [],
        documentation=documentation
    )
    
    # Add request-specific information if available
    if request:
        error_response.request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    # Log error for monitoring
    logger.error(
        f"API Error: {code} - {message}",
        extra={
            "status_code": status_code,
            "error_code": code,
            "request_id": error_response.request_id,
            "details": [detail.dict() for detail in (details or [])]
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


def validation_error_handler(exc_info: Any) -> List[ErrorDetail]:
    """Convert Pydantic validation errors to standardized format."""
    details = []
    
    if hasattr(exc_info, 'errors'):
        for error in exc_info.errors():
            field_path = " -> ".join(str(loc) for loc in error.get('loc', []))
            detail = ErrorDetail(
                type=error.get('type', 'validation_error'),
                title=f"Validation Error in {field_path}",
                detail=error.get('msg', 'Invalid value'),
                instance=field_path
            )
            details.append(detail)
    
    return details


# Standard error response factories
def not_found_error(resource: str, identifier: str = None) -> RRIOAPIError:
    """Create standardized 404 error."""
    message = f"{resource} not found"
    if identifier:
        message += f": {identifier}"
    
    return RRIOAPIError(
        status_code=404,
        code=ErrorCodes.RESOURCE_NOT_FOUND,
        message=message,
        documentation="https://docs.riskxobservatory.com/api/errors#not-found"
    )


def validation_error(message: str, details: List[ErrorDetail] = None) -> RRIOAPIError:
    """Create standardized 422 validation error."""
    return RRIOAPIError(
        status_code=422,
        code=ErrorCodes.VALIDATION_ERROR,
        message=message,
        details=details or [],
        documentation="https://docs.riskxobservatory.com/api/errors#validation"
    )


def server_error(message: str, code: str = None) -> RRIOAPIError:
    """Create standardized 500 server error."""
    return RRIOAPIError(
        status_code=500,
        code=code or ErrorCodes.INTERNAL_SERVER_ERROR,
        message=message,
        documentation="https://docs.riskxobservatory.com/api/errors#server-errors"
    )


def ml_model_error(model_name: str, issue: str) -> RRIOAPIError:
    """Create standardized ML model error."""
    return RRIOAPIError(
        status_code=503,
        code=ErrorCodes.ML_MODEL_ERROR,
        message=f"ML model '{model_name}' error: {issue}",
        details=[
            ErrorDetail(
                type="ml_model_issue",
                title="Machine Learning Model Error",
                detail=f"The {model_name} model encountered an issue: {issue}",
                instance=model_name
            )
        ],
        documentation="https://docs.riskxobservatory.com/api/errors#ml-errors"
    )


def insufficient_data_error(component: str, required: int, available: int) -> RRIOAPIError:
    """Create standardized insufficient data error."""
    return RRIOAPIError(
        status_code=422,
        code=ErrorCodes.INSUFFICIENT_DATA,
        message=f"Insufficient data for {component}",
        details=[
            ErrorDetail(
                type="data_insufficiency",
                title="Insufficient Data Points",
                detail=f"Component '{component}' requires {required} data points but only {available} are available",
                instance=component
            )
        ],
        documentation="https://docs.riskxobservatory.com/api/errors#insufficient-data"
    )


def stale_model_error(model_name: str, age_hours: float, max_hours: int) -> RRIOAPIError:
    """Create standardized stale model error."""
    return RRIOAPIError(
        status_code=503,
        code=ErrorCodes.STALE_ML_MODEL,
        message=f"ML model '{model_name}' is stale",
        details=[
            ErrorDetail(
                type="model_staleness",
                title="Stale Machine Learning Model",
                detail=f"Model '{model_name}' is {age_hours:.1f}h old (max: {max_hours}h)",
                instance=model_name
            )
        ],
        documentation="https://docs.riskxobservatory.com/api/errors#stale-models"
    )


# Exception handlers for FastAPI
async def rrio_api_error_handler(request: Request, exc: RRIOAPIError) -> JSONResponse:
    """Global handler for RRIOAPIError exceptions."""
    return create_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
        documentation=exc.documentation,
        request=request
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global handler for unhandled exceptions."""
    logger.exception("Unhandled exception in API", extra={"path": request.url.path})
    
    return create_error_response(
        status_code=500,
        code=ErrorCodes.INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred. Please try again later.",
        request=request
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global handler for Pydantic validation exceptions."""
    details = validation_error_handler(exc)
    
    return create_error_response(
        status_code=422,
        code=ErrorCodes.VALIDATION_ERROR,
        message="Request validation failed",
        details=details,
        request=request
    )