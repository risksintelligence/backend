"""API endpoints for advanced export functionality with multiple formats and sharing."""
import os
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.src.services.advanced_export_service import (
    get_advanced_export_service,
    AdvancedExportService,
    ExportRequest,
    ExportFormat,
    ExportScope
)
from backend.src.services.auth_service import User
from backend.src.api.middleware.auth import require_auth, limit_exports, optional_auth
from backend.src.services.subscription_service import get_subscription_service, FeatureCategory

router = APIRouter(prefix="/api/v1/exports", tags=["advanced_exports"])


class CreateExportRequest(BaseModel):
    scope: str  # scenario_runs, saved_scenarios, alert_history, collaboration_activity
    format: str  # csv, json, xlsx
    filters: Dict[str, Any] = {}
    limit: int = 1000
    include_metadata: bool = True
    share_publicly: bool = False
    expires_in_hours: Optional[int] = None


class ExportResponse(BaseModel):
    export_id: str
    download_url: str
    file_size_bytes: int
    record_count: int
    format: str
    scope: str
    is_public: bool
    expires_at: Optional[str] = None
    created_at: str


@router.post("/", response_model=ExportResponse)
async def create_export(
    request: CreateExportRequest,
    user: User = Depends(limit_exports),  # Check export limits
    export_service: AdvancedExportService = Depends(get_advanced_export_service),
    subscription_service = Depends(get_subscription_service)
):
    """Create a new export with specified parameters."""
    # Validate format and scope
    try:
        export_format = ExportFormat(request.format)
        export_scope = ExportScope(request.scope)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid format or scope: {str(e)}")
    
    # Check premium features for advanced formats
    if export_format in [ExportFormat.EXCEL] or request.share_publicly:
        has_advanced = await subscription_service.check_feature_access(user, FeatureCategory.ADVANCED_ANALYTICS)
        if not has_advanced:
            raise HTTPException(
                status_code=403, 
                detail="Advanced export formats and public sharing require Premium subscription or higher"
            )
    
    # Check collaboration access for collaboration activity exports
    if export_scope == ExportScope.COLLABORATION_ACTIVITY:
        has_collaboration = await subscription_service.check_feature_access(user, FeatureCategory.COLLABORATION)
        if not has_collaboration:
            raise HTTPException(
                status_code=403, 
                detail="Collaboration activity exports require Premium subscription or higher"
            )
    
    # Validate limits
    if request.limit > 10000:
        raise HTTPException(status_code=400, detail="Export limit cannot exceed 10,000 records")
    
    if request.expires_in_hours and request.expires_in_hours > 8760:  # 1 year
        raise HTTPException(status_code=400, detail="Export cannot expire more than 1 year in the future")
    
    try:
        export_request = ExportRequest(
            user_id=user.id,
            scope=export_scope,
            format=export_format,
            filters=request.filters,
            limit=request.limit,
            include_metadata=request.include_metadata,
            share_publicly=request.share_publicly,
            expires_in_hours=request.expires_in_hours
        )
        
        result = await export_service.create_export(export_request)
        
        return ExportResponse(
            export_id=result.export_id,
            download_url=result.download_url,
            file_size_bytes=result.file_size_bytes,
            record_count=result.record_count,
            format=export_format.value,
            scope=export_scope.value,
            is_public=result.is_public,
            expires_at=result.expires_at.isoformat() if result.expires_at else None,
            created_at=result.created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export creation failed: {str(e)}")


@router.get("/", response_model=List[ExportResponse])
async def list_user_exports(
    limit: int = 50,
    user: User = Depends(require_auth),
    export_service: AdvancedExportService = Depends(get_advanced_export_service)
):
    """List all exports for the authenticated user."""
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    exports = await export_service.get_user_exports(user.id, limit)
    
    return [
        ExportResponse(
            export_id=export.export_id,
            download_url=export.download_url,
            file_size_bytes=export.file_size_bytes,
            record_count=export.record_count,
            format=os.path.splitext(export.file_path)[1][1:],  # Extract format from file extension
            scope="unknown",  # Would need to store this in export result
            is_public=export.is_public,
            expires_at=export.expires_at.isoformat() if export.expires_at else None,
            created_at=export.created_at.isoformat()
        )
        for export in exports
    ]


@router.get("/{export_id}")
async def get_export_info(
    export_id: str,
    user: User = Depends(optional_auth),
    export_service: AdvancedExportService = Depends(get_advanced_export_service)
):
    """Get export information (public exports don't require authentication)."""
    export = await export_service.get_export(export_id, user.id if user else None)
    
    if not export:
        raise HTTPException(status_code=404, detail="Export not found or access denied")
    
    return {
        "export_id": export.export_id,
        "download_url": export.download_url,
        "file_size_bytes": export.file_size_bytes,
        "record_count": export.record_count,
        "is_public": export.is_public,
        "expires_at": export.expires_at.isoformat() if export.expires_at else None,
        "created_at": export.created_at.isoformat()
    }


@router.get("/{export_id}/download")
async def download_export(
    export_id: str,
    user: User = Depends(optional_auth),
    export_service: AdvancedExportService = Depends(get_advanced_export_service)
):
    """Download export file (public exports don't require authentication)."""
    export = await export_service.get_export(export_id, user.id if user else None)
    
    if not export:
        raise HTTPException(status_code=404, detail="Export not found or access denied")
    
    if not os.path.exists(export.file_path):
        raise HTTPException(status_code=404, detail="Export file not found on disk")
    
    # Determine content type based on file extension
    file_extension = os.path.splitext(export.file_path)[1].lower()
    content_type_map = {
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pdf': 'application/pdf'
    }
    
    media_type = content_type_map.get(file_extension, 'application/octet-stream')
    filename = f"ris_export_{export_id}{file_extension}"
    
    return FileResponse(
        path=export.file_path,
        media_type=media_type,
        filename=filename
    )


@router.delete("/{export_id}")
async def delete_export(
    export_id: str,
    user: User = Depends(require_auth),
    export_service: AdvancedExportService = Depends(get_advanced_export_service)
):
    """Delete an export (only by owner)."""
    success = await export_service.delete_export(export_id, user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Export not found or access denied")
    
    return {"success": True, "message": "Export deleted successfully"}


@router.get("/formats/supported")
async def get_supported_formats():
    """Get list of supported export formats and their capabilities."""
    formats = {
        "csv": {
            "name": "CSV",
            "description": "Comma-separated values",
            "supports_metadata": True,
            "max_file_size": "500MB",
            "premium_required": False
        },
        "json": {
            "name": "JSON",
            "description": "JavaScript Object Notation",
            "supports_metadata": True,
            "max_file_size": "100MB",
            "premium_required": False
        },
        "xlsx": {
            "name": "Excel",
            "description": "Microsoft Excel spreadsheet",
            "supports_metadata": True,
            "max_file_size": "50MB",
            "premium_required": True
        }
    }
    
    scopes = {
        "scenario_runs": {
            "name": "Scenario Runs",
            "description": "Historical scenario simulation results",
            "available_filters": ["date_from", "date_to", "min_delta"]
        },
        "saved_scenarios": {
            "name": "Saved Scenarios",
            "description": "User's saved scenario configurations",
            "available_filters": ["include_shared", "is_public"]
        },
        "alert_history": {
            "name": "Alert History",
            "description": "Historical alert events and triggers",
            "available_filters": ["date_from", "trigger_type"]
        },
        "collaboration_activity": {
            "name": "Collaboration Activity",
            "description": "Comments, shares, and collaboration events",
            "available_filters": [],
            "premium_required": True
        }
    }
    
    return {
        "supported_formats": formats,
        "supported_scopes": scopes,
        "limits": {
            "max_records_per_export": 10000,
            "max_exports_per_day": 50,
            "max_file_retention_hours": 8760
        }
    }


# Background task endpoints (admin only)
@router.post("/maintenance/cleanup")
async def cleanup_expired_exports(
    user: User = Depends(require_auth),
    export_service: AdvancedExportService = Depends(get_advanced_export_service)
):
    """Cleanup expired exports (admin only)."""
    # This would typically be an admin-only endpoint
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    deleted_count = await export_service.cleanup_expired_exports()
    
    return {
        "success": True,
        "deleted_exports": deleted_count,
        "message": f"Cleaned up {deleted_count} expired exports"
    }