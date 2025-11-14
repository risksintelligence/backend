"""
Backup and Recovery Management API
Admin endpoints for managing database backups and recovery operations
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.src.services.backup_service import get_backup_service, BackupService
from backend.src.services.auth_service import User
from backend.src.api.middleware.auth import require_admin, require_deployment_control
from backend.src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["backup"])

class BackupRequest(BaseModel):
    backup_type: str = "full"  # 'full', 'postgres', 'redis'
    compress: bool = True
    upload_to_s3: bool = False

class RecoveryRequest(BaseModel):
    backup_filename: str
    recovery_type: str = "full"  # 'full', 'postgres', 'redis'
    target_timestamp: Optional[str] = None

@router.get("/backups/status")
async def get_backup_status(
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_admin)
):
    """Get current backup system status and recent backup history."""
    try:
        # Get recent backup logs from admin actions
        from backend.src.core.database import get_database_pool
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            backup_sessions = await conn.fetch("""
                SELECT action, metadata, occurred_at 
                FROM admin_actions 
                WHERE action = 'full_backup_session'
                ORDER BY occurred_at DESC 
                LIMIT 10
            """)
            
            backup_history = []
            for session in backup_sessions:
                metadata = session['metadata']
                backup_history.append({
                    'session_id': metadata.get('session_id'),
                    'started_at': metadata.get('started_at'),
                    'completed_at': metadata.get('completed_at'),
                    'backups_count': len(metadata.get('backups', [])),
                    'uploads_count': len(metadata.get('uploads', [])),
                    'errors_count': len(metadata.get('errors', [])),
                    'success': len(metadata.get('errors', [])) == 0
                })
        
        # Check backup system health
        health_status = await backup_service.health_check()
        
        return {
            "status": "healthy" if health_status else "unhealthy",
            "backup_directory": str(backup_service.backup_dir),
            "s3_configured": backup_service.s3_client is not None,
            "recent_backups": backup_history,
            "last_check": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting backup status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backup status: {e}")

@router.post("/backups/create")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_deployment_control)
):
    """Create a new backup manually."""
    try:
        logger.info(f"Manual backup requested by {current_user.username}: {request.backup_type}")
        
        if request.backup_type == "full":
            # Run full backup in background
            background_tasks.add_task(
                backup_service.create_full_backup,
                upload_to_s3=request.upload_to_s3
            )
            
            return {
                "status": "initiated",
                "message": f"Full backup initiated by {current_user.username}",
                "backup_type": request.backup_type,
                "upload_to_s3": request.upload_to_s3
            }
            
        elif request.backup_type == "postgres":
            backup_result = await backup_service.create_postgres_backup(compress=request.compress)
            
            if request.upload_to_s3:
                from backend.src.core.config import get_settings
                settings = get_settings()
                if settings.BACKUP_S3_BUCKET:
                    upload_result = await backup_service.upload_to_s3(
                        backup_result, 
                        settings.BACKUP_S3_BUCKET
                    )
                    backup_result['upload'] = upload_result
            
            return {
                "status": "completed", 
                "message": "PostgreSQL backup completed",
                "backup": backup_result
            }
            
        elif request.backup_type == "redis":
            backup_result = await backup_service.create_redis_backup()
            
            if request.upload_to_s3:
                from backend.src.core.config import get_settings
                settings = get_settings()
                if settings.BACKUP_S3_BUCKET:
                    upload_result = await backup_service.upload_to_s3(
                        backup_result,
                        settings.BACKUP_S3_BUCKET
                    )
                    backup_result['upload'] = upload_result
            
            return {
                "status": "completed",
                "message": "Redis backup completed", 
                "backup": backup_result
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown backup type: {request.backup_type}")
            
    except Exception as e:
        logger.error(f"Manual backup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")

@router.get("/backups/files")
async def list_backup_files(
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_admin)
):
    """List available backup files."""
    try:
        backup_files = []
        
        if backup_service.backup_dir.exists():
            for backup_file in backup_service.backup_dir.iterdir():
                if backup_file.is_file():
                    stat = backup_file.stat()
                    backup_files.append({
                        'filename': backup_file.name,
                        'size_bytes': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'backup_type': 'postgres' if 'postgres' in backup_file.name else 'redis' if 'redis' in backup_file.name else 'unknown'
                    })
        
        # Sort by creation time, newest first
        backup_files.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            "backup_files": backup_files,
            "total_files": len(backup_files),
            "total_size_bytes": sum(f['size_bytes'] for f in backup_files)
        }
        
    except Exception as e:
        logger.error(f"Error listing backup files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backup files: {e}")

@router.post("/backups/cleanup")
async def cleanup_old_backups(
    keep_days: int = 7,
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_deployment_control)
):
    """Clean up old backup files."""
    try:
        logger.info(f"Backup cleanup requested by {current_user.username}: keep {keep_days} days")
        
        cleanup_result = await backup_service.cleanup_old_backups(keep_days)
        
        return {
            "status": "completed",
            "message": f"Cleaned up {cleanup_result['removed_count']} old backup files",
            "removed_files": cleanup_result['removed_files'],
            "keep_days": keep_days
        }
        
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

@router.get("/backups/health")
async def check_backup_health(
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_admin)
):
    """Check backup system health."""
    try:
        health_status = await backup_service.health_check()
        
        # Additional health checks
        health_details = {
            "overall_status": "healthy" if health_status else "unhealthy",
            "checks": {
                "backup_directory": backup_service.backup_dir.exists(),
                "s3_configuration": backup_service.s3_client is not None,
                "database_connectivity": health_status,  # Already checked in health_check
            }
        }
        
        # Check disk space
        import os
        if backup_service.backup_dir.exists():
            statvfs = os.statvfs(backup_service.backup_dir)
            free_space = statvfs.f_bavail * statvfs.f_frsize
            total_space = statvfs.f_blocks * statvfs.f_frsize
            
            health_details["disk_space"] = {
                "free_bytes": free_space,
                "total_bytes": total_space,
                "free_percentage": (free_space / total_space) * 100,
                "adequate": free_space > 1024**3  # At least 1GB free
            }
            
            health_details["checks"]["disk_space"] = health_details["disk_space"]["adequate"]
        
        # Overall health is healthy if all checks pass
        all_checks_pass = all(health_details["checks"].values())
        health_details["overall_status"] = "healthy" if all_checks_pass else "unhealthy"
        
        return health_details
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/recovery/prepare")
async def prepare_recovery(
    request: RecoveryRequest,
    backup_service: BackupService = Depends(get_backup_service),
    current_user: User = Depends(require_deployment_control)
):
    """Prepare recovery operation (dry-run and validation)."""
    try:
        logger.warning(f"Recovery preparation requested by {current_user.username}: {request.backup_filename}")
        
        # Validate backup file exists
        backup_path = backup_service.backup_dir / request.backup_filename
        if not backup_path.exists():
            raise HTTPException(status_code=404, detail=f"Backup file not found: {request.backup_filename}")
        
        # Basic validation
        file_size = backup_path.stat().st_size
        
        recovery_plan = {
            "recovery_type": request.recovery_type,
            "backup_file": {
                "filename": request.backup_filename,
                "size_bytes": file_size,
                "backup_type": "postgres" if "postgres" in request.backup_filename else "redis" if "redis" in request.backup_filename else "unknown"
            },
            "estimated_duration_minutes": max(5, file_size // (1024**2)),  # Rough estimate: 1 minute per MB
            "warnings": [
                "This operation will replace current data",
                "All active connections will be terminated",
                "Application downtime is expected during recovery"
            ],
            "prerequisites": [
                "Ensure no critical operations are running",
                "Create a backup of current data if needed", 
                "Coordinate with team for maintenance window"
            ]
        }
        
        return {
            "status": "ready",
            "message": "Recovery plan prepared - review carefully before proceeding",
            "recovery_plan": recovery_plan,
            "prepared_by": current_user.username,
            "prepared_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Recovery preparation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recovery preparation failed: {e}")

@router.get("/recovery/status")
async def get_recovery_status(
    current_user: User = Depends(require_admin)
):
    """Get recovery operation status."""
    # This would track ongoing recovery operations
    # For now, return a simple status
    return {
        "active_recovery": None,
        "last_recovery": None,
        "status": "No active recovery operations",
        "timestamp": datetime.utcnow().isoformat()
    }