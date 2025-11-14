from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.services.admin_service import get_admin_service, AdminService
from src.services.scenario_service import get_alert_service
from src.services.research_service import get_research_service
from src.services.scenario_service import get_scenario_service
from src.services.feature_flag_service import get_feature_flag_service, FeatureFlagService
from src.services.auth_service import get_auth_service, AuthService, User
from src.services.deployment_service import get_deployment_service, DeploymentService
from src.api.middleware.auth import require_admin, require_deployment_control

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

class CreateFeatureFlagRequest(BaseModel):
    name: str
    description: Optional[str] = None
    is_enabled: bool = False
    rollout_percentage: int = 0
    target_roles: List[str] = []
    target_subscription_tiers: List[str] = []
    metadata: Dict[str, Any] = {}

class UpdateFeatureFlagRequest(BaseModel):
    is_enabled: Optional[bool] = None
    rollout_percentage: Optional[int] = None
    target_roles: Optional[List[str]] = None
    target_subscription_tiers: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class DeploymentActionRequest(BaseModel):
    action_type: str  # 'restart', 'deploy', 'scale'
    target_service: str
    parameters: Dict[str, Any] = {}

@router.get("/health")
async def get_system_health(
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_admin)
):
    """Get comprehensive system health and metrics."""
    return await admin_service.get_system_health()

@router.get("/deployments")
async def get_deployments(
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_admin)
):
    """Get deployment status from Render API."""
    return await admin_service.get_deployment_status()

@router.get("/data-quality")
async def get_data_quality(
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_admin)
):
    """Get data quality and freshness metrics."""
    return await admin_service.get_data_quality_metrics()

@router.get("/audit-log")
async def get_audit_log(
    limit: int = 50,
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_admin)
):
    """Get recent admin audit log entries."""
    return {"logs": await admin_service.get_audit_log(limit)}

@router.get("/alerts")
async def get_alerts(
    alert_service = Depends(get_alert_service),
    current_user: User = Depends(require_admin)
):
    """Get alert subscriptions and delivery status."""
    return {
        "subscriptions": alert_service.list_subscriptions(),
        "deliveries": alert_service.deliveries(),
    }

@router.get("/peer-reviews")
async def get_peer_reviews(
    service = Depends(get_research_service),
    current_user: User = Depends(require_admin)
):
    """Get peer review submissions."""
    return {"reviews": []}

@router.get("/scenario-runs")
async def get_scenario_runs(
    service = Depends(get_scenario_service),
    current_user: User = Depends(require_admin)
):
    """Get recent scenario simulation runs."""
    return {"runs": service.list_runs(20)}

# Feature Flag Management
@router.get("/feature-flags")
async def list_feature_flags(
    flag_service: FeatureFlagService = Depends(get_feature_flag_service),
    current_user: User = Depends(require_admin)
):
    """List all feature flags."""
    flags = await flag_service.list_feature_flags()
    return {
        "flags": [
            {
                "id": flag.id,
                "name": flag.name,
                "description": flag.description,
                "is_enabled": flag.is_enabled,
                "rollout_percentage": flag.rollout_percentage,
                "target_roles": flag.target_roles,
                "target_subscription_tiers": flag.target_subscription_tiers,
                "metadata": flag.metadata,
                "created_at": flag.created_at.isoformat(),
                "updated_at": flag.updated_at.isoformat()
            }
            for flag in flags
        ]
    }

@router.post("/feature-flags")
async def create_feature_flag(
    request: CreateFeatureFlagRequest,
    flag_service: FeatureFlagService = Depends(get_feature_flag_service),
    current_user: User = Depends(require_admin)
):
    """Create a new feature flag."""
    try:
        flag = await flag_service.create_feature_flag(
            name=request.name,
            description=request.description,
            is_enabled=request.is_enabled,
            rollout_percentage=request.rollout_percentage,
            target_roles=request.target_roles,
            target_subscription_tiers=request.target_subscription_tiers,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": f"Feature flag '{flag.name}' created successfully",
            "flag": {
                "id": flag.id,
                "name": flag.name,
                "is_enabled": flag.is_enabled
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/feature-flags/{flag_name}")
async def update_feature_flag(
    flag_name: str,
    request: UpdateFeatureFlagRequest,
    flag_service: FeatureFlagService = Depends(get_feature_flag_service),
    current_user: User = Depends(require_admin)
):
    """Update a feature flag."""
    try:
        flag = await flag_service.update_feature_flag(
            name=flag_name,
            is_enabled=request.is_enabled,
            rollout_percentage=request.rollout_percentage,
            target_roles=request.target_roles,
            target_subscription_tiers=request.target_subscription_tiers,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": f"Feature flag '{flag.name}' updated successfully",
            "flag": {
                "id": flag.id,
                "name": flag.name,
                "is_enabled": flag.is_enabled,
                "rollout_percentage": flag.rollout_percentage
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/feature-flags/{flag_name}")
async def delete_feature_flag(
    flag_name: str,
    flag_service: FeatureFlagService = Depends(get_feature_flag_service),
    current_user: User = Depends(require_admin)
):
    """Delete a feature flag."""
    try:
        await flag_service.delete_feature_flag(flag_name)
        
        return {
            "success": True,
            "message": f"Feature flag '{flag_name}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# User Management
@router.get("/users")
async def list_users(
    auth_service: AuthService = Depends(get_auth_service),
    current_user: User = Depends(require_admin)
):
    """List all users (admin only)."""
    # This would need to be implemented in AuthService
    # For now, return placeholder
    return {"users": [], "message": "User listing not yet implemented"}

# Deployment Controls
@router.get("/services")
async def list_services(
    deployment_service: DeploymentService = Depends(get_deployment_service),
    current_user: User = Depends(require_admin)
):
    """List available Render services."""
    services = await deployment_service.get_available_services()
    return {"services": services}

@router.get("/deployment-actions")
async def list_deployment_actions(
    user_id: Optional[int] = None,
    limit: int = 50,
    deployment_service: DeploymentService = Depends(get_deployment_service),
    current_user: User = Depends(require_admin)
):
    """List deployment actions."""
    actions = await deployment_service.list_deployment_actions(user_id, limit)
    return {
        "actions": [
            {
                "id": action.id,
                "user_id": action.user_id,
                "action_type": action.action_type.value,
                "target_service": action.target_service,
                "parameters": action.parameters,
                "status": action.status.value,
                "result": action.result,
                "started_at": action.started_at.isoformat(),
                "completed_at": action.completed_at.isoformat() if action.completed_at else None
            }
            for action in actions
        ]
    }

@router.get("/deployment-actions/{action_id}")
async def get_deployment_action(
    action_id: int,
    deployment_service: DeploymentService = Depends(get_deployment_service),
    current_user: User = Depends(require_admin)
):
    """Get a specific deployment action."""
    action = await deployment_service.get_deployment_action(action_id)
    
    if not action:
        raise HTTPException(status_code=404, detail="Deployment action not found")
    
    return {
        "id": action.id,
        "user_id": action.user_id,
        "action_type": action.action_type.value,
        "target_service": action.target_service,
        "parameters": action.parameters,
        "status": action.status.value,
        "result": action.result,
        "started_at": action.started_at.isoformat(),
        "completed_at": action.completed_at.isoformat() if action.completed_at else None
    }

@router.post("/actions/refresh-cache")
async def refresh_cache(
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_deployment_control)
):
    """Manually trigger cache refresh for all data sources."""
    await admin_service.log_admin_action(current_user.username, "cache_refresh_triggered")
    
    # TODO: Implement actual cache refresh logic
    # This should trigger the cache refresh jobs
    
    return {"status": "triggered", "message": "Cache refresh initiated"}

@router.post("/actions/retrain-models")
async def retrain_models(
    admin_service: AdminService = Depends(get_admin_service),
    current_user: User = Depends(require_deployment_control)
):
    """Manually trigger ML model retraining."""
    await admin_service.log_admin_action(current_user.username, "model_retrain_triggered")
    
    # TODO: Implement actual model retraining trigger
    # This should start the ML training jobs
    
    return {"status": "triggered", "message": "Model retraining initiated"}

@router.post("/actions/deployment")
async def trigger_deployment_action(
    request: DeploymentActionRequest,
    admin_service: AdminService = Depends(get_admin_service),
    deployment_service: DeploymentService = Depends(get_deployment_service),
    current_user: User = Depends(require_deployment_control)
):
    """Trigger deployment actions (restart, deploy, scale)."""
    try:
        # Create and execute the deployment action
        action = await deployment_service.create_deployment_action(
            user_id=current_user.id,
            action_type=request.action_type,
            target_service=request.target_service,
            parameters=request.parameters
        )
        
        # Log the action
        await admin_service.log_admin_action(
            current_user.username, 
            f"deployment_action_{request.action_type}",
            {
                "action_id": action.id,
                "target_service": request.target_service,
                "parameters": request.parameters
            }
        )
        
        return {
            "success": True,
            "action_id": action.id,
            "status": action.status.value,
            "message": f"Deployment action '{request.action_type}' initiated for service '{request.target_service}'"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stats/overview")
async def get_overview_stats(admin_service: AdminService = Depends(get_admin_service)):
    """Get high-level system statistics for dashboard."""
    health = await admin_service.get_system_health()
    quality = await admin_service.get_data_quality_metrics()
    
    return {
        "system_status": health["status"],
        "latest_geri": health["data"]["latest_geri"],
        "activity_24h": health["activity"],
        "data_series_count": len(quality["data_freshness"]),
        "stale_series_count": len([s for s in quality["data_freshness"] if s["status"] == "stale"]),
        "active_models": len(health["ml_models"]),
        "system_resources": health["system"]
    }
