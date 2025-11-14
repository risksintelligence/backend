"""API endpoints for scenario sharing and collaboration."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.src.services.scenario_sharing_service import (
    get_scenario_sharing_service,
    ScenarioSharingService,
    PermissionLevel,
    SharingError
)
from backend.src.services.auth_service import User
from backend.src.api.middleware.auth import require_auth, limit_scenarios
from backend.src.services.subscription_service import get_subscription_service, FeatureCategory
from backend.src.services.scenario_collaboration_service import (
    get_scenario_collaboration_service,
    ScenarioCollaborationService,
    ActivityType
)

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenario_sharing"])


class CreateScenarioRequest(BaseModel):
    name: str
    description: Optional[str] = None
    shocks: List[Dict[str, Any]]
    horizon_hours: int
    baseline_value: Optional[float] = None
    scenario_value: Optional[float] = None
    is_public: bool = False


class UpdateScenarioRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    shocks: Optional[List[Dict[str, Any]]] = None
    horizon_hours: Optional[int] = None
    baseline_value: Optional[float] = None
    scenario_value: Optional[float] = None


class ShareScenarioRequest(BaseModel):
    shared_with_user_id: Optional[int] = None
    shared_with_email: Optional[EmailStr] = None
    permission_level: str = "view"
    expires_in_hours: Optional[int] = None


class CommentRequest(BaseModel):
    comment_text: str


@router.post("/")
async def create_scenario(
    request: CreateScenarioRequest,
    user: User = Depends(limit_scenarios),  # Check subscription limits
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Create a new scenario."""
    try:
        scenario = await sharing_service.save_scenario(
            user_id=user.id,
            name=request.name,
            description=request.description,
            shocks=request.shocks,
            horizon_hours=request.horizon_hours,
            baseline_value=request.baseline_value,
            scenario_value=request.scenario_value,
            is_public=request.is_public
        )
        
        return {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "shocks": scenario.shocks,
            "horizon_hours": scenario.horizon_hours,
            "baseline_value": scenario.baseline_value,
            "scenario_value": scenario.scenario_value,
            "is_public": scenario.is_public,
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat()
        }
    except SharingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create scenario")


@router.get("/")
async def list_scenarios(
    include_shared: bool = True,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Get all scenarios accessible to the user."""
    scenarios = await sharing_service.get_user_scenarios(user.id, include_shared)
    
    return {
        "scenarios": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "shocks": s.shocks,
                "horizon_hours": s.horizon_hours,
                "baseline_value": s.baseline_value,
                "scenario_value": s.scenario_value,
                "is_public": s.is_public,
                "is_owner": s.user_id == user.id,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat()
            }
            for s in scenarios
        ]
    }


@router.get("/shared")
async def list_shared_scenarios(
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Get scenarios shared with the user."""
    shared_scenarios = await sharing_service.get_shared_scenarios(user.id)
    
    return {
        "shared_scenarios": [
            {
                "scenario": {
                    "id": sv.scenario.id,
                    "name": sv.scenario.name,
                    "description": sv.scenario.description,
                    "shocks": sv.scenario.shocks,
                    "horizon_hours": sv.scenario.horizon_hours,
                    "baseline_value": sv.scenario.baseline_value,
                    "scenario_value": sv.scenario.scenario_value,
                    "is_public": sv.scenario.is_public,
                    "created_at": sv.scenario.created_at.isoformat(),
                    "updated_at": sv.scenario.updated_at.isoformat()
                },
                "owner_username": sv.owner_username,
                "permission_level": sv.permission_level.value,
                "shared_at": sv.shared_at.isoformat(),
                "expires_at": sv.expires_at.isoformat() if sv.expires_at else None
            }
            for sv in shared_scenarios
        ]
    }


@router.get("/{scenario_id}")
async def get_scenario(
    scenario_id: int,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Get a specific scenario."""
    try:
        scenario = await sharing_service.get_scenario(scenario_id, user.id)
        permission = await sharing_service.check_permission(scenario_id, user.id)
        
        return {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "shocks": scenario.shocks,
            "horizon_hours": scenario.horizon_hours,
            "baseline_value": scenario.baseline_value,
            "scenario_value": scenario.scenario_value,
            "is_public": scenario.is_public,
            "is_owner": scenario.user_id == user.id,
            "permission_level": permission.value if permission else None,
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat()
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.put("/{scenario_id}")
async def update_scenario(
    scenario_id: int,
    request: UpdateScenarioRequest,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Update a scenario."""
    try:
        scenario = await sharing_service.update_scenario(
            scenario_id=scenario_id,
            user_id=user.id,
            name=request.name,
            description=request.description,
            shocks=request.shocks,
            horizon_hours=request.horizon_hours,
            baseline_value=request.baseline_value,
            scenario_value=request.scenario_value
        )
        
        return {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "shocks": scenario.shocks,
            "horizon_hours": scenario.horizon_hours,
            "baseline_value": scenario.baseline_value,
            "scenario_value": scenario.scenario_value,
            "is_public": scenario.is_public,
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat()
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{scenario_id}")
async def delete_scenario(
    scenario_id: int,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Delete a scenario."""
    try:
        success = await sharing_service.delete_scenario(scenario_id, user.id)
        return {"success": success, "message": "Scenario deleted successfully"}
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{scenario_id}/share")
async def share_scenario(
    scenario_id: int,
    request: ShareScenarioRequest,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service),
    subscription_service = Depends(get_subscription_service)
):
    """Share a scenario with another user."""
    # Check if user has collaboration features
    has_collaboration = await subscription_service.check_feature_access(user, FeatureCategory.COLLABORATION)
    if not has_collaboration:
        raise HTTPException(
            status_code=403, 
            detail="Scenario sharing requires Premium subscription or higher"
        )
    
    try:
        permission_level = PermissionLevel(request.permission_level)
        share = await sharing_service.share_scenario(
            scenario_id=scenario_id,
            owner_user_id=user.id,
            shared_with_user_id=request.shared_with_user_id,
            shared_with_email=request.shared_with_email,
            permission_level=permission_level,
            expires_in_hours=request.expires_in_hours
        )
        
        return {
            "share_id": share.id,
            "scenario_id": share.scenario_id,
            "shared_with_user": share.shared_with_user,
            "shared_with_email": share.shared_with_email,
            "permission_level": share.permission_level.value,
            "expires_at": share.expires_at.isoformat() if share.expires_at else None,
            "created_at": share.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid permission level")
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{scenario_id}/shares")
async def list_scenario_shares(
    scenario_id: int,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """List all shares for a scenario (owner only)."""
    try:
        shares = await sharing_service.get_scenario_shares(scenario_id, user.id)
        
        return {
            "shares": [
                {
                    "share_id": share.id,
                    "shared_with_user": share.shared_with_user,
                    "shared_with_email": share.shared_with_email,
                    "permission_level": share.permission_level.value,
                    "expires_at": share.expires_at.isoformat() if share.expires_at else None,
                    "created_at": share.created_at.isoformat()
                }
                for share in shares
            ]
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{scenario_id}/shares/{share_id}")
async def revoke_scenario_share(
    scenario_id: int,
    share_id: int,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Revoke a scenario share."""
    try:
        success = await sharing_service.revoke_share(share_id, user.id)
        return {"success": success, "message": "Share revoked successfully"}
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{scenario_id}/visibility")
async def set_scenario_visibility(
    scenario_id: int,
    is_public: bool,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Make a scenario public or private."""
    try:
        scenario = await sharing_service.make_scenario_public(scenario_id, user.id, is_public)
        return {
            "scenario_id": scenario.id,
            "is_public": scenario.is_public,
            "message": f"Scenario visibility set to {'public' if is_public else 'private'}"
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{scenario_id}/permission")
async def check_scenario_permission(
    scenario_id: int,
    user: User = Depends(require_auth),
    sharing_service: ScenarioSharingService = Depends(get_scenario_sharing_service)
):
    """Check user's permission level for a scenario."""
    permission = await sharing_service.check_permission(scenario_id, user.id)
    
    return {
        "scenario_id": scenario_id,
        "user_id": user.id,
        "permission_level": permission.value if permission else None,
        "has_access": permission is not None
    }


# Collaboration endpoints
class ForkScenarioRequest(BaseModel):
    new_name: str
    new_description: Optional[str] = None


@router.post("/{scenario_id}/fork")
async def fork_scenario(
    scenario_id: int,
    request: ForkScenarioRequest,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service),
    subscription_service = Depends(get_subscription_service)
):
    """Create a copy (fork) of a scenario."""
    # Check if user has collaboration features
    has_collaboration = await subscription_service.check_feature_access(user, FeatureCategory.COLLABORATION)
    if not has_collaboration:
        raise HTTPException(
            status_code=403, 
            detail="Scenario forking requires Premium subscription or higher"
        )
    
    try:
        forked_scenario_id = await collaboration_service.fork_scenario(
            scenario_id=scenario_id,
            user_id=user.id,
            new_name=request.new_name,
            new_description=request.new_description
        )
        
        return {
            "forked_scenario_id": forked_scenario_id,
            "original_scenario_id": scenario_id,
            "name": request.new_name,
            "message": "Scenario forked successfully"
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{scenario_id}/comments")
async def add_comment(
    scenario_id: int,
    request: CommentRequest,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service),
    subscription_service = Depends(get_subscription_service)
):
    """Add a comment to a scenario."""
    # Check if user has collaboration features
    has_collaboration = await subscription_service.check_feature_access(user, FeatureCategory.COLLABORATION)
    if not has_collaboration:
        raise HTTPException(
            status_code=403, 
            detail="Scenario comments require Premium subscription or higher"
        )
    
    try:
        comment = await collaboration_service.add_comment(
            scenario_id=scenario_id,
            user_id=user.id,
            comment_text=request.comment_text
        )
        
        return {
            "comment_id": comment.id,
            "scenario_id": comment.scenario_id,
            "username": comment.username,
            "comment_text": comment.comment_text,
            "is_resolved": comment.is_resolved,
            "created_at": comment.created_at.isoformat(),
            "updated_at": comment.updated_at.isoformat()
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{scenario_id}/comments")
async def get_comments(
    scenario_id: int,
    include_resolved: bool = True,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service)
):
    """Get all comments for a scenario."""
    try:
        comments = await collaboration_service.get_scenario_comments(
            scenario_id=scenario_id,
            user_id=user.id,
            include_resolved=include_resolved
        )
        
        return {
            "comments": [
                {
                    "comment_id": comment.id,
                    "username": comment.username,
                    "comment_text": comment.comment_text,
                    "is_resolved": comment.is_resolved,
                    "created_at": comment.created_at.isoformat(),
                    "updated_at": comment.updated_at.isoformat()
                }
                for comment in comments
            ]
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


class UpdateCommentRequest(BaseModel):
    comment_text: Optional[str] = None
    is_resolved: Optional[bool] = None


@router.put("/{scenario_id}/comments/{comment_id}")
async def update_comment(
    scenario_id: int,
    comment_id: int,
    request: UpdateCommentRequest,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service)
):
    """Update a comment (by comment author or scenario owner)."""
    try:
        comment = await collaboration_service.update_comment(
            comment_id=comment_id,
            user_id=user.id,
            comment_text=request.comment_text,
            is_resolved=request.is_resolved
        )
        
        return {
            "comment_id": comment.id,
            "username": comment.username,
            "comment_text": comment.comment_text,
            "is_resolved": comment.is_resolved,
            "created_at": comment.created_at.isoformat(),
            "updated_at": comment.updated_at.isoformat()
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{scenario_id}/comments/{comment_id}")
async def delete_comment(
    scenario_id: int,
    comment_id: int,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service)
):
    """Delete a comment (by comment author or scenario owner)."""
    try:
        success = await collaboration_service.delete_comment(comment_id, user.id)
        return {"success": success, "message": "Comment deleted successfully"}
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{scenario_id}/activity")
async def get_scenario_activity(
    scenario_id: int,
    limit: int = 50,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service)
):
    """Get activity history for a scenario."""
    try:
        activities = await collaboration_service.get_scenario_activity(
            scenario_id=scenario_id,
            user_id=user.id,
            limit=limit
        )
        
        return {
            "activities": [
                {
                    "activity_id": activity.id,
                    "username": activity.username,
                    "activity_type": activity.activity_type.value,
                    "activity_data": activity.activity_data,
                    "created_at": activity.created_at.isoformat()
                }
                for activity in activities
            ]
        }
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{scenario_id}/stats")
async def get_scenario_statistics(
    scenario_id: int,
    user: User = Depends(require_auth),
    collaboration_service: ScenarioCollaborationService = Depends(get_scenario_collaboration_service)
):
    """Get collaboration statistics for a scenario."""
    try:
        stats = await collaboration_service.get_scenario_statistics(scenario_id, user.id)
        return stats
    except SharingError as e:
        raise HTTPException(status_code=403, detail=str(e))