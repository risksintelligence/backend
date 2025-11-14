"""Subscription management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.services.subscription_service import get_subscription_service, SubscriptionService, FeatureCategory
from src.services.auth_service import User
from src.api.middleware.auth import require_auth

router = APIRouter(prefix="/api/v1/subscription", tags=["subscription"])

class UpgradeRequest(BaseModel):
    tier: str
    billing_info: Dict[str, Any] = {}

@router.get("/status")
async def get_subscription_status(
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Get user's subscription status and usage."""
    status = await subscription_service.get_subscription_status(user)
    return status

@router.get("/limits")
async def get_subscription_limits(
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Get subscription limits for user's tier."""
    limits = subscription_service.get_subscription_limits(user.subscription_tier)
    
    return {
        "tier": user.subscription_tier,
        "limits": {
            "scenarios_per_month": limits.scenarios_per_month,
            "saved_scenarios": limits.saved_scenarios,
            "alert_thresholds": limits.alert_thresholds,
            "simultaneous_shocks": limits.simultaneous_shocks,
            "api_calls_per_day": limits.api_calls_per_day,
            "export_downloads_per_month": limits.export_downloads_per_month,
            "historical_data_months": limits.historical_data_months,
            "shared_scenarios": limits.shared_scenarios,
            "team_members": limits.team_members
        },
        "features": {
            "premium_analytics": limits.premium_analytics,
            "real_time_alerts": limits.real_time_alerts,
            "webhook_integrations": limits.webhook_integrations,
            "priority_support": limits.priority_support,
            "white_label": limits.white_label
        }
    }

@router.get("/features/{feature}")
async def check_feature_access(
    feature: str,
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Check if user has access to a specific feature."""
    try:
        feature_category = FeatureCategory(feature)
        has_access = await subscription_service.check_feature_access(user, feature_category)
        
        return {
            "feature": feature,
            "has_access": has_access,
            "user_tier": user.subscription_tier
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {feature}")

@router.post("/upgrade")
async def upgrade_subscription(
    request: UpgradeRequest,
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Upgrade user's subscription tier."""
    try:
        success = await subscription_service.upgrade_subscription(
            user_id=user.id,
            new_tier=request.tier,
            billing_info=request.billing_info
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Invalid subscription tier or upgrade failed")
        
        return {
            "success": True,
            "message": f"Subscription upgraded to {request.tier}",
            "new_tier": request.tier
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/usage")
async def get_current_usage(
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Get current usage metrics for the user."""
    usage = await subscription_service.get_current_usage(user.id)
    
    return {
        "period_start": usage.period_start.isoformat(),
        "period_end": usage.period_end.isoformat(),
        "usage": {
            "scenarios_run": usage.scenarios_run,
            "api_calls_made": usage.api_calls_made,
            "exports_downloaded": usage.exports_downloaded,
            "alerts_triggered": usage.alerts_triggered,
            "storage_used_mb": usage.storage_used_mb
        }
    }

@router.post("/track-usage")
async def track_usage_endpoint(
    feature: str,
    amount: int = 1,
    user: User = Depends(require_auth),
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """Track usage for a specific feature (internal endpoint)."""
    await subscription_service.track_usage(user.id, feature, amount)
    
    return {
        "success": True,
        "message": f"Tracked {amount} usage for {feature}"
    }

@router.get("/tiers")
async def list_subscription_tiers(
    subscription_service: SubscriptionService = Depends(get_subscription_service)
):
    """List all available subscription tiers and their features."""
    from src.services.subscription_service import SubscriptionTier
    
    tiers = {}
    for tier in SubscriptionTier:
        limits = subscription_service.get_subscription_limits(tier.value)
        tiers[tier.value] = {
            "name": tier.value.title(),
            "limits": {
                "scenarios_per_month": limits.scenarios_per_month,
                "saved_scenarios": limits.saved_scenarios,
                "alert_thresholds": limits.alert_thresholds,
                "simultaneous_shocks": limits.simultaneous_shocks,
                "api_calls_per_day": limits.api_calls_per_day,
                "export_downloads_per_month": limits.export_downloads_per_month,
                "historical_data_months": limits.historical_data_months,
                "shared_scenarios": limits.shared_scenarios,
                "team_members": limits.team_members
            },
            "features": {
                "premium_analytics": limits.premium_analytics,
                "real_time_alerts": limits.real_time_alerts,
                "webhook_integrations": limits.webhook_integrations,
                "priority_support": limits.priority_support,
                "white_label": limits.white_label
            }
        }
    
    return {"tiers": tiers}