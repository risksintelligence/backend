"""Premium subscription service for feature differentiation and usage tracking."""
from __future__ import annotations

import asyncpg
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.services.auth_service import User

class SubscriptionTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class FeatureCategory(Enum):
    SCENARIO_STUDIO = "scenario_studio"
    ADVANCED_ANALYTICS = "advanced_analytics"
    API_ACCESS = "api_access"
    COLLABORATION = "collaboration"
    ALERTS = "alerts"
    EXPORTS = "exports"
    ADMIN = "admin"

@dataclass
class SubscriptionLimits:
    # Scenario Studio limits
    scenarios_per_month: int
    saved_scenarios: int
    alert_thresholds: int
    simultaneous_shocks: int
    
    # Data access limits
    api_calls_per_day: int
    export_downloads_per_month: int
    historical_data_months: int
    
    # Collaboration limits
    shared_scenarios: int
    team_members: int
    
    # Features enabled
    premium_analytics: bool
    real_time_alerts: bool
    webhook_integrations: bool
    priority_support: bool
    white_label: bool

@dataclass
class UsageMetrics:
    user_id: int
    period_start: datetime
    period_end: datetime
    scenarios_run: int
    api_calls_made: int
    exports_downloaded: int
    alerts_triggered: int
    storage_used_mb: float

class SubscriptionService:
    """Production subscription service with feature gating and usage tracking."""
    
    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self._limits = self._define_subscription_limits()
    
    def _define_subscription_limits(self) -> Dict[SubscriptionTier, SubscriptionLimits]:
        """Define limits and features for each subscription tier."""
        return {
            SubscriptionTier.FREE: SubscriptionLimits(
                scenarios_per_month=10,
                saved_scenarios=3,
                alert_thresholds=1,
                simultaneous_shocks=2,
                api_calls_per_day=100,
                export_downloads_per_month=5,
                historical_data_months=3,
                shared_scenarios=0,
                team_members=1,
                premium_analytics=False,
                real_time_alerts=False,
                webhook_integrations=False,
                priority_support=False,
                white_label=False
            ),
            
            SubscriptionTier.BASIC: SubscriptionLimits(
                scenarios_per_month=100,
                saved_scenarios=25,
                alert_thresholds=5,
                simultaneous_shocks=5,
                api_calls_per_day=1000,
                export_downloads_per_month=50,
                historical_data_months=12,
                shared_scenarios=10,
                team_members=5,
                premium_analytics=True,
                real_time_alerts=True,
                webhook_integrations=False,
                priority_support=False,
                white_label=False
            ),
            
            SubscriptionTier.PREMIUM: SubscriptionLimits(
                scenarios_per_month=500,
                saved_scenarios=100,
                alert_thresholds=25,
                simultaneous_shocks=10,
                api_calls_per_day=10000,
                export_downloads_per_month=500,
                historical_data_months=36,
                shared_scenarios=100,
                team_members=25,
                premium_analytics=True,
                real_time_alerts=True,
                webhook_integrations=True,
                priority_support=True,
                white_label=False
            ),
            
            SubscriptionTier.ENTERPRISE: SubscriptionLimits(
                scenarios_per_month=-1,  # Unlimited
                saved_scenarios=-1,
                alert_thresholds=-1,
                simultaneous_shocks=-1,
                api_calls_per_day=-1,
                export_downloads_per_month=-1,
                historical_data_months=-1,
                shared_scenarios=-1,
                team_members=-1,
                premium_analytics=True,
                real_time_alerts=True,
                webhook_integrations=True,
                priority_support=True,
                white_label=True
            )
        }
    
    def get_subscription_limits(self, tier: str) -> SubscriptionLimits:
        """Get subscription limits for a tier."""
        try:
            subscription_tier = SubscriptionTier(tier)
            return self._limits[subscription_tier]
        except ValueError:
            # Default to free tier for unknown tiers
            return self._limits[SubscriptionTier.FREE]
    
    async def check_feature_access(self, user: User, feature: FeatureCategory) -> bool:
        """Check if user has access to a specific feature."""
        limits = self.get_subscription_limits(user.subscription_tier)
        
        if feature == FeatureCategory.SCENARIO_STUDIO:
            return True  # Basic access for all tiers
        elif feature == FeatureCategory.ADVANCED_ANALYTICS:
            return limits.premium_analytics
        elif feature == FeatureCategory.API_ACCESS:
            return True  # Basic API access for all tiers
        elif feature == FeatureCategory.COLLABORATION:
            return limits.shared_scenarios > 0
        elif feature == FeatureCategory.ALERTS:
            return limits.real_time_alerts
        elif feature == FeatureCategory.EXPORTS:
            return True  # Basic export for all tiers
        elif feature == FeatureCategory.ADMIN:
            return user.role == "admin"
        
        return False
    
    async def check_usage_limit(
        self,
        user: User,
        feature: str,
        increment: int = 1
    ) -> tuple[bool, Optional[str]]:
        """Check if user is within usage limits for a feature."""
        limits = self.get_subscription_limits(user.subscription_tier)
        current_usage = await self.get_current_usage(user.id)
        
        if feature == "scenarios_per_month":
            limit = limits.scenarios_per_month
            current = current_usage.scenarios_run
        elif feature == "api_calls_per_day":
            limit = limits.api_calls_per_day
            current = current_usage.api_calls_made
        elif feature == "export_downloads_per_month":
            limit = limits.export_downloads_per_month  
            current = current_usage.exports_downloaded
        elif feature == "saved_scenarios":
            limit = limits.saved_scenarios
            current = await self.count_saved_scenarios(user.id)
        elif feature == "alert_thresholds":
            limit = limits.alert_thresholds
            current = await self.count_alert_thresholds(user.id)
        else:
            return True, None
        
        # Unlimited for enterprise
        if limit == -1:
            return True, None
        
        if current + increment > limit:
            tier_name = user.subscription_tier.upper()
            return False, f"Usage limit exceeded. {tier_name} tier allows {limit} {feature.replace('_', ' ')} per period. Current usage: {current}"
        
        return True, None
    
    async def track_usage(
        self,
        user_id: int,
        feature: str,
        amount: int = 1,
        metadata: Dict[str, Any] = None
    ):
        """Track usage for billing and limit enforcement."""
        # Get current period
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            # Update or insert usage record
            await conn.execute("""
                INSERT INTO subscription_usage (user_id, feature, usage_count, period_start, period_end)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, feature, period_start)
                DO UPDATE SET 
                    usage_count = subscription_usage.usage_count + EXCLUDED.usage_count,
                    created_at = NOW()
            """, user_id, feature, amount, period_start, period_end)
        finally:
            await conn.close()
    
    async def get_current_usage(self, user_id: int) -> UsageMetrics:
        """Get current usage metrics for a user."""
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            results = await conn.fetch("""
                SELECT feature, usage_count
                FROM subscription_usage
                WHERE user_id = $1 AND period_start = $2
            """, user_id, period_start)
            
            usage_dict = {result["feature"]: result["usage_count"] for result in results}
            
            return UsageMetrics(
                user_id=user_id,
                period_start=period_start,
                period_end=period_end,
                scenarios_run=usage_dict.get("scenarios_per_month", 0),
                api_calls_made=usage_dict.get("api_calls_per_day", 0),
                exports_downloaded=usage_dict.get("export_downloads_per_month", 0),
                alerts_triggered=usage_dict.get("alerts_triggered", 0),
                storage_used_mb=usage_dict.get("storage_used_mb", 0.0)
            )
        finally:
            await conn.close()
    
    async def count_saved_scenarios(self, user_id: int) -> int:
        """Count user's saved scenarios."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM saved_scenarios WHERE user_id = $1
            """, user_id)
            return result or 0
        finally:
            await conn.close()
    
    async def count_alert_thresholds(self, user_id: int) -> int:
        """Count user's active alert thresholds."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM alert_thresholds WHERE user_id = $1 AND is_active = TRUE
            """, user_id)
            return result or 0
        finally:
            await conn.close()
    
    async def get_subscription_status(self, user: User) -> Dict[str, Any]:
        """Get comprehensive subscription status for a user."""
        limits = self.get_subscription_limits(user.subscription_tier)
        usage = await self.get_current_usage(user.id)
        
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
            },
            "current_usage": {
                "scenarios_run": usage.scenarios_run,
                "api_calls_made": usage.api_calls_made,
                "exports_downloaded": usage.exports_downloaded,
                "alerts_triggered": usage.alerts_triggered,
                "saved_scenarios": await self.count_saved_scenarios(user.id),
                "alert_thresholds": await self.count_alert_thresholds(user.id)
            },
            "period": {
                "start": usage.period_start.isoformat(),
                "end": usage.period_end.isoformat()
            }
        }
    
    async def upgrade_subscription(
        self,
        user_id: int,
        new_tier: str,
        billing_info: Dict[str, Any] = None
    ) -> bool:
        """Upgrade user's subscription tier."""
        try:
            SubscriptionTier(new_tier)  # Validate tier
        except ValueError:
            return False
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.execute("""
                UPDATE users 
                SET subscription_tier = $1 
                WHERE id = $2
            """, new_tier, user_id)
            
            # Log the upgrade
            await conn.execute("""
                INSERT INTO admin_audit_log (actor, action, payload)
                VALUES ($1, $2, $3)
            """, f"user_{user_id}", "subscription_upgrade", {
                "new_tier": new_tier,
                "billing_info": billing_info or {}
            })
            
            return result != "UPDATE 0"
        finally:
            await conn.close()

# Global service instance
_subscription_service: Optional[SubscriptionService] = None

def get_subscription_service() -> SubscriptionService:
    """Dependency injection for subscription service."""
    global _subscription_service
    if _subscription_service is None:
        postgres_dsn = os.environ.get("RIS_POSTGRES_DSN") or "postgresql://placeholder:placeholder@localhost/placeholder"
        if postgres_dsn.endswith("/placeholder"):
            logger = logging.getLogger(__name__)
            logger.warning("SubscriptionService initialized with placeholder DSN. Set RIS_POSTGRES_DSN for persistence.")
        _subscription_service = SubscriptionService(postgres_dsn)
    return _subscription_service
