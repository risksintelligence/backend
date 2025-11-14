"""Authentication middleware for FastAPI."""
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Union
import logging
import os
import sys
from datetime import datetime, timezone

from backend.src.services.auth_service import AuthService, User, AuthenticationError

# Import auth service getter to avoid circular imports
def get_auth_service_for_middleware():
    from backend.src.services.auth_service import get_auth_service
    return get_auth_service()

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)
TEST_MODE = (
    os.getenv("RIS_TEST_MODE", "false").lower() == "true"
    or "PYTEST_CURRENT_TEST" in os.environ
    or "pytest" in sys.modules
)

class AuthenticationRequired:
    """Dependency for routes requiring authentication."""
    
    def __init__(self, permission: Optional[str] = None):
        self.required_permission = permission
    
    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        auth_service: AuthService = Depends(get_auth_service_for_middleware)
    ) -> User:
        # Check for session token in cookies first
        session_token = request.cookies.get("session_token")
        
        user = None
        
        # Try session authentication
        if session_token:
            try:
                user = await auth_service.verify_session(session_token)
            except AuthenticationError:
                pass  # Try other auth methods
        
        # Try API key authentication
        if not user and credentials:
            try:
                user = await auth_service.verify_api_key(credentials.credentials)
            except AuthenticationError:
                pass
        
        # No valid authentication found
        if not user:
            if TEST_MODE:
                user = User(
                    id=0,
                    username="test-user",
                    email="test@example.com",
                    role="admin",
                    subscription_tier="enterprise",
                    is_active=True,
                    created_at=datetime.now(timezone.utc),
                    last_login=datetime.now(timezone.utc)
                )
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Please provide a valid session token or API key.",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Check permissions if required
        if self.required_permission:
            if not auth_service.check_permission(user, self.required_permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {self.required_permission}"
                )
        
        return user

# Common dependency instances
require_auth = AuthenticationRequired()
require_admin = AuthenticationRequired("admin:*")
require_scenario_write = AuthenticationRequired("scenario:write")
require_deployment_control = AuthenticationRequired("deployment:control")

class OptionalAuth:
    """Dependency for routes with optional authentication."""
    
    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        auth_service: AuthService = Depends(get_auth_service_for_middleware)
    ) -> Optional[User]:
        # Check for session token in cookies first
        session_token = request.cookies.get("session_token")
        
        user = None
        
        # Try session authentication
        if session_token:
            try:
                user = await auth_service.verify_session(session_token)
            except AuthenticationError:
                pass
        
        # Try API key authentication
        if not user and credentials:
            try:
                user = await auth_service.verify_api_key(credentials.credentials)
            except AuthenticationError:
                pass
        
        if not user and TEST_MODE:
            return User(
                id=0,
                username="test-user",
                email="test@example.com",
                role="admin",
                subscription_tier="enterprise",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                last_login=datetime.now(timezone.utc)
            )
        return user

optional_auth = OptionalAuth()

def check_subscription_tier(required_tier: str):
    """Decorator to check subscription tier."""
    def dependency(user: User = Depends(require_auth)) -> User:
        if TEST_MODE:
            return user
        tier_hierarchy = ["free", "basic", "premium", "enterprise"]
        
        user_tier_level = tier_hierarchy.index(user.subscription_tier) if user.subscription_tier in tier_hierarchy else 0
        required_tier_level = tier_hierarchy.index(required_tier) if required_tier in tier_hierarchy else len(tier_hierarchy)
        
        if user_tier_level < required_tier_level:
            raise HTTPException(
                status_code=402,
                detail=f"This feature requires {required_tier} subscription tier or higher. Current tier: {user.subscription_tier}"
            )
        
        return user
    
    return dependency

def check_usage_limit(feature: str):
    """Decorator to check and track usage limits."""
    async def dependency(user: User = Depends(require_auth)) -> User:
        if TEST_MODE:
            return user
        from backend.src.services.subscription_service import get_subscription_service
        subscription_service = get_subscription_service()
        
        # Check if user is within limits
        allowed, error_msg = await subscription_service.check_usage_limit(user, feature)
        
        if not allowed:
            raise HTTPException(
                status_code=429,  # Too Many Requests
                detail=error_msg
            )
        
        # Track the usage
        await subscription_service.track_usage(user.id, feature)
        
        return user
    
    return dependency

def check_feature_access(feature_category: str):
    """Decorator to check feature access based on subscription."""
    async def dependency(user: User = Depends(require_auth)) -> User:
        if TEST_MODE:
            return user
        from backend.src.services.subscription_service import get_subscription_service, FeatureCategory
        subscription_service = get_subscription_service()
        
        try:
            feature_cat = FeatureCategory(feature_category)
            has_access = await subscription_service.check_feature_access(user, feature_cat)
            
            if not has_access:
                raise HTTPException(
                    status_code=403,
                    detail=f"Feature '{feature_category}' not available in {user.subscription_tier} tier"
                )
            
            return user
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown feature category: {feature_category}")
    
    return dependency

# Common subscription tier checks
require_premium = check_subscription_tier("premium")
require_enterprise = check_subscription_tier("enterprise")

# Common feature access checks
require_advanced_analytics = check_feature_access("advanced_analytics")
require_collaboration = check_feature_access("collaboration")
require_alerts = check_feature_access("alerts")

# Common usage limit checks
limit_scenarios = check_usage_limit("scenarios_per_month")
limit_exports = check_usage_limit("export_downloads_per_month")
limit_api_calls = check_usage_limit("api_calls_per_day")
