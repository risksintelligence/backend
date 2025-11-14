"""Authentication endpoints for login, logout, API key management."""
from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel
from typing import List, Optional

from backend.src.services.auth_service import get_auth_service, AuthService, User, AuthenticationError
from backend.src.api.middleware.auth import require_auth, require_admin

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "analyst"
    subscription_tier: str = "free"

class CreateAPIKeyRequest(BaseModel):
    name: str
    permissions: List[str]
    expires_days: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    subscription_tier: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None

class APIKeyResponse(BaseModel):
    id: int
    name: str
    permissions: List[str]
    last_used: Optional[str]
    expires_at: Optional[str]
    is_active: bool

@router.post("/login")
async def login(
    request: LoginRequest,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate user and create session."""
    try:
        user = await auth_service.authenticate_user(request.username, request.password)
        session = await auth_service.create_session(user)
        
        # Set HTTP-only cookie
        response.set_cookie(
            key="session_token",
            value=session.session_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=24 * 60 * 60  # 24 hours
        )
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "subscription_tier": user.subscription_tier
            }
        }
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout user and invalidate session."""
    session_token = request.cookies.get("session_token")
    if session_token:
        await auth_service.logout_session(session_token)
    
    response.delete_cookie(key="session_token")
    return {"success": True, "message": "Logout successful"}

@router.get("/me")
async def get_current_user(user: User = Depends(require_auth)) -> UserResponse:
    """Get current authenticated user info."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        subscription_tier=user.subscription_tier,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None
    )

@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: User = Depends(require_admin)
):
    """Create a new user (admin only)."""
    try:
        user = await auth_service.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=request.role,
            subscription_tier=request.subscription_tier
        )
        
        return {
            "success": True,
            "message": "User created successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "subscription_tier": user.subscription_tier
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(require_auth),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create a new API key for the authenticated user."""
    try:
        api_key = await auth_service.create_api_key(
            user_id=user.id,
            name=request.name,
            permissions=request.permissions,
            expires_days=request.expires_days
        )
        
        return {
            "success": True,
            "message": "API key created successfully",
            "api_key": api_key,
            "warning": "Save this key securely. It will not be shown again."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api-keys")
async def list_api_keys(
    user: User = Depends(require_auth),
    auth_service: AuthService = Depends(get_auth_service)
) -> List[APIKeyResponse]:
    """List API keys for the authenticated user."""
    try:
        api_keys = await auth_service.list_user_api_keys(user.id)
        
        return [
            APIKeyResponse(
                id=key.id,
                name=key.name,
                permissions=key.permissions,
                last_used=key.last_used.isoformat() if key.last_used else None,
                expires_at=key.expires_at.isoformat() if key.expires_at else None,
                is_active=key.is_active
            )
            for key in api_keys
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/api-keys/{api_key_id}")
async def revoke_api_key(
    api_key_id: int,
    user: User = Depends(require_auth),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Revoke an API key."""
    try:
        await auth_service.revoke_api_key(user.id, api_key_id)
        
        return {
            "success": True,
            "message": "API key revoked successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))