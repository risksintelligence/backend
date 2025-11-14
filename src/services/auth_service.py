"""Authentication and authorization service with JWT and API key support."""
from __future__ import annotations

import hashlib
import secrets
import bcrypt
import jwt
import asyncpg
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class User:
    id: int
    username: str
    email: str
    role: str
    subscription_tier: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

@dataclass
class APIKey:
    id: int
    user_id: int
    name: str
    permissions: List[str]
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool

@dataclass
class AuthSession:
    user: User
    session_token: str
    expires_at: datetime

class AuthenticationError(Exception):
    pass

class AuthorizationError(Exception):
    pass

class AuthService:
    """Production authentication service with JWT and API key support."""
    
    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self.jwt_secret = os.environ.get("JWT_SECRET") or "dev-secret-key"
        self.jwt_algorithm = "HS256"
        self.session_timeout_hours = 24
        
    async def create_user(self, username: str, email: str, password: str, 
                         role: str = "analyst", subscription_tier: str = "free") -> User:
        """Create a new user with hashed password."""
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                INSERT INTO users (username, email, password_hash, role, subscription_tier)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, username, email, role, subscription_tier, is_active, created_at
            """, username, email, password_hash, role, subscription_tier)
            
            return User(
                id=result["id"],
                username=result["username"], 
                email=result["email"],
                role=result["role"],
                subscription_tier=result["subscription_tier"],
                is_active=result["is_active"],
                created_at=result["created_at"]
            )
        finally:
            await conn.close()
    
    async def authenticate_user(self, username: str, password: str) -> User:
        """Authenticate user by username/password."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                SELECT id, username, email, password_hash, role, subscription_tier, is_active, created_at, last_login
                FROM users 
                WHERE username = $1 AND is_active = TRUE
            """, username)
            
            if not result:
                raise AuthenticationError("Invalid credentials")
            
            if not bcrypt.checkpw(password.encode('utf-8'), result["password_hash"].encode('utf-8')):
                raise AuthenticationError("Invalid credentials")
            
            # Update last login
            await conn.execute("""
                UPDATE users SET last_login = NOW() WHERE id = $1
            """, result["id"])
            
            return User(
                id=result["id"],
                username=result["username"],
                email=result["email"],
                role=result["role"],
                subscription_tier=result["subscription_tier"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                last_login=datetime.now(timezone.utc)
            )
        finally:
            await conn.close()
    
    async def create_session(self, user: User) -> AuthSession:
        """Create a new session for authenticated user."""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self.session_timeout_hours)
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            await conn.execute("""
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES ($1, $2, $3)
            """, user.id, session_token, expires_at)
            
            return AuthSession(
                user=user,
                session_token=session_token,
                expires_at=expires_at
            )
        finally:
            await conn.close()
    
    async def verify_session(self, session_token: str) -> User:
        """Verify session token and return user."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                SELECT u.id, u.username, u.email, u.role, u.subscription_tier, u.is_active, u.created_at, u.last_login,
                       s.expires_at
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = $1 AND s.expires_at > NOW() AND u.is_active = TRUE
            """, session_token)
            
            if not result:
                raise AuthenticationError("Invalid or expired session")
            
            return User(
                id=result["id"],
                username=result["username"],
                email=result["email"],
                role=result["role"],
                subscription_tier=result["subscription_tier"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                last_login=result["last_login"]
            )
        finally:
            await conn.close()
    
    async def create_api_key(self, user_id: int, name: str, permissions: List[str], 
                           expires_days: Optional[int] = None) -> str:
        """Create a new API key for user."""
        raw_key = f"ris_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            await conn.execute("""
                INSERT INTO api_keys (user_id, key_hash, name, permissions, expires_at)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, key_hash, name, permissions, expires_at)
            
            return raw_key
        finally:
            await conn.close()
    
    async def verify_api_key(self, api_key: str) -> User:
        """Verify API key and return user."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                SELECT u.id, u.username, u.email, u.role, u.subscription_tier, u.is_active, u.created_at, u.last_login,
                       k.permissions, k.expires_at
                FROM api_keys k
                JOIN users u ON k.user_id = u.id
                WHERE k.key_hash = $1 AND k.is_active = TRUE 
                  AND (k.expires_at IS NULL OR k.expires_at > NOW()) 
                  AND u.is_active = TRUE
            """, key_hash)
            
            if not result:
                raise AuthenticationError("Invalid or expired API key")
            
            # Update last used
            await conn.execute("""
                UPDATE api_keys SET last_used = NOW() WHERE key_hash = $1
            """, key_hash)
            
            user = User(
                id=result["id"],
                username=result["username"],
                email=result["email"],
                role=result["role"],
                subscription_tier=result["subscription_tier"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                last_login=result["last_login"]
            )
            
            # Store permissions in user object for authorization
            user.api_permissions = result["permissions"]
            return user
        finally:
            await conn.close()
    
    def check_permission(self, user: User, required_permission: str) -> bool:
        """Check if user has required permission."""
        # Admins have all permissions
        if user.role == "admin":
            return True
        
        # Check API key permissions if present
        if hasattr(user, 'api_permissions'):
            return required_permission in user.api_permissions
        
        # Default role-based permissions
        role_permissions = {
            "admin": ["*"],
            "analyst": ["scenario:read", "scenario:write", "geri:read", "research:read"],
            "viewer": ["geri:read", "research:read"]
        }
        
        user_permissions = role_permissions.get(user.role, [])
        return "*" in user_permissions or required_permission in user_permissions
    
    async def logout_session(self, session_token: str):
        """Invalidate a session."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            await conn.execute("""
                DELETE FROM user_sessions WHERE session_token = $1
            """, session_token)
        finally:
            await conn.close()
    
    async def revoke_api_key(self, user_id: int, api_key_id: int):
        """Revoke an API key."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            await conn.execute("""
                UPDATE api_keys SET is_active = FALSE WHERE id = $1 AND user_id = $2
            """, api_key_id, user_id)
        finally:
            await conn.close()
    
    async def list_user_api_keys(self, user_id: int) -> List[APIKey]:
        """List API keys for a user."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            results = await conn.fetch("""
                SELECT id, user_id, name, permissions, last_used, expires_at, is_active
                FROM api_keys 
                WHERE user_id = $1
                ORDER BY created_at DESC
            """, user_id)
            
            return [
                APIKey(
                    id=result["id"],
                    user_id=result["user_id"],
                    name=result["name"],
                    permissions=result["permissions"],
                    last_used=result["last_used"],
                    expires_at=result["expires_at"],
                    is_active=result["is_active"]
                )
                for result in results
            ]
        finally:
            await conn.close()


# Global auth service instance
_auth_service: Optional[AuthService] = None

def get_auth_service() -> AuthService:
    """Dependency injection for auth service."""
    global _auth_service
    if _auth_service is None:
        postgres_dsn = os.environ.get("RIS_POSTGRES_DSN")
        if not postgres_dsn:
            raise RuntimeError("RIS_POSTGRES_DSN environment variable not set")
        _auth_service = AuthService(postgres_dsn)
    return _auth_service