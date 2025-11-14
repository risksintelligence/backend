"""Feature flag management service for controlling system capabilities."""
from __future__ import annotations

import asyncpg
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from backend.src.services.auth_service import User

@dataclass
class FeatureFlag:
    id: int
    name: str
    description: Optional[str]
    is_enabled: bool
    rollout_percentage: int
    target_roles: List[str]
    target_subscription_tiers: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class FeatureFlagService:
    """Production feature flag service with role and subscription targeting."""
    
    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self._cache: Dict[str, FeatureFlag] = {}
        self._cache_updated = datetime.min
        self._cache_ttl_seconds = 300  # 5 minutes
    
    async def create_feature_flag(
        self,
        name: str,
        description: Optional[str] = None,
        is_enabled: bool = False,
        rollout_percentage: int = 0,
        target_roles: List[str] = None,
        target_subscription_tiers: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> FeatureFlag:
        """Create a new feature flag."""
        target_roles = target_roles or []
        target_subscription_tiers = target_subscription_tiers or []
        metadata = metadata or {}
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                INSERT INTO feature_flags (name, description, is_enabled, rollout_percentage, 
                                         target_roles, target_subscription_tiers, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id, name, description, is_enabled, rollout_percentage, 
                         target_roles, target_subscription_tiers, metadata, created_at, updated_at
            """, name, description, is_enabled, rollout_percentage, 
                target_roles, target_subscription_tiers, metadata)
            
            flag = FeatureFlag(
                id=result["id"],
                name=result["name"],
                description=result["description"],
                is_enabled=result["is_enabled"],
                rollout_percentage=result["rollout_percentage"],
                target_roles=result["target_roles"],
                target_subscription_tiers=result["target_subscription_tiers"],
                metadata=result["metadata"],
                created_at=result["created_at"],
                updated_at=result["updated_at"]
            )
            
            # Clear cache
            self._cache_updated = datetime.min
            
            return flag
        finally:
            await conn.close()
    
    async def update_feature_flag(
        self,
        name: str,
        is_enabled: Optional[bool] = None,
        rollout_percentage: Optional[int] = None,
        target_roles: Optional[List[str]] = None,
        target_subscription_tiers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureFlag:
        """Update an existing feature flag."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            # Build dynamic update query
            updates = []
            params = [name]
            param_counter = 2
            
            if is_enabled is not None:
                updates.append(f"is_enabled = ${param_counter}")
                params.append(is_enabled)
                param_counter += 1
            
            if rollout_percentage is not None:
                updates.append(f"rollout_percentage = ${param_counter}")
                params.append(rollout_percentage)
                param_counter += 1
            
            if target_roles is not None:
                updates.append(f"target_roles = ${param_counter}")
                params.append(target_roles)
                param_counter += 1
            
            if target_subscription_tiers is not None:
                updates.append(f"target_subscription_tiers = ${param_counter}")
                params.append(target_subscription_tiers)
                param_counter += 1
            
            if metadata is not None:
                updates.append(f"metadata = ${param_counter}")
                params.append(metadata)
                param_counter += 1
            
            updates.append("updated_at = NOW()")
            
            if not updates:
                raise ValueError("No fields to update")
            
            query = f"""
                UPDATE feature_flags 
                SET {', '.join(updates)}
                WHERE name = $1
                RETURNING id, name, description, is_enabled, rollout_percentage,
                         target_roles, target_subscription_tiers, metadata, created_at, updated_at
            """
            
            result = await conn.fetchrow(query, *params)
            
            if not result:
                raise ValueError(f"Feature flag '{name}' not found")
            
            flag = FeatureFlag(
                id=result["id"],
                name=result["name"],
                description=result["description"],
                is_enabled=result["is_enabled"],
                rollout_percentage=result["rollout_percentage"],
                target_roles=result["target_roles"],
                target_subscription_tiers=result["target_subscription_tiers"],
                metadata=result["metadata"],
                created_at=result["created_at"],
                updated_at=result["updated_at"]
            )
            
            # Clear cache
            self._cache_updated = datetime.min
            
            return flag
        finally:
            await conn.close()
    
    async def _refresh_cache(self):
        """Refresh the feature flag cache from database."""
        now = datetime.now(timezone.utc)
        if (now - self._cache_updated).total_seconds() < self._cache_ttl_seconds:
            return  # Cache still valid
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            results = await conn.fetch("""
                SELECT id, name, description, is_enabled, rollout_percentage,
                       target_roles, target_subscription_tiers, metadata, created_at, updated_at
                FROM feature_flags
                ORDER BY name
            """)
            
            self._cache = {}
            for result in results:
                flag = FeatureFlag(
                    id=result["id"],
                    name=result["name"],
                    description=result["description"],
                    is_enabled=result["is_enabled"],
                    rollout_percentage=result["rollout_percentage"],
                    target_roles=result["target_roles"],
                    target_subscription_tiers=result["target_subscription_tiers"],
                    metadata=result["metadata"],
                    created_at=result["created_at"],
                    updated_at=result["updated_at"]
                )
                self._cache[flag.name] = flag
            
            self._cache_updated = now
        finally:
            await conn.close()
    
    async def is_enabled(self, flag_name: str, user: Optional[User] = None) -> bool:
        """Check if a feature flag is enabled for the given user."""
        await self._refresh_cache()
        
        flag = self._cache.get(flag_name)
        if not flag:
            return False  # Default to disabled for unknown flags
        
        if not flag.is_enabled:
            return False
        
        # If no user provided, just check global flag
        if not user:
            return True
        
        # Check role targeting
        if flag.target_roles and user.role not in flag.target_roles:
            return False
        
        # Check subscription tier targeting
        if flag.target_subscription_tiers and user.subscription_tier not in flag.target_subscription_tiers:
            return False
        
        # Check rollout percentage
        if flag.rollout_percentage < 100:
            # Use user ID for consistent assignment
            user_hash = hash(f"{flag.name}:{user.id}") % 100
            if user_hash >= flag.rollout_percentage:
                return False
        
        return True
    
    async def get_feature_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a specific feature flag."""
        await self._refresh_cache()
        return self._cache.get(name)
    
    async def list_feature_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        await self._refresh_cache()
        return list(self._cache.values())
    
    async def delete_feature_flag(self, name: str):
        """Delete a feature flag."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.execute("""
                DELETE FROM feature_flags WHERE name = $1
            """, name)
            
            if result == "DELETE 0":
                raise ValueError(f"Feature flag '{name}' not found")
            
            # Clear cache
            self._cache_updated = datetime.min
        finally:
            await conn.close()
    
    async def get_user_flags(self, user: User) -> Dict[str, bool]:
        """Get all feature flags for a specific user."""
        await self._refresh_cache()
        
        result = {}
        for flag_name, flag in self._cache.items():
            result[flag_name] = await self.is_enabled(flag_name, user)
        
        return result

# Global service instance
_feature_flag_service: Optional[FeatureFlagService] = None

def get_feature_flag_service() -> FeatureFlagService:
    """Dependency injection for feature flag service."""
    global _feature_flag_service
    if _feature_flag_service is None:
        postgres_dsn = os.environ.get("RIS_POSTGRES_DSN")
        if not postgres_dsn:
            raise RuntimeError("RIS_POSTGRES_DSN environment variable not set")
        _feature_flag_service = FeatureFlagService(postgres_dsn)
    return _feature_flag_service