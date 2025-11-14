"""Production scenario sharing and collaboration service with comprehensive access control."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncpg
import os

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"


class SharingError(Exception):
    """Base exception for scenario sharing operations."""
    pass


@dataclass
class SavedScenario:
    id: int
    user_id: int
    name: str
    description: Optional[str]
    shocks: List[Dict[str, Any]]
    horizon_hours: int
    baseline_value: Optional[float]
    scenario_value: Optional[float]
    is_public: bool
    shared_with: List[int]
    created_at: datetime
    updated_at: datetime


@dataclass
class ScenarioShare:
    id: int
    scenario_id: int
    shared_by: int
    shared_with_user: Optional[int]
    shared_with_email: Optional[str]
    permission_level: PermissionLevel
    expires_at: Optional[datetime]
    created_at: datetime


@dataclass
class SharedScenarioView:
    scenario: SavedScenario
    owner_username: str
    permission_level: PermissionLevel
    shared_at: datetime
    expires_at: Optional[datetime]


class ScenarioSharingService:
    """Production scenario sharing service with comprehensive collaboration features."""

    def __init__(self, postgres_dsn: Optional[str] = None):
        self.postgres_dsn = postgres_dsn or os.getenv("RIS_POSTGRES_DSN")
        if not self.postgres_dsn:
            raise ValueError("RIS_POSTGRES_DSN environment variable is required")

    async def save_scenario(
        self,
        user_id: int,
        name: str,
        shocks: List[Dict[str, Any]],
        horizon_hours: int,
        description: Optional[str] = None,
        baseline_value: Optional[float] = None,
        scenario_value: Optional[float] = None,
        is_public: bool = False
    ) -> SavedScenario:
        """Save a scenario to the database with sharing metadata."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            scenario_id = await conn.fetchval("""
                INSERT INTO saved_scenarios (
                    user_id, name, description, shocks, horizon_hours, 
                    baseline_value, scenario_value, is_public
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, user_id, name, description, shocks, horizon_hours, baseline_value, scenario_value, is_public)
            
            return await self.get_scenario(scenario_id, user_id)

    async def get_scenario(self, scenario_id: int, requesting_user_id: int) -> SavedScenario:
        """Get a scenario if the user has access to it."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Check if user owns the scenario or has shared access
            scenario_data = await conn.fetchrow("""
                SELECT s.* FROM saved_scenarios s
                WHERE s.id = $1 AND (
                    s.user_id = $2 OR 
                    s.is_public = TRUE OR
                    EXISTS (
                        SELECT 1 FROM scenario_shares sh 
                        WHERE sh.scenario_id = s.id 
                        AND (sh.shared_with_user = $2 OR sh.shared_with_email = (
                            SELECT email FROM users WHERE id = $2
                        ))
                        AND (sh.expires_at IS NULL OR sh.expires_at > NOW())
                    )
                )
            """, scenario_id, requesting_user_id)
            
            if not scenario_data:
                raise SharingError("Scenario not found or access denied")
            
            return SavedScenario(
                id=scenario_data["id"],
                user_id=scenario_data["user_id"],
                name=scenario_data["name"],
                description=scenario_data["description"],
                shocks=scenario_data["shocks"],
                horizon_hours=scenario_data["horizon_hours"],
                baseline_value=scenario_data["baseline_value"],
                scenario_value=scenario_data["scenario_value"],
                is_public=scenario_data["is_public"],
                shared_with=scenario_data["shared_with"] or [],
                created_at=scenario_data["created_at"],
                updated_at=scenario_data["updated_at"]
            )

    async def update_scenario(
        self,
        scenario_id: int,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        shocks: Optional[List[Dict[str, Any]]] = None,
        horizon_hours: Optional[int] = None,
        baseline_value: Optional[float] = None,
        scenario_value: Optional[float] = None
    ) -> SavedScenario:
        """Update a scenario if the user has edit access."""
        # Check if user has edit access
        permission = await self.check_permission(scenario_id, user_id)
        if permission not in [PermissionLevel.EDIT, PermissionLevel.ADMIN]:
            raise SharingError("Insufficient permissions to edit scenario")
        
        async with asyncpg.connect(self.postgres_dsn) as conn:
            update_fields = []
            values = []
            param_count = 1
            
            if name is not None:
                update_fields.append(f"name = ${param_count}")
                values.append(name)
                param_count += 1
                
            if description is not None:
                update_fields.append(f"description = ${param_count}")
                values.append(description)
                param_count += 1
                
            if shocks is not None:
                update_fields.append(f"shocks = ${param_count}")
                values.append(shocks)
                param_count += 1
                
            if horizon_hours is not None:
                update_fields.append(f"horizon_hours = ${param_count}")
                values.append(horizon_hours)
                param_count += 1
                
            if baseline_value is not None:
                update_fields.append(f"baseline_value = ${param_count}")
                values.append(baseline_value)
                param_count += 1
                
            if scenario_value is not None:
                update_fields.append(f"scenario_value = ${param_count}")
                values.append(scenario_value)
                param_count += 1
            
            if not update_fields:
                return await self.get_scenario(scenario_id, user_id)
            
            update_fields.append("updated_at = NOW()")
            values.extend([scenario_id])
            
            query = f"""
                UPDATE saved_scenarios 
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
            """
            
            await conn.execute(query, *values)
            return await self.get_scenario(scenario_id, user_id)

    async def share_scenario(
        self,
        scenario_id: int,
        owner_user_id: int,
        shared_with_user_id: Optional[int] = None,
        shared_with_email: Optional[str] = None,
        permission_level: PermissionLevel = PermissionLevel.VIEW,
        expires_in_hours: Optional[int] = None
    ) -> ScenarioShare:
        """Share a scenario with another user or email address."""
        # Check if user owns the scenario or has admin access
        permission = await self.check_permission(scenario_id, owner_user_id)
        if permission != PermissionLevel.ADMIN:
            # Check if user owns the scenario
            scenario = await self.get_scenario(scenario_id, owner_user_id)
            if scenario.user_id != owner_user_id:
                raise SharingError("Only scenario owners can share scenarios")
        
        if not shared_with_user_id and not shared_with_email:
            raise SharingError("Must specify either user_id or email for sharing")
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        async with asyncpg.connect(self.postgres_dsn) as conn:
            share_id = await conn.fetchval("""
                INSERT INTO scenario_shares (
                    scenario_id, shared_by, shared_with_user, shared_with_email,
                    permission_level, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, scenario_id, owner_user_id, shared_with_user_id, shared_with_email, 
                permission_level.value, expires_at)
            
            share_data = await conn.fetchrow("""
                SELECT * FROM scenario_shares WHERE id = $1
            """, share_id)
            
            return ScenarioShare(
                id=share_data["id"],
                scenario_id=share_data["scenario_id"],
                shared_by=share_data["shared_by"],
                shared_with_user=share_data["shared_with_user"],
                shared_with_email=share_data["shared_with_email"],
                permission_level=PermissionLevel(share_data["permission_level"]),
                expires_at=share_data["expires_at"],
                created_at=share_data["created_at"]
            )

    async def revoke_share(self, share_id: int, user_id: int) -> bool:
        """Revoke a scenario share."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Check if user is the one who shared it or the scenario owner
            result = await conn.fetchrow("""
                SELECT sh.*, s.user_id as scenario_owner
                FROM scenario_shares sh
                JOIN saved_scenarios s ON sh.scenario_id = s.id
                WHERE sh.id = $1
            """, share_id)
            
            if not result:
                raise SharingError("Share not found")
            
            if result["shared_by"] != user_id and result["scenario_owner"] != user_id:
                raise SharingError("Insufficient permissions to revoke share")
            
            await conn.execute("DELETE FROM scenario_shares WHERE id = $1", share_id)
            return True

    async def get_user_scenarios(self, user_id: int, include_shared: bool = True) -> List[SavedScenario]:
        """Get all scenarios accessible to a user (owned + shared)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            if include_shared:
                scenarios_data = await conn.fetch("""
                    SELECT DISTINCT s.* FROM saved_scenarios s
                    WHERE s.user_id = $1 
                    OR s.is_public = TRUE
                    OR EXISTS (
                        SELECT 1 FROM scenario_shares sh 
                        WHERE sh.scenario_id = s.id 
                        AND (sh.shared_with_user = $1 OR sh.shared_with_email = (
                            SELECT email FROM users WHERE id = $1
                        ))
                        AND (sh.expires_at IS NULL OR sh.expires_at > NOW())
                    )
                    ORDER BY s.updated_at DESC
                """, user_id)
            else:
                scenarios_data = await conn.fetch("""
                    SELECT * FROM saved_scenarios WHERE user_id = $1 ORDER BY updated_at DESC
                """, user_id)
            
            scenarios = []
            for row in scenarios_data:
                scenarios.append(SavedScenario(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    description=row["description"],
                    shocks=row["shocks"],
                    horizon_hours=row["horizon_hours"],
                    baseline_value=row["baseline_value"],
                    scenario_value=row["scenario_value"],
                    is_public=row["is_public"],
                    shared_with=row["shared_with"] or [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                ))
            
            return scenarios

    async def get_shared_scenarios(self, user_id: int) -> List[SharedScenarioView]:
        """Get scenarios shared with the user with sharing details."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            user_email = await conn.fetchval("SELECT email FROM users WHERE id = $1", user_id)
            
            shares_data = await conn.fetch("""
                SELECT s.*, u.username as owner_username, sh.permission_level, 
                       sh.created_at as shared_at, sh.expires_at
                FROM saved_scenarios s
                JOIN scenario_shares sh ON s.id = sh.scenario_id
                JOIN users u ON s.user_id = u.id
                WHERE (sh.shared_with_user = $1 OR sh.shared_with_email = $2)
                AND (sh.expires_at IS NULL OR sh.expires_at > NOW())
                ORDER BY sh.created_at DESC
            """, user_id, user_email)
            
            shared_scenarios = []
            for row in shares_data:
                scenario = SavedScenario(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    description=row["description"],
                    shocks=row["shocks"],
                    horizon_hours=row["horizon_hours"],
                    baseline_value=row["baseline_value"],
                    scenario_value=row["scenario_value"],
                    is_public=row["is_public"],
                    shared_with=row["shared_with"] or [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                
                shared_scenarios.append(SharedScenarioView(
                    scenario=scenario,
                    owner_username=row["owner_username"],
                    permission_level=PermissionLevel(row["permission_level"]),
                    shared_at=row["shared_at"],
                    expires_at=row["expires_at"]
                ))
            
            return shared_scenarios

    async def get_scenario_shares(self, scenario_id: int, user_id: int) -> List[ScenarioShare]:
        """Get all shares for a scenario (only accessible to scenario owner)."""
        scenario = await self.get_scenario(scenario_id, user_id)
        if scenario.user_id != user_id:
            raise SharingError("Only scenario owners can view shares")
        
        async with asyncpg.connect(self.postgres_dsn) as conn:
            shares_data = await conn.fetch("""
                SELECT * FROM scenario_shares 
                WHERE scenario_id = $1 
                ORDER BY created_at DESC
            """, scenario_id)
            
            shares = []
            for row in shares_data:
                shares.append(ScenarioShare(
                    id=row["id"],
                    scenario_id=row["scenario_id"],
                    shared_by=row["shared_by"],
                    shared_with_user=row["shared_with_user"],
                    shared_with_email=row["shared_with_email"],
                    permission_level=PermissionLevel(row["permission_level"]),
                    expires_at=row["expires_at"],
                    created_at=row["created_at"]
                ))
            
            return shares

    async def check_permission(self, scenario_id: int, user_id: int) -> Optional[PermissionLevel]:
        """Check the user's permission level for a scenario."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Check if user owns the scenario
            owner_check = await conn.fetchval("""
                SELECT user_id FROM saved_scenarios WHERE id = $1 AND user_id = $2
            """, scenario_id, user_id)
            
            if owner_check:
                return PermissionLevel.ADMIN
            
            # Check shared permissions
            user_email = await conn.fetchval("SELECT email FROM users WHERE id = $1", user_id)
            
            permission = await conn.fetchval("""
                SELECT permission_level FROM scenario_shares
                WHERE scenario_id = $1 
                AND (shared_with_user = $2 OR shared_with_email = $3)
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY 
                    CASE permission_level 
                        WHEN 'admin' THEN 1 
                        WHEN 'edit' THEN 2 
                        WHEN 'view' THEN 3 
                    END
                LIMIT 1
            """, scenario_id, user_id, user_email)
            
            if permission:
                return PermissionLevel(permission)
            
            # Check if scenario is public
            is_public = await conn.fetchval("""
                SELECT is_public FROM saved_scenarios WHERE id = $1
            """, scenario_id)
            
            if is_public:
                return PermissionLevel.VIEW
            
            return None

    async def delete_scenario(self, scenario_id: int, user_id: int) -> bool:
        """Delete a scenario (only by owner)."""
        scenario = await self.get_scenario(scenario_id, user_id)
        if scenario.user_id != user_id:
            raise SharingError("Only scenario owners can delete scenarios")
        
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Cascade delete will handle scenario_shares
            await conn.execute("DELETE FROM saved_scenarios WHERE id = $1", scenario_id)
            return True

    async def make_scenario_public(self, scenario_id: int, user_id: int, is_public: bool = True) -> SavedScenario:
        """Make a scenario public or private (only by owner)."""
        scenario = await self.get_scenario(scenario_id, user_id)
        if scenario.user_id != user_id:
            raise SharingError("Only scenario owners can change visibility")
        
        async with asyncpg.connect(self.postgres_dsn) as conn:
            await conn.execute("""
                UPDATE saved_scenarios 
                SET is_public = $1, updated_at = NOW()
                WHERE id = $2
            """, is_public, scenario_id)
            
            return await self.get_scenario(scenario_id, user_id)

    async def cleanup_expired_shares(self) -> int:
        """Clean up expired shares (background task)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            deleted_count = await conn.fetchval("""
                DELETE FROM scenario_shares 
                WHERE expires_at IS NOT NULL AND expires_at <= NOW()
                RETURNING COUNT(*)
            """)
            
            return deleted_count or 0


# Singleton service instance
_SCENARIO_SHARING_SERVICE: ScenarioSharingService | None = None


def get_scenario_sharing_service() -> ScenarioSharingService:
    """Get or create the scenario sharing service singleton."""
    global _SCENARIO_SHARING_SERVICE
    if _SCENARIO_SHARING_SERVICE is None:
        _SCENARIO_SHARING_SERVICE = ScenarioSharingService()
    return _SCENARIO_SHARING_SERVICE