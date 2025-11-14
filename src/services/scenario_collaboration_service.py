"""Production scenario collaboration service with comments and activity tracking."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncpg
import os

from backend.src.services.scenario_sharing_service import get_scenario_sharing_service, PermissionLevel, SharingError

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    CREATED = "created"
    UPDATED = "updated"
    SHARED = "shared"
    COMMENTED = "commented"
    FORKED = "forked"
    VISIBILITY_CHANGED = "visibility_changed"


@dataclass
class ScenarioComment:
    id: int
    scenario_id: int
    user_id: int
    username: str
    comment_text: str
    is_resolved: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class ScenarioActivity:
    id: int
    scenario_id: int
    user_id: int
    username: str
    activity_type: ActivityType
    activity_data: Dict[str, Any]
    created_at: datetime


class ScenarioCollaborationService:
    """Production collaboration service for scenarios with comments and activity tracking."""

    def __init__(self, postgres_dsn: Optional[str] = None):
        self.postgres_dsn = postgres_dsn or os.getenv("RIS_POSTGRES_DSN")
        if not self.postgres_dsn:
            raise ValueError("RIS_POSTGRES_DSN environment variable is required")
        
        self.sharing_service = get_scenario_sharing_service()

    async def add_comment(
        self,
        scenario_id: int,
        user_id: int,
        comment_text: str
    ) -> ScenarioComment:
        """Add a comment to a scenario."""
        # Check if user has access to the scenario
        permission = await self.sharing_service.check_permission(scenario_id, user_id)
        if not permission:
            raise SharingError("Access denied: Cannot comment on this scenario")

        async with asyncpg.connect(self.postgres_dsn) as conn:
            comment_id = await conn.fetchval("""
                INSERT INTO scenario_comments (scenario_id, user_id, comment_text)
                VALUES ($1, $2, $3)
                RETURNING id
            """, scenario_id, user_id, comment_text)

            # Log activity
            await self._log_activity(
                conn, scenario_id, user_id, ActivityType.COMMENTED,
                {"comment_id": comment_id, "comment_preview": comment_text[:100]}
            )

            # Get the comment with username
            comment_data = await conn.fetchrow("""
                SELECT c.*, u.username
                FROM scenario_comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.id = $1
            """, comment_id)

            return ScenarioComment(
                id=comment_data["id"],
                scenario_id=comment_data["scenario_id"],
                user_id=comment_data["user_id"],
                username=comment_data["username"],
                comment_text=comment_data["comment_text"],
                is_resolved=comment_data["is_resolved"],
                created_at=comment_data["created_at"],
                updated_at=comment_data["updated_at"]
            )

    async def get_scenario_comments(
        self,
        scenario_id: int,
        user_id: int,
        include_resolved: bool = True
    ) -> List[ScenarioComment]:
        """Get all comments for a scenario."""
        # Check if user has access to the scenario
        permission = await self.sharing_service.check_permission(scenario_id, user_id)
        if not permission:
            raise SharingError("Access denied: Cannot view comments for this scenario")

        async with asyncpg.connect(self.postgres_dsn) as conn:
            where_clause = "WHERE c.scenario_id = $1"
            params = [scenario_id]
            
            if not include_resolved:
                where_clause += " AND c.is_resolved = FALSE"

            comments_data = await conn.fetch(f"""
                SELECT c.*, u.username
                FROM scenario_comments c
                JOIN users u ON c.user_id = u.id
                {where_clause}
                ORDER BY c.created_at ASC
            """, *params)

            comments = []
            for row in comments_data:
                comments.append(ScenarioComment(
                    id=row["id"],
                    scenario_id=row["scenario_id"],
                    user_id=row["user_id"],
                    username=row["username"],
                    comment_text=row["comment_text"],
                    is_resolved=row["is_resolved"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                ))

            return comments

    async def update_comment(
        self,
        comment_id: int,
        user_id: int,
        comment_text: Optional[str] = None,
        is_resolved: Optional[bool] = None
    ) -> ScenarioComment:
        """Update a comment (only by comment author or scenario owner)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Check if user can update this comment
            comment_data = await conn.fetchrow("""
                SELECT c.*, s.user_id as scenario_owner
                FROM scenario_comments c
                JOIN saved_scenarios s ON c.scenario_id = s.id
                WHERE c.id = $1
            """, comment_id)

            if not comment_data:
                raise SharingError("Comment not found")

            # User can update if they're the comment author or scenario owner
            can_update = (
                comment_data["user_id"] == user_id or 
                comment_data["scenario_owner"] == user_id
            )

            if not can_update:
                raise SharingError("Insufficient permissions to update comment")

            # Build update query
            update_fields = []
            values = []
            param_count = 1

            if comment_text is not None:
                update_fields.append(f"comment_text = ${param_count}")
                values.append(comment_text)
                param_count += 1

            if is_resolved is not None:
                update_fields.append(f"is_resolved = ${param_count}")
                values.append(is_resolved)
                param_count += 1

            if not update_fields:
                # No updates, return current comment
                return await self.get_comment(comment_id, user_id)

            update_fields.append("updated_at = NOW()")
            values.append(comment_id)

            query = f"""
                UPDATE scenario_comments 
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
            """

            await conn.execute(query, *values)
            return await self.get_comment(comment_id, user_id)

    async def get_comment(self, comment_id: int, user_id: int) -> ScenarioComment:
        """Get a specific comment."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            comment_data = await conn.fetchrow("""
                SELECT c.*, u.username
                FROM scenario_comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.id = $1
            """, comment_id)

            if not comment_data:
                raise SharingError("Comment not found")

            # Check if user has access to the scenario
            permission = await self.sharing_service.check_permission(
                comment_data["scenario_id"], user_id
            )
            if not permission:
                raise SharingError("Access denied")

            return ScenarioComment(
                id=comment_data["id"],
                scenario_id=comment_data["scenario_id"],
                user_id=comment_data["user_id"],
                username=comment_data["username"],
                comment_text=comment_data["comment_text"],
                is_resolved=comment_data["is_resolved"],
                created_at=comment_data["created_at"],
                updated_at=comment_data["updated_at"]
            )

    async def delete_comment(self, comment_id: int, user_id: int) -> bool:
        """Delete a comment (only by comment author or scenario owner)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            # Check if user can delete this comment
            comment_data = await conn.fetchrow("""
                SELECT c.*, s.user_id as scenario_owner
                FROM scenario_comments c
                JOIN saved_scenarios s ON c.scenario_id = s.id
                WHERE c.id = $1
            """, comment_id)

            if not comment_data:
                raise SharingError("Comment not found")

            can_delete = (
                comment_data["user_id"] == user_id or 
                comment_data["scenario_owner"] == user_id
            )

            if not can_delete:
                raise SharingError("Insufficient permissions to delete comment")

            await conn.execute("DELETE FROM scenario_comments WHERE id = $1", comment_id)
            return True

    async def get_scenario_activity(
        self,
        scenario_id: int,
        user_id: int,
        limit: int = 50
    ) -> List[ScenarioActivity]:
        """Get activity history for a scenario."""
        # Check if user has access to the scenario
        permission = await self.sharing_service.check_permission(scenario_id, user_id)
        if not permission:
            raise SharingError("Access denied: Cannot view activity for this scenario")

        async with asyncpg.connect(self.postgres_dsn) as conn:
            activities_data = await conn.fetch("""
                SELECT a.*, u.username
                FROM scenario_activity a
                JOIN users u ON a.user_id = u.id
                WHERE a.scenario_id = $1
                ORDER BY a.created_at DESC
                LIMIT $2
            """, scenario_id, limit)

            activities = []
            for row in activities_data:
                activities.append(ScenarioActivity(
                    id=row["id"],
                    scenario_id=row["scenario_id"],
                    user_id=row["user_id"],
                    username=row["username"],
                    activity_type=ActivityType(row["activity_type"]),
                    activity_data=row["activity_data"] or {},
                    created_at=row["created_at"]
                ))

            return activities

    async def log_activity(
        self,
        scenario_id: int,
        user_id: int,
        activity_type: ActivityType,
        activity_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an activity for a scenario."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            await self._log_activity(conn, scenario_id, user_id, activity_type, activity_data)

    async def _log_activity(
        self,
        conn: asyncpg.Connection,
        scenario_id: int,
        user_id: int,
        activity_type: ActivityType,
        activity_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal method to log activity within a transaction."""
        await conn.execute("""
            INSERT INTO scenario_activity (scenario_id, user_id, activity_type, activity_data)
            VALUES ($1, $2, $3, $4)
        """, scenario_id, user_id, activity_type.value, activity_data or {})

    async def fork_scenario(
        self,
        scenario_id: int,
        user_id: int,
        new_name: str,
        new_description: Optional[str] = None
    ) -> int:
        """Create a copy of a scenario (fork)."""
        # Check if user has access to the scenario
        permission = await self.sharing_service.check_permission(scenario_id, user_id)
        if not permission:
            raise SharingError("Access denied: Cannot fork this scenario")

        # Get the original scenario
        original_scenario = await self.sharing_service.get_scenario(scenario_id, user_id)

        # Create the forked scenario
        forked_scenario = await self.sharing_service.save_scenario(
            user_id=user_id,
            name=new_name,
            description=new_description or f"Forked from: {original_scenario.name}",
            shocks=original_scenario.shocks,
            horizon_hours=original_scenario.horizon_hours,
            baseline_value=original_scenario.baseline_value,
            scenario_value=original_scenario.scenario_value,
            is_public=False  # Forks are private by default
        )

        # Log activity on both scenarios
        await self.log_activity(
            scenario_id, user_id, ActivityType.FORKED,
            {"forked_to": forked_scenario.id, "forked_name": new_name}
        )

        await self.log_activity(
            forked_scenario.id, user_id, ActivityType.CREATED,
            {"forked_from": scenario_id, "original_name": original_scenario.name}
        )

        return forked_scenario.id

    async def get_scenario_statistics(self, scenario_id: int, user_id: int) -> Dict[str, Any]:
        """Get collaboration statistics for a scenario."""
        # Check if user has access to the scenario
        permission = await self.sharing_service.check_permission(scenario_id, user_id)
        if not permission:
            raise SharingError("Access denied")

        async with asyncpg.connect(self.postgres_dsn) as conn:
            stats = {}

            # Comment statistics
            comment_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_comments,
                    COUNT(*) FILTER (WHERE is_resolved = FALSE) as open_comments,
                    COUNT(DISTINCT user_id) as unique_commenters
                FROM scenario_comments
                WHERE scenario_id = $1
            """, scenario_id)

            stats["comments"] = {
                "total": comment_stats["total_comments"],
                "open": comment_stats["open_comments"],
                "resolved": comment_stats["total_comments"] - comment_stats["open_comments"],
                "unique_commenters": comment_stats["unique_commenters"]
            }

            # Activity statistics
            activity_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_activities,
                    COUNT(DISTINCT user_id) as unique_contributors
                FROM scenario_activity
                WHERE scenario_id = $1
            """, scenario_id)

            stats["activity"] = {
                "total_activities": activity_stats["total_activities"],
                "unique_contributors": activity_stats["unique_contributors"]
            }

            # Share statistics (only for scenario owner)
            scenario = await self.sharing_service.get_scenario(scenario_id, user_id)
            if scenario.user_id == user_id:
                share_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_shares,
                        COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) as active_shares
                    FROM scenario_shares
                    WHERE scenario_id = $1
                """, scenario_id)

                stats["shares"] = {
                    "total": share_stats["total_shares"],
                    "active": share_stats["active_shares"]
                }

            return stats


# Singleton service instance
_COLLABORATION_SERVICE: ScenarioCollaborationService | None = None


def get_scenario_collaboration_service() -> ScenarioCollaborationService:
    """Get or create the scenario collaboration service singleton."""
    global _COLLABORATION_SERVICE
    if _COLLABORATION_SERVICE is None:
        _COLLABORATION_SERVICE = ScenarioCollaborationService()
    return _COLLABORATION_SERVICE