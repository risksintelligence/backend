"""Advanced export service with multiple formats and sharing capabilities."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import asyncpg

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    PDF = "pdf"


class ExportScope(Enum):
    SCENARIO_RUNS = "scenario_runs"
    SAVED_SCENARIOS = "saved_scenarios"
    ALERT_HISTORY = "alert_history"
    COLLABORATION_ACTIVITY = "collaboration_activity"


@dataclass
class ExportRequest:
    user_id: int
    scope: ExportScope
    format: ExportFormat
    filters: Dict[str, Any]
    limit: int
    include_metadata: bool
    share_publicly: bool
    expires_in_hours: Optional[int]


@dataclass
class ExportResult:
    export_id: str
    file_path: str
    download_url: str
    file_size_bytes: int
    record_count: int
    created_at: datetime
    expires_at: Optional[datetime]
    is_public: bool


class AdvancedExportService:
    """Production export service with multiple formats and sharing capabilities."""

    def __init__(self, postgres_dsn: Optional[str] = None, export_base_path: str = "/tmp/ris_exports"):
        self.postgres_dsn = postgres_dsn or os.getenv("RIS_POSTGRES_DSN")
        if not self.postgres_dsn:
            raise ValueError("RIS_POSTGRES_DSN environment variable is required")
        
        self.export_base_path = export_base_path
        os.makedirs(export_base_path, exist_ok=True)

    async def create_export(self, request: ExportRequest) -> ExportResult:
        """Create an export based on the request parameters."""
        export_id = str(uuid.uuid4())
        
        # Fetch data based on scope
        data, metadata = await self._fetch_data(request)
        
        # Generate export file
        file_info = await self._generate_export_file(
            export_id, data, metadata, request.format, request.include_metadata
        )
        
        # Calculate expiry
        expires_at = None
        if request.expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=request.expires_in_hours)
        
        # Store export record
        await self._store_export_record(
            export_id=export_id,
            user_id=request.user_id,
            scope=request.scope,
            format=request.format,
            file_path=file_info["file_path"],
            file_size=file_info["file_size"],
            record_count=len(data),
            is_public=request.share_publicly,
            expires_at=expires_at,
            filters=request.filters
        )
        
        return ExportResult(
            export_id=export_id,
            file_path=file_info["file_path"],
            download_url=f"/api/v1/exports/{export_id}/download",
            file_size_bytes=file_info["file_size"],
            record_count=len(data),
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_public=request.share_publicly
        )

    async def _fetch_data(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fetch data based on export scope and filters."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            if request.scope == ExportScope.SCENARIO_RUNS:
                return await self._fetch_scenario_runs(conn, request)
            elif request.scope == ExportScope.SAVED_SCENARIOS:
                return await self._fetch_saved_scenarios(conn, request)
            elif request.scope == ExportScope.ALERT_HISTORY:
                return await self._fetch_alert_history(conn, request)
            elif request.scope == ExportScope.COLLABORATION_ACTIVITY:
                return await self._fetch_collaboration_activity(conn, request)
            else:
                raise ValueError(f"Unsupported export scope: {request.scope}")

    async def _fetch_scenario_runs(self, conn: asyncpg.Connection, request: ExportRequest) -> tuple[List[Dict], Dict]:
        """Fetch scenario run data."""
        where_clauses = ["1=1"]
        params = []
        param_count = 1
        
        # Apply filters
        if "date_from" in request.filters:
            where_clauses.append(f"created_at >= ${param_count}")
            params.append(request.filters["date_from"])
            param_count += 1
            
        if "date_to" in request.filters:
            where_clauses.append(f"created_at <= ${param_count}")
            params.append(request.filters["date_to"])
            param_count += 1
            
        if "min_delta" in request.filters:
            where_clauses.append(f"ABS(scenario_value - baseline_value) >= ${param_count}")
            params.append(request.filters["min_delta"])
            param_count += 1
        
        # Add user access control
        where_clauses.append(f"""
            (requester = 'internal' OR EXISTS (
                SELECT 1 FROM saved_scenarios s 
                WHERE s.user_id = ${param_count}
            ))
        """)
        params.append(request.user_id)
        param_count += 1
        
        query = f"""
            SELECT sr.*, 
                   CASE WHEN sr.requester = 'internal' THEN 'System' ELSE 'User' END as run_type
            FROM scenario_runs sr
            WHERE {' AND '.join(where_clauses)}
            ORDER BY sr.created_at DESC
            LIMIT ${param_count}
        """
        params.append(request.limit)
        
        rows = await conn.fetch(query, *params)
        
        data = []
        for row in rows:
            data.append({
                "id": row["id"],
                "created_at": row["created_at"].isoformat(),
                "run_type": row["run_type"],
                "shocks": row["shocks"],
                "horizon_hours": row["horizon_hours"],
                "baseline_value": float(row["baseline_value"]),
                "scenario_value": float(row["scenario_value"]),
                "delta": float(row["scenario_value"] - row["baseline_value"]),
                "requester": row["requester"]
            })
        
        metadata = {
            "export_scope": "scenario_runs",
            "total_records": len(data),
            "filters_applied": request.filters,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return data, metadata

    async def _fetch_saved_scenarios(self, conn: asyncpg.Connection, request: ExportRequest) -> tuple[List[Dict], Dict]:
        """Fetch saved scenario data."""
        where_clauses = [f"s.user_id = ${1}"]
        params = [request.user_id]
        param_count = 2
        
        # Apply filters
        if "include_shared" in request.filters and request.filters["include_shared"]:
            where_clauses = [f"""
                (s.user_id = ${1} OR s.is_public = TRUE OR EXISTS (
                    SELECT 1 FROM scenario_shares sh 
                    WHERE sh.scenario_id = s.id 
                    AND (sh.shared_with_user = ${1} OR sh.shared_with_email = (
                        SELECT email FROM users WHERE id = ${1}
                    ))
                    AND (sh.expires_at IS NULL OR sh.expires_at > NOW())
                ))
            """]
        
        if "is_public" in request.filters:
            where_clauses.append(f"s.is_public = ${param_count}")
            params.append(request.filters["is_public"])
            param_count += 1
        
        query = f"""
            SELECT s.*, u.username as owner_username,
                   (s.user_id = ${1}) as is_owner,
                   COALESCE(comment_stats.comment_count, 0) as comment_count,
                   COALESCE(share_stats.share_count, 0) as share_count
            FROM saved_scenarios s
            JOIN users u ON s.user_id = u.id
            LEFT JOIN (
                SELECT scenario_id, COUNT(*) as comment_count
                FROM scenario_comments
                GROUP BY scenario_id
            ) comment_stats ON s.id = comment_stats.scenario_id
            LEFT JOIN (
                SELECT scenario_id, COUNT(*) as share_count
                FROM scenario_shares
                WHERE expires_at IS NULL OR expires_at > NOW()
                GROUP BY scenario_id
            ) share_stats ON s.id = share_stats.scenario_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY s.updated_at DESC
            LIMIT ${param_count}
        """
        params.append(request.limit)
        
        rows = await conn.fetch(query, *params)
        
        data = []
        for row in rows:
            data.append({
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "owner_username": row["owner_username"],
                "is_owner": row["is_owner"],
                "shocks": row["shocks"],
                "horizon_hours": row["horizon_hours"],
                "baseline_value": float(row["baseline_value"]) if row["baseline_value"] else None,
                "scenario_value": float(row["scenario_value"]) if row["scenario_value"] else None,
                "is_public": row["is_public"],
                "comment_count": row["comment_count"],
                "share_count": row["share_count"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            })
        
        metadata = {
            "export_scope": "saved_scenarios",
            "total_records": len(data),
            "filters_applied": request.filters,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return data, metadata

    async def _fetch_alert_history(self, conn: asyncpg.Connection, request: ExportRequest) -> tuple[List[Dict], Dict]:
        """Fetch alert history data."""
        where_clauses = [f"ae.user_id = ${1}"]
        params = [request.user_id]
        param_count = 2
        
        # Apply filters
        if "date_from" in request.filters:
            where_clauses.append(f"ae.created_at >= ${param_count}")
            params.append(request.filters["date_from"])
            param_count += 1
            
        if "trigger_type" in request.filters:
            where_clauses.append(f"ae.trigger_type = ${param_count}")
            params.append(request.filters["trigger_type"])
            param_count += 1
        
        query = f"""
            SELECT ae.*, at.name as threshold_name, at.geri_threshold, at.delta_threshold
            FROM alert_events ae
            JOIN alert_thresholds at ON ae.threshold_id = at.id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY ae.created_at DESC
            LIMIT ${param_count}
        """
        params.append(request.limit)
        
        rows = await conn.fetch(query, *params)
        
        data = []
        for row in rows:
            data.append({
                "id": row["id"],
                "threshold_name": row["threshold_name"],
                "trigger_type": row["trigger_type"],
                "trigger_value": float(row["trigger_value"]),
                "geri_threshold": float(row["geri_threshold"]) if row["geri_threshold"] else None,
                "delta_threshold": float(row["delta_threshold"]) if row["delta_threshold"] else None,
                "scenario_data": row["scenario_data"],
                "created_at": row["created_at"].isoformat()
            })
        
        metadata = {
            "export_scope": "alert_history",
            "total_records": len(data),
            "filters_applied": request.filters,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return data, metadata

    async def _fetch_collaboration_activity(self, conn: asyncpg.Connection, request: ExportRequest) -> tuple[List[Dict], Dict]:
        """Fetch collaboration activity data."""
        # Get scenarios the user has access to
        accessible_scenarios = await conn.fetch("""
            SELECT DISTINCT s.id
            FROM saved_scenarios s
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
        """, request.user_id)
        
        scenario_ids = [row["id"] for row in accessible_scenarios]
        
        if not scenario_ids:
            return [], {"export_scope": "collaboration_activity", "total_records": 0}
        
        # Fetch activity data
        query = f"""
            SELECT sa.*, u.username, s.name as scenario_name
            FROM scenario_activity sa
            JOIN users u ON sa.user_id = u.id
            JOIN saved_scenarios s ON sa.scenario_id = s.id
            WHERE sa.scenario_id = ANY($1)
            ORDER BY sa.created_at DESC
            LIMIT $2
        """
        
        rows = await conn.fetch(query, scenario_ids, request.limit)
        
        data = []
        for row in rows:
            data.append({
                "id": row["id"],
                "scenario_id": row["scenario_id"],
                "scenario_name": row["scenario_name"],
                "username": row["username"],
                "activity_type": row["activity_type"],
                "activity_data": row["activity_data"],
                "created_at": row["created_at"].isoformat()
            })
        
        metadata = {
            "export_scope": "collaboration_activity",
            "total_records": len(data),
            "accessible_scenarios": len(scenario_ids),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return data, metadata

    async def _generate_export_file(
        self, 
        export_id: str, 
        data: List[Dict[str, Any]], 
        metadata: Dict[str, Any], 
        format: ExportFormat, 
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Generate export file in the specified format."""
        filename = f"{export_id}.{format.value}"
        file_path = os.path.join(self.export_base_path, filename)
        
        if format == ExportFormat.CSV:
            return await self._generate_csv(file_path, data, metadata, include_metadata)
        elif format == ExportFormat.JSON:
            return await self._generate_json(file_path, data, metadata, include_metadata)
        elif format == ExportFormat.EXCEL and EXCEL_AVAILABLE:
            return await self._generate_excel(file_path, data, metadata, include_metadata)
        else:
            raise ValueError(f"Unsupported or unavailable export format: {format}")

    async def _generate_csv(self, file_path: str, data: List[Dict], metadata: Dict, include_metadata: bool) -> Dict:
        """Generate CSV export file."""
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            if include_metadata:
                # Write metadata as comments
                csvfile.write(f"# Export Generated: {metadata.get('generated_at', 'Unknown')}\n")
                csvfile.write(f"# Total Records: {metadata.get('total_records', 0)}\n")
                csvfile.write(f"# Export Scope: {metadata.get('export_scope', 'Unknown')}\n")
                if metadata.get('filters_applied'):
                    csvfile.write(f"# Filters Applied: {json.dumps(metadata['filters_applied'])}\n")
                csvfile.write("#\n")
            
            if data:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in data:
                    # Convert complex objects to JSON strings for CSV
                    csv_row = {}
                    for key, value in row.items():
                        if isinstance(value, (dict, list)):
                            csv_row[key] = json.dumps(value)
                        else:
                            csv_row[key] = value
                    writer.writerow(csv_row)
        
        file_size = os.path.getsize(file_path)
        return {"file_path": file_path, "file_size": file_size}

    async def _generate_json(self, file_path: str, data: List[Dict], metadata: Dict, include_metadata: bool) -> Dict:
        """Generate JSON export file."""
        export_data = {
            "data": data
        }
        
        if include_metadata:
            export_data["metadata"] = metadata
        
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)
        
        file_size = os.path.getsize(file_path)
        return {"file_path": file_path, "file_size": file_size}

    async def _generate_excel(self, file_path: str, data: List[Dict], metadata: Dict, include_metadata: bool) -> Dict:
        """Generate Excel export file."""
        if not PANDAS_AVAILABLE:
            raise ValueError("pandas is required for Excel export")
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Main data sheet
            if data:
                df = pd.DataFrame(data)
                # Convert complex objects to JSON strings
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
                df.to_excel(writer, sheet_name='Data', index=False)
            
            # Metadata sheet if requested
            if include_metadata:
                metadata_rows = []
                for key, value in metadata.items():
                    metadata_rows.append({"Property": key, "Value": str(value)})
                
                if metadata_rows:
                    metadata_df = pd.DataFrame(metadata_rows)
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        file_size = os.path.getsize(file_path)
        return {"file_path": file_path, "file_size": file_size}

    async def _store_export_record(self, **kwargs) -> None:
        """Store export record in database."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            await conn.execute("""
                INSERT INTO export_records (
                    export_id, user_id, scope, format, file_path, file_size_bytes,
                    record_count, is_public, expires_at, filters, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            """, 
                kwargs["export_id"],
                kwargs["user_id"], 
                kwargs["scope"].value,
                kwargs["format"].value,
                kwargs["file_path"],
                kwargs["file_size"],
                kwargs["record_count"],
                kwargs["is_public"],
                kwargs["expires_at"],
                json.dumps(kwargs["filters"])
            )

    async def get_export(self, export_id: str, requesting_user_id: Optional[int] = None) -> Optional[ExportResult]:
        """Get export information and check access permissions."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            export_data = await conn.fetchrow("""
                SELECT * FROM export_records WHERE export_id = $1
            """, export_id)
            
            if not export_data:
                return None
            
            # Check access permissions
            has_access = (
                export_data["is_public"] or
                (requesting_user_id and export_data["user_id"] == requesting_user_id)
            )
            
            if not has_access:
                return None
            
            # Check if expired
            if export_data["expires_at"] and export_data["expires_at"] <= datetime.utcnow():
                return None
            
            return ExportResult(
                export_id=export_data["export_id"],
                file_path=export_data["file_path"],
                download_url=f"/api/v1/exports/{export_id}/download",
                file_size_bytes=export_data["file_size_bytes"],
                record_count=export_data["record_count"],
                created_at=export_data["created_at"],
                expires_at=export_data["expires_at"],
                is_public=export_data["is_public"]
            )

    async def get_user_exports(self, user_id: int, limit: int = 50) -> List[ExportResult]:
        """Get all exports for a user."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            exports_data = await conn.fetch("""
                SELECT * FROM export_records 
                WHERE user_id = $1 
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            results = []
            for row in exports_data:
                results.append(ExportResult(
                    export_id=row["export_id"],
                    file_path=row["file_path"],
                    download_url=f"/api/v1/exports/{row['export_id']}/download",
                    file_size_bytes=row["file_size_bytes"],
                    record_count=row["record_count"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    is_public=row["is_public"]
                ))
            
            return results

    async def delete_export(self, export_id: str, user_id: int) -> bool:
        """Delete an export (only by owner)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            export_data = await conn.fetchrow("""
                SELECT file_path FROM export_records 
                WHERE export_id = $1 AND user_id = $2
            """, export_id, user_id)
            
            if not export_data:
                return False
            
            # Delete file
            try:
                if os.path.exists(export_data["file_path"]):
                    os.remove(export_data["file_path"])
            except Exception as e:
                logger.warning(f"Failed to delete export file {export_data['file_path']}: {e}")
            
            # Delete database record
            await conn.execute("""
                DELETE FROM export_records WHERE export_id = $1 AND user_id = $2
            """, export_id, user_id)
            
            return True

    async def cleanup_expired_exports(self) -> int:
        """Cleanup expired exports (background task)."""
        async with asyncpg.connect(self.postgres_dsn) as conn:
            expired_exports = await conn.fetch("""
                SELECT export_id, file_path FROM export_records
                WHERE expires_at IS NOT NULL AND expires_at <= NOW()
            """)
            
            deleted_count = 0
            for export in expired_exports:
                try:
                    if os.path.exists(export["file_path"]):
                        os.remove(export["file_path"])
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete expired export file {export['file_path']}: {e}")
            
            # Delete database records
            await conn.execute("""
                DELETE FROM export_records 
                WHERE expires_at IS NOT NULL AND expires_at <= NOW()
            """)
            
            return deleted_count


# Singleton service instance
_EXPORT_SERVICE: AdvancedExportService | None = None


def get_advanced_export_service() -> AdvancedExportService:
    """Get or create the advanced export service singleton."""
    global _EXPORT_SERVICE
    if _EXPORT_SERVICE is None:
        _EXPORT_SERVICE = AdvancedExportService()
    return _EXPORT_SERVICE