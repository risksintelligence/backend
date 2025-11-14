"""Admin service providing system monitoring and management data."""
from __future__ import annotations

import asyncio
import asyncpg
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import psutil

logger = logging.getLogger(__name__)

class AdminService:
    """Production admin service with real system data."""
    
    def __init__(self, postgres_dsn: str, render_api_key: Optional[str] = None):
        self.postgres_dsn = postgres_dsn
        self.render_api_key = render_api_key
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            # Database health
            db_connections = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()))")
            
            # Data freshness
            latest_geri = await conn.fetchrow("""
                SELECT ts_utc, value, band, created_at 
                FROM computed_indices 
                ORDER BY ts_utc DESC LIMIT 1
            """)
            
            # Recent activity counts
            activity = await conn.fetchrow("""
                SELECT 
                    (SELECT count(*) FROM scenario_runs WHERE created_at > NOW() - INTERVAL '24 hours') as scenarios_24h,
                    (SELECT count(*) FROM peer_reviews WHERE created_at > NOW() - INTERVAL '7 days') as reviews_7d,
                    (SELECT count(*) FROM research_api_requests WHERE requested_at > NOW() - INTERVAL '24 hours') as api_requests_24h,
                    (SELECT count(*) FROM alert_subscriptions WHERE active = true) as active_alerts,
                    (SELECT count(*) FROM admin_audit_log WHERE occurred_at > NOW() - INTERVAL '24 hours') as admin_actions_24h
            """)
            
            # ML Model status
            models = await conn.fetch("""
                SELECT model_name, version, created_at, metadata
                FROM model_registry 
                ORDER BY created_at DESC
            """)
            
            # System resources
            system_info = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "database": {
                    "connections": db_connections,
                    "size": db_size,
                    "status": "connected"
                },
                "data": {
                    "latest_geri": {
                        "value": float(latest_geri["value"]) if latest_geri else None,
                        "band": latest_geri["band"] if latest_geri else None,
                        "timestamp": latest_geri["ts_utc"].isoformat() if latest_geri else None,
                        "age_hours": (datetime.utcnow() - latest_geri["ts_utc"]).total_seconds() / 3600 if latest_geri else None
                    }
                },
                "activity": {
                    "scenarios_24h": activity["scenarios_24h"],
                    "reviews_7d": activity["reviews_7d"],
                    "api_requests_24h": activity["api_requests_24h"],
                    "active_alerts": activity["active_alerts"],
                    "admin_actions_24h": activity["admin_actions_24h"]
                },
                "ml_models": [
                    {
                        "name": model["model_name"],
                        "version": model["version"],
                        "trained_at": model["created_at"].isoformat(),
                        "status": "active",
                        "metadata": model["metadata"]
                    } for model in models
                ],
                "system": system_info
            }
            
        finally:
            await conn.close()
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status from Render API."""
        if not self.render_api_key:
            return {
                "status": "no_api_key",
                "message": "Render API key not configured",
                "services": []
            }
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.render_api_key}"}
                
                # Get services
                services_resp = await client.get(
                    "https://api.render.com/v1/services",
                    headers=headers,
                    timeout=10
                )
                
                if services_resp.status_code != 200:
                    return {
                        "status": "api_error",
                        "message": f"Render API returned {services_resp.status_code}",
                        "services": []
                    }
                
                services_data = services_resp.json()
                
                # Get deployment info for each service
                service_status = []
                for service in services_data.get("services", [])[:5]:  # Limit to 5 services
                    service_id = service["id"]
                    
                    # Get latest deploy
                    deploys_resp = await client.get(
                        f"https://api.render.com/v1/services/{service_id}/deploys",
                        headers=headers,
                        params={"limit": 1}
                    )
                    
                    if deploys_resp.status_code == 200:
                        deploys = deploys_resp.json().get("deploys", [])
                        latest_deploy = deploys[0] if deploys else None
                    else:
                        latest_deploy = None
                    
                    service_status.append({
                        "name": service["name"],
                        "type": service["type"],
                        "status": service["status"],
                        "created_at": service["createdAt"],
                        "latest_deploy": {
                            "id": latest_deploy["id"] if latest_deploy else None,
                            "status": latest_deploy["status"] if latest_deploy else "unknown",
                            "created_at": latest_deploy["createdAt"] if latest_deploy else None,
                            "finished_at": latest_deploy.get("finishedAt") if latest_deploy else None
                        } if latest_deploy else None
                    })
                
                return {
                    "status": "success",
                    "services": service_status,
                    "total_services": len(services_data.get("services", []))
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch deployment status: {str(e)}",
                "services": []
            }
    
    async def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent admin audit log entries."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            logs = await conn.fetch("""
                SELECT actor, action, payload, occurred_at
                FROM admin_audit_log
                ORDER BY occurred_at DESC
                LIMIT $1
            """, limit)
            
            return [
                {
                    "actor": log["actor"],
                    "action": log["action"],
                    "payload": log["payload"],
                    "occurred_at": log["occurred_at"].isoformat()
                } for log in logs
            ]
        finally:
            await conn.close()
    
    async def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality and freshness metrics."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            # Check data freshness per series
            freshness = await conn.fetch("""
                SELECT 
                    series_id,
                    COUNT(*) as total_observations,
                    MAX(observed_at) as latest_observation,
                    MAX(fetched_at) as latest_fetch,
                    EXTRACT(EPOCH FROM (NOW() - MAX(observed_at))) / 3600 as hours_stale
                FROM raw_observations
                GROUP BY series_id
                ORDER BY latest_observation DESC
            """)
            
            # Check cache performance
            cache_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_keys,
                    COUNT(*) FILTER (WHERE status = 'active') as active_keys,
                    COUNT(*) FILTER (WHERE soft_expiry < NOW()) as soft_expired,
                    COUNT(*) FILTER (WHERE hard_expiry < NOW()) as hard_expired,
                    AVG(refresh_attempts) as avg_refresh_attempts
                FROM cache_metadata
            """)
            
            # Check GERI computation gaps
            geri_gaps = await conn.fetchval("""
                SELECT COUNT(*)
                FROM generate_series(
                    (SELECT MAX(ts_utc) - INTERVAL '7 days' FROM computed_indices),
                    (SELECT MAX(ts_utc) FROM computed_indices),
                    INTERVAL '1 hour'
                ) AS expected_time
                LEFT JOIN computed_indices ci ON ci.ts_utc = expected_time
                WHERE ci.ts_utc IS NULL
            """)
            
            return {
                "data_freshness": [
                    {
                        "series_id": row["series_id"],
                        "total_observations": row["total_observations"],
                        "latest_observation": row["latest_observation"].isoformat() if row["latest_observation"] else None,
                        "hours_stale": float(row["hours_stale"]) if row["hours_stale"] else None,
                        "status": "stale" if row["hours_stale"] and row["hours_stale"] > 48 else "fresh"
                    } for row in freshness
                ],
                "cache_performance": {
                    "total_keys": cache_stats["total_keys"],
                    "active_keys": cache_stats["active_keys"],
                    "soft_expired": cache_stats["soft_expired"],
                    "hard_expired": cache_stats["hard_expired"],
                    "avg_refresh_attempts": float(cache_stats["avg_refresh_attempts"]) if cache_stats["avg_refresh_attempts"] else 0
                },
                "geri_computation": {
                    "missing_hourly_computations_last_7d": geri_gaps
                }
            }
            
        finally:
            await conn.close()
    
    async def log_admin_action(self, actor: str, action: str, payload: Dict[str, Any] = None):
        """Log an admin action to the audit trail."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            await conn.execute("""
                INSERT INTO admin_audit_log (actor, action, payload, occurred_at)
                VALUES ($1, $2, $3, NOW())
            """, actor, action, payload)
        finally:
            await conn.close()


def get_admin_service() -> AdminService:
    """Dependency injection for admin service."""
    fallback_dsn = "postgresql://ris_user:ris_password@ris-postgres:5432/ris_production"
    postgres_dsn = os.environ.get("RIS_POSTGRES_DSN") or fallback_dsn
    render_api_key = os.environ.get("RENDER_API_KEY")

    if postgres_dsn == fallback_dsn:
        logger.warning("AdminService using fallback DSN; set RIS_POSTGRES_DSN in production.")

    return AdminService(postgres_dsn, render_api_key)
