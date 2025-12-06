"""
API Health Monitoring Service

Provides comprehensive health monitoring for all external APIs, database connections,
and internal services to ensure production readiness and quick issue detection.
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.config import get_settings
from app.core.unified_cache import UnifiedCache
from app.db import get_db

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    response_time_ms: Optional[float]
    last_check: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class HealthMonitor:
    """Comprehensive health monitoring for all system components."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = UnifiedCache("health_monitor")
        self.client = httpx.AsyncClient(timeout=15.0)
        
        # Service endpoints to monitor
        self.external_apis = {
            "gdelt": {
                "url": "http://127.0.0.1:8000/api/v1/geopolitical/disruptions?days=1",
                "timeout": 8,
                "display_name": "GDELT Free Geopolitical Intelligence",
                "service_type": "free_intelligence"
            },
            "maritime_intelligence": {
                "url": "http://127.0.0.1:8000/api/v1/maritime/health", 
                "timeout": 8,
                "display_name": "Free Maritime Intelligence",
                "service_type": "free_intelligence"
            },
            "wto_statistics": {
                "url": "https://stats.wto.org/api/v1/datasets",
                "timeout": 15,
                "display_name": "WTO Trade Statistics",
                "service_type": "external_api"
            },
            "world_bank": {
                "url": "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD",
                "params": {"format": "json", "per_page": 1},
                "timeout": 10,
                "display_name": "World Bank WITS Data",
                "service_type": "external_api"
            },
            "sec_edgar": {
                "url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                "headers": {"User-Agent": "RiskX Observatory Health Monitor"},
                "timeout": 10,
                "display_name": "SEC EDGAR Financial Data",
                "service_type": "external_api"
            }
        }
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all system components."""
        results = {}
        
        # Check external APIs
        api_checks = await self._check_external_apis()
        results.update(api_checks)
        
        # Check internal services
        internal_checks = await self._check_internal_services()
        # Ensure Redis required for health caching in production
        if self.settings.is_production and not self.settings.redis_url:
            results = {"redis_cache": ServiceHealth(name="Redis Cache", status=HealthStatus.UNHEALTHY, response_time_ms=None, last_check=datetime.utcnow(), error_message="Redis URL missing", metadata={})}
            internal_checks.update(results)
        results.update(internal_checks)
        
        # Cache results for quick access
        health_summary = {
            name: asdict(health) for name, health in results.items()
        }
        self.cache.set(
            key="system_health",
            value=health_summary,
            source="health_monitor",
            derivation_flag="computed",
            soft_ttl=60,  # 1 minute
            hard_ttl=300  # 5 minutes
        )
        
        return results
    
    async def _check_external_apis(self) -> Dict[str, ServiceHealth]:
        """Check health of external API endpoints."""
        results = {}
        
        for api_name, config in self.external_apis.items():
            try:
                start_time = datetime.utcnow()
                
                # Prepare request
                url = config["url"]
                params = config.get("params", {})
                headers = config.get("headers", {})
                timeout = config.get("timeout", 10)
                
                # Add API keys where available
                if api_name == "gdelt":
                    # GDELT is free - no authentication required
                    pass
                elif api_name == "maritime_intelligence":  # Free service - no API key needed
                    pass  # No authentication required for free maritime intelligence sources
                
                # Make request
                response = await self.client.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=timeout
                )
                
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Determine health status
                if response.status_code == 200:
                    status = HealthStatus.HEALTHY
                    error_message = None
                elif response.status_code in [429, 503]:
                    status = HealthStatus.DEGRADED
                    error_message = f"Rate limited or service unavailable: {response.status_code}"
                else:
                    status = HealthStatus.UNHEALTHY
                    error_message = f"HTTP {response.status_code}: {response.text[:100]}"
                
                display_name = config.get("display_name", f"{api_name.upper()} API")
                service_type = config.get("service_type", "external_api")
                
                results[f"api_{api_name}"] = ServiceHealth(
                    name=display_name,
                    status=status,
                    response_time_ms=response_time,
                    last_check=datetime.utcnow(),
                    error_message=error_message,
                    metadata={
                        "endpoint": url,
                        "status_code": response.status_code,
                        "service_type": service_type,
                        "api_identifier": api_name,
                        "has_api_key": self._has_valid_api_key(api_name),
                        "is_free_service": service_type == "free_intelligence"
                    }
                )
                
            except asyncio.TimeoutError:
                display_name = config.get("display_name", f"{api_name.upper()} API")
                service_type = config.get("service_type", "external_api")
                
                results[f"api_{api_name}"] = ServiceHealth(
                    name=display_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=None,
                    last_check=datetime.utcnow(),
                    error_message=f"Request timeout after {timeout}s",
                    metadata={
                        "endpoint": config["url"],
                        "service_type": service_type,
                        "api_identifier": api_name,
                        "is_free_service": service_type == "free_intelligence"
                    }
                )
            except Exception as e:
                display_name = config.get("display_name", f"{api_name.upper()} API")
                service_type = config.get("service_type", "external_api")
                
                results[f"api_{api_name}"] = ServiceHealth(
                    name=display_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=None,
                    last_check=datetime.utcnow(),
                    error_message=str(e),
                    metadata={
                        "endpoint": config["url"],
                        "service_type": service_type,
                        "api_identifier": api_name,
                        "is_free_service": service_type == "free_intelligence"
                    }
                )
        
        return results
    
    async def _check_internal_services(self) -> Dict[str, ServiceHealth]:
        """Check health of internal services and components."""
        results = {}
        
        # Check database connection
        try:
            start_time = datetime.utcnow()
            from sqlalchemy import text
            db = next(get_db())
            db.execute(text("SELECT 1"))
            db.close()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Determine database type from connection URL
            db_type = "PostgreSQL" if self.settings.database_url.startswith("postgresql") else "Database"
            connection_type = "postgresql" if self.settings.database_url.startswith("postgresql") else "sqlite"
            
            results["database"] = ServiceHealth(
                name=f"Database ({db_type})",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                metadata={"connection_type": connection_type, "is_production_ready": db_type == "PostgreSQL"}
            )
        except Exception as e:
            # Determine database type for error reporting
            db_type = "PostgreSQL" if self.settings.database_url.startswith("postgresql") else "Database"
            
            results["database"] = ServiceHealth(
                name=f"Database ({db_type})",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        # Check Redis cache
        try:
            start_time = datetime.utcnow()
            test_key = f"health_check_{datetime.utcnow().timestamp()}"
            test_value = {"test": True}
            
            self.cache.set(test_key, test_value, "health_monitor")
            cached_data, _ = self.cache.get(test_key)
            # Note: invalidate method may not be available in all cache implementations
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if cached_data and cached_data.get("test"):
                status = HealthStatus.HEALTHY
                error_message = None
            else:
                status = HealthStatus.DEGRADED
                error_message = "Cache read/write test failed"
            
            results["redis_cache"] = ServiceHealth(
                name="Redis Cache",
                status=status,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                error_message=error_message,
                metadata={"redis_url": self.settings.redis_url is not None}
            )
        except Exception as e:
            results["redis_cache"] = ServiceHealth(
                name="Redis Cache",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        # Check ML models availability
        try:
            from app.services.ml_intelligence_service import MLIntelligenceService
            ml_service = MLIntelligenceService()
            start_time = datetime.utcnow()
            
            # Check if models are loaded
            model_status = await ml_service.get_model_status()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            loaded_models = sum(1 for status in model_status.values() if status.get("loaded", False))
            total_models = len(model_status)
            
            if loaded_models == total_models:
                status = HealthStatus.HEALTHY
                error_message = None
            elif loaded_models > 0:
                status = HealthStatus.DEGRADED
                error_message = f"Only {loaded_models}/{total_models} models loaded"
            else:
                status = HealthStatus.UNHEALTHY
                error_message = "No ML models loaded"
            
            results["ml_models"] = ServiceHealth(
                name="ML Models",
                status=status,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                error_message=error_message,
                metadata={
                    "loaded_models": loaded_models,
                    "total_models": total_models,
                    "model_details": model_status
                }
            )
        except Exception as e:
            results["ml_models"] = ServiceHealth(
                name="ML Models",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        # Check background refresh service
        try:
            from app.services.background_refresh import refresh_service
            
            results["background_refresh"] = ServiceHealth(
                name="Background Refresh Service",
                status=HealthStatus.HEALTHY if refresh_service.running else HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.utcnow(),
                error_message=None if refresh_service.running else "Service not running",
                metadata={
                    "running": refresh_service.running,
                    "queue_length": len(refresh_service.refresh_queue)
                }
            )
        except Exception as e:
            results["background_refresh"] = ServiceHealth(
                name="Background Refresh Service",
                status=HealthStatus.UNKNOWN,
                response_time_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        return results
    
    def _has_valid_api_key(self, api_name: str) -> bool:
        """Check if a valid API key is configured for the service."""
        if api_name == "gdelt":
            return True  # GDELT is free - always available
        elif api_name == "maritime_intelligence":
            return True  # Free maritime intelligence sources - always available
        return False
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive system health summary."""
        health_checks = await self.check_all_services()
        
        # Calculate overall health metrics
        total_services = len(health_checks)
        healthy_count = sum(1 for h in health_checks.values() if h.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for h in health_checks.values() if h.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for h in health_checks.values() if h.status == HealthStatus.UNHEALTHY)
        
        # Determine overall system status
        if unhealthy_count == 0 and degraded_count == 0:
            overall_status = HealthStatus.HEALTHY
        elif unhealthy_count == 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        # Calculate average response time for responsive services
        responsive_times = [h.response_time_ms for h in health_checks.values() 
                          if h.response_time_ms is not None]
        avg_response_time = sum(responsive_times) / len(responsive_times) if responsive_times else None
        
        # Identify critical issues
        critical_issues = [
            h.name for h in health_checks.values() 
            if h.status == HealthStatus.UNHEALTHY
        ]
        
        return {
            "overall_status": overall_status.value,
            "summary": {
                "total_services": total_services,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "health_percentage": round((healthy_count / total_services) * 100, 1)
            },
            "performance": {
                "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
                "services_responding": len(responsive_times)
            },
            "critical_issues": critical_issues,
            "services": {name: asdict(health) for name, health in health_checks.items()},
            "recommendations": self._generate_recommendations(health_checks),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(self, health_checks: Dict[str, ServiceHealth]) -> List[str]:
        """Generate actionable recommendations based on health status."""
        recommendations = []
        
        # Check for API key issues
        api_key_issues = [
            h.name for h in health_checks.values()
            if h.name.startswith("External API") and 
            h.metadata and not h.metadata.get("has_api_key", True)
        ]
        if api_key_issues:
            recommendations.append(f"Configure API keys for: {', '.join(api_key_issues)}")
        
        # Check for slow response times
        slow_services = [
            h.name for h in health_checks.values()
            if h.response_time_ms and h.response_time_ms > 5000
        ]
        if slow_services:
            recommendations.append(f"Investigate slow response times for: {', '.join(slow_services)}")
        
        # Check for unhealthy services
        unhealthy_services = [
            h.name for h in health_checks.values()
            if h.status == HealthStatus.UNHEALTHY
        ]
        if unhealthy_services:
            recommendations.append(f"Fix unhealthy services: {', '.join(unhealthy_services)}")
        
        # Check Redis cache
        redis_health = next((h for h in health_checks.values() if h.name == "Redis Cache"), None)
        if redis_health and redis_health.status != HealthStatus.HEALTHY:
            recommendations.append("Check Redis configuration and connectivity")
        
        # Check ML models
        ml_health = next((h for h in health_checks.values() if h.name == "ML Models"), None)
        if ml_health and ml_health.status != HealthStatus.HEALTHY:
            recommendations.append("Verify ML model training and loading process")
        
        if not recommendations:
            recommendations.append("All systems operating normally")
        
        return recommendations
    
    async def cleanup(self):
        """Clean up resources."""
        await self.client.aclose()

# Global instance
health_monitor = HealthMonitor()
