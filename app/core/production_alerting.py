"""
Production Health Alerting System

Integrates with existing health monitoring to provide comprehensive alerting
for critical system health issues, API failures, and performance degradations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

import sentry_sdk
from app.core.unified_cache import UnifiedCache
from app.services.health_monitor import health_monitor, HealthStatus, ServiceHealth
from app.core.provider_failover import failover_manager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AlertType(Enum):
    SERVICE_DOWN = "service_down"
    API_DEGRADED = "api_degraded"
    HIGH_ERROR_RATE = "high_error_rate"
    RESPONSE_TIME_HIGH = "response_time_high"
    CACHE_FAILURE = "cache_failure"
    DATABASE_ISSUE = "database_issue"
    PROVIDER_FAILURE = "provider_failure"

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    alert_type: AlertType
    service_name: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "alert_type": self.alert_type.value,
            "service_name": self.service_name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() + "Z",
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() + "Z" if self.resolved_at else None
        }

class ProductionAlertingService:
    """Production alerting service for critical health monitoring."""
    
    def __init__(self):
        self.cache = UnifiedCache("production_alerting")
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.thresholds = self._get_alert_thresholds()
        
    def _get_alert_thresholds(self) -> Dict[str, Any]:
        """Define alerting thresholds for different metrics."""
        return {
            "response_time_critical_ms": 10000,  # 10 seconds
            "response_time_high_ms": 5000,       # 5 seconds
            "error_rate_critical": 0.5,          # 50%
            "error_rate_high": 0.2,              # 20%
            "cache_miss_rate_critical": 0.9,     # 90%
            "provider_failure_threshold": 3,     # 3 consecutive failures
            "database_timeout_ms": 5000,         # 5 seconds
            "service_down_duration_minutes": 5   # 5 minutes
        }
    
    async def check_system_health_and_alert(self) -> Dict[str, Any]:
        """
        Main alerting method that checks all system components and generates alerts.
        Should be called periodically by a background task.
        """
        try:
            # Get comprehensive health status
            all_health = await health_monitor.check_all_services()
            system_summary = await health_monitor.get_system_health_summary()
            provider_health = failover_manager.get_provider_health()
            
            alerts_generated = []
            alerts_resolved = []
            
            # Check service health
            for service_name, health in all_health.items():
                new_alerts, resolved_alerts = self._evaluate_service_health(service_name, health)
                alerts_generated.extend(new_alerts)
                alerts_resolved.extend(resolved_alerts)
            
            # Check provider health
            provider_alerts, provider_resolved = self._evaluate_provider_health(provider_health)
            alerts_generated.extend(provider_alerts)
            alerts_resolved.extend(provider_resolved)
            
            # Check system-wide metrics
            system_alerts, system_resolved = self._evaluate_system_metrics(system_summary)
            alerts_generated.extend(system_alerts)
            alerts_resolved.extend(system_resolved)
            
            # Send alerts to monitoring systems
            for alert in alerts_generated:
                await self._send_alert(alert)
                
            for alert in alerts_resolved:
                await self._resolve_alert(alert)
            
            # Cache alert summary
            alert_summary = {
                "total_active_alerts": len(self.active_alerts),
                "critical_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                "high_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH]),
                "alerts_generated": len(alerts_generated),
                "alerts_resolved": len(alerts_resolved),
                "last_check": datetime.utcnow().isoformat() + "Z"
            }
            
            self.cache.set(
                "alert_summary", 
                alert_summary,
                source="production_alerting",
                soft_ttl=300  # 5 minutes
            )
            
            return alert_summary
            
        except Exception as e:
            logger.error(f"Production alerting check failed: {e}")
            sentry_sdk.capture_exception(e)
            return {"error": str(e), "last_check": datetime.utcnow().isoformat() + "Z"}
    
    def _evaluate_service_health(self, service_name: str, health: ServiceHealth) -> Tuple[List[Alert], List[Alert]]:
        """Evaluate individual service health and generate/resolve alerts."""
        new_alerts = []
        resolved_alerts = []
        
        alert_id_base = f"service_{service_name}"
        
        # Check if service is down
        if health.status == HealthStatus.UNHEALTHY:
            alert_id = f"{alert_id_base}_down"
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.SERVICE_DOWN,
                    service_name=service_name,
                    message=f"Service {service_name} is unhealthy: {health.error_message or 'Unknown error'}",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "service_name": service_name,
                        "response_time_ms": health.response_time_ms,
                        "error_message": health.error_message,
                        "last_successful_check": health.last_check.isoformat() + "Z"
                    }
                )
                self.active_alerts[alert_id] = alert
                new_alerts.append(alert)
        else:
            # Resolve service down alert if it exists
            alert_id = f"{alert_id_base}_down"
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
        
        # Check response times
        if health.response_time_ms:
            if health.response_time_ms > self.thresholds["response_time_critical_ms"]:
                alert_id = f"{alert_id_base}_slow_critical"
                if alert_id not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        severity=AlertSeverity.CRITICAL,
                        alert_type=AlertType.RESPONSE_TIME_HIGH,
                        service_name=service_name,
                        message=f"Service {service_name} response time critical: {health.response_time_ms:.0f}ms",
                        timestamp=datetime.utcnow(),
                        metadata={
                            "response_time_ms": health.response_time_ms,
                            "threshold_ms": self.thresholds["response_time_critical_ms"]
                        }
                    )
                    self.active_alerts[alert_id] = alert
                    new_alerts.append(alert)
            elif health.response_time_ms > self.thresholds["response_time_high_ms"]:
                alert_id = f"{alert_id_base}_slow_high"
                if alert_id not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        severity=AlertSeverity.HIGH,
                        alert_type=AlertType.RESPONSE_TIME_HIGH,
                        service_name=service_name,
                        message=f"Service {service_name} response time elevated: {health.response_time_ms:.0f}ms",
                        timestamp=datetime.utcnow(),
                        metadata={
                            "response_time_ms": health.response_time_ms,
                            "threshold_ms": self.thresholds["response_time_high_ms"]
                        }
                    )
                    self.active_alerts[alert_id] = alert
                    new_alerts.append(alert)
            else:
                # Resolve response time alerts if they exist
                for alert_suffix in ["_slow_critical", "_slow_high"]:
                    alert_id = f"{alert_id_base}{alert_suffix}"
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        alert.resolved_at = datetime.utcnow()
                        resolved_alerts.append(alert)
                        del self.active_alerts[alert_id]
        
        return new_alerts, resolved_alerts
    
    def _evaluate_provider_health(self, provider_health: Dict[str, Any]) -> Tuple[List[Alert], List[Alert]]:
        """Evaluate provider health and generate alerts for failing providers."""
        new_alerts = []
        resolved_alerts = []
        
        for provider_name, stats in provider_health.items():
            failure_count = stats.get("failure_count", 0)
            reliability_score = stats.get("reliability_score", 1.0)
            should_skip = stats.get("should_skip", False)
            
            alert_id = f"provider_{provider_name}_failure"
            
            if should_skip or failure_count >= self.thresholds["provider_failure_threshold"]:
                if alert_id not in self.active_alerts:
                    severity = AlertSeverity.CRITICAL if reliability_score < 0.5 else AlertSeverity.HIGH
                    alert = Alert(
                        id=alert_id,
                        severity=severity,
                        alert_type=AlertType.PROVIDER_FAILURE,
                        service_name=f"provider_{provider_name}",
                        message=f"Provider {provider_name} failing: {failure_count} failures, reliability {reliability_score:.2f}",
                        timestamp=datetime.utcnow(),
                        metadata={
                            "provider_name": provider_name,
                            "failure_count": failure_count,
                            "reliability_score": reliability_score,
                            "should_skip": should_skip
                        }
                    )
                    self.active_alerts[alert_id] = alert
                    new_alerts.append(alert)
            else:
                # Resolve provider failure alert if it exists
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    resolved_alerts.append(alert)
                    del self.active_alerts[alert_id]
        
        return new_alerts, resolved_alerts
    
    def _evaluate_system_metrics(self, system_summary: Dict[str, Any]) -> Tuple[List[Alert], List[Alert]]:
        """Evaluate system-wide metrics and generate alerts."""
        new_alerts = []
        resolved_alerts = []
        
        # Check overall system health
        overall_health = system_summary.get("overall_health", "unknown")
        alert_id = "system_overall_health"
        
        if overall_health in ["unhealthy", "critical"]:
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.SERVICE_DOWN,
                    service_name="system_overall",
                    message=f"Overall system health is {overall_health}",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "overall_health": overall_health,
                        "healthy_services": system_summary.get("healthy_services", 0),
                        "total_services": system_summary.get("total_services", 0)
                    }
                )
                self.active_alerts[alert_id] = alert
                new_alerts.append(alert)
        elif overall_health == "degraded":
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    severity=AlertSeverity.HIGH,
                    alert_type=AlertType.API_DEGRADED,
                    service_name="system_overall",
                    message="System health is degraded",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "overall_health": overall_health,
                        "healthy_services": system_summary.get("healthy_services", 0),
                        "total_services": system_summary.get("total_services", 0)
                    }
                )
                self.active_alerts[alert_id] = alert
                new_alerts.append(alert)
        else:
            # Resolve system health alert if it exists
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
        
        return new_alerts, resolved_alerts
    
    async def _send_alert(self, alert: Alert) -> None:
        """Send alert to monitoring systems (Sentry, logs, etc.)."""
        try:
            # Log the alert
            log_level = logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.ERROR
            logger.log(log_level, f"PRODUCTION ALERT [{alert.severity.value.upper()}]: {alert.message}")
            
            # Send to Sentry
            sentry_sdk.set_context("production_alert", {
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "alert_type": alert.alert_type.value,
                "service_name": alert.service_name,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat() + "Z",
                "metadata": alert.metadata
            })
            
            if alert.severity == AlertSeverity.CRITICAL:
                # Capture as exception for critical alerts
                sentry_sdk.capture_message(
                    f"CRITICAL PRODUCTION ALERT: {alert.message}",
                    level="error"
                )
            else:
                # Add as breadcrumb for non-critical alerts
                sentry_sdk.add_breadcrumb(
                    message=f"Production Alert: {alert.message}",
                    category="alert",
                    level="warning" if alert.severity == AlertSeverity.HIGH else "info",
                    data={
                        "alert_id": alert.id,
                        "service_name": alert.service_name,
                        "alert_type": alert.alert_type.value
                    }
                )
            
            # Store in cache for API access
            alert_key = f"active_alert_{alert.id}"
            self.cache.set(
                alert_key,
                alert.to_dict(),
                source="production_alerting",
                soft_ttl=3600  # 1 hour
            )
            
        except Exception as e:
            logger.error(f"Failed to send alert {alert.id}: {e}")
    
    async def _resolve_alert(self, alert: Alert) -> None:
        """Mark alert as resolved and notify monitoring systems."""
        try:
            logger.info(f"PRODUCTION ALERT RESOLVED: {alert.message}")
            
            # Add to Sentry
            sentry_sdk.add_breadcrumb(
                message=f"Production Alert Resolved: {alert.message}",
                category="alert_resolved",
                level="info",
                data={
                    "alert_id": alert.id,
                    "service_name": alert.service_name,
                    "resolved_at": alert.resolved_at.isoformat() + "Z" if alert.resolved_at else None
                }
            )
            
            # Remove from active alerts cache
            alert_key = f"active_alert_{alert.id}"
            self.cache.delete(alert_key)
            
            # Store in resolved alerts history
            self.alert_history.append(alert)
            
            # Keep only last 100 resolved alerts in memory
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert.id}: {e}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return [alert.to_dict() for alert in self.alert_history[-limit:]]
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        cached_summary, _ = self.cache.get("alert_summary")
        if cached_summary:
            return cached_summary
        
        # If no cached summary, run a quick check
        return await self.check_system_health_and_alert()

# Singleton instance
production_alerting = ProductionAlertingService()