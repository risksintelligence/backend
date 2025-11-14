"""Background job for continuous alert monitoring and delivery."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.services.admin_service import get_admin_service
from src.services.alert_triggers import AlertTriggerService, AlertEvent
from src.services.scenario_service import get_alert_service
from src.ml.inference.service import MLInferenceService

logger = logging.getLogger(__name__)


class AlertMonitoringJob:
    """Continuous monitoring job that evaluates alert conditions and sends notifications."""
    
    def __init__(self):
        self.admin_service = get_admin_service()
        self.alert_service = get_alert_service()
        self.trigger_service = AlertTriggerService()
        self.ml_service = MLInferenceService()
        self._previous_geri = None
        self._last_check = datetime.utcnow()
    
    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run one complete monitoring cycle checking all alert conditions."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "alerts_triggered": 0,
            "alerts_sent": 0,
            "checks_performed": [],
            "errors": []
        }
        
        try:
            # Get current system health
            system_health = await self.admin_service.get_system_health()
            data_quality = await self.admin_service.get_data_quality_metrics()
            
            # Check GERI alerts
            await self._check_geri_alerts(system_health, results)
            
            # Check data freshness alerts
            await self._check_data_freshness_alerts(data_quality, results)
            
            # Check ML anomaly alerts
            await self._check_ml_anomaly_alerts(results)
            
            # Check system health alerts
            await self._check_system_health_alerts(system_health, results)
            
            logger.info(f"Alert monitoring cycle completed: {results['alerts_triggered']} alerts triggered, {results['alerts_sent']} sent")
            
        except Exception as e:
            logger.error(f"Error in alert monitoring cycle: {e}")
            results["errors"].append(str(e))
        
        self._last_check = datetime.utcnow()
        return results
    
    async def _check_geri_alerts(self, system_health: Dict[str, Any], results: Dict[str, Any]):
        """Check GERI threshold and change alerts."""
        try:
            latest_geri_data = system_health.get("data", {}).get("latest_geri", {})
            current_geri = latest_geri_data.get("value")
            
            if current_geri is None:
                results["checks_performed"].append("geri_alerts_skipped_no_data")
                return
            
            context = {
                "band": latest_geri_data.get("band"),
                "age_hours": latest_geri_data.get("age_hours"),
                "timestamp": latest_geri_data.get("timestamp")
            }
            
            # Evaluate GERI alerts
            alert_events = self.trigger_service.evaluate_geri_alerts(
                current_geri=current_geri,
                previous_geri=self._previous_geri,
                context=context
            )
            
            # Send any triggered alerts
            for event in alert_events:
                await self._send_alert(event, results)
            
            self._previous_geri = current_geri
            results["checks_performed"].append(f"geri_alerts_checked_value_{current_geri:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking GERI alerts: {e}")
            results["errors"].append(f"geri_alerts: {e}")
    
    async def _check_data_freshness_alerts(self, data_quality: Dict[str, Any], results: Dict[str, Any]):
        """Check data freshness alerts for stale series."""
        try:
            freshness_data = data_quality.get("data_freshness", [])
            
            if not freshness_data:
                results["checks_performed"].append("data_freshness_skipped_no_data")
                return
            
            # Evaluate data freshness alerts
            alert_events = self.trigger_service.evaluate_data_freshness_alerts(freshness_data)
            
            # Send any triggered alerts
            for event in alert_events:
                await self._send_alert(event, results)
            
            results["checks_performed"].append(f"data_freshness_checked_{len(freshness_data)}_series")
            
        except Exception as e:
            logger.error(f"Error checking data freshness alerts: {e}")
            results["errors"].append(f"data_freshness: {e}")
    
    async def _check_ml_anomaly_alerts(self, results: Dict[str, Any]):
        """Check ML anomaly detection alerts."""
        try:
            # Get recent anomaly detections
            anomaly_results = await self.ml_service.detect_recent_anomalies(24)  # Last 24 hours
            
            if not anomaly_results:
                results["checks_performed"].append("ml_anomaly_skipped_no_data")
                return
            
            # Check most recent anomaly
            latest_anomaly = anomaly_results[0]  # Most recent
            anomaly_score = latest_anomaly.get("anomaly_score", 1.0)
            contributing_features = [
                item[0] for item in latest_anomaly.get("contributing_features", [])
            ]
            
            context = {
                "model_version": latest_anomaly.get("model_version"),
                "detection_time": latest_anomaly.get("analyzed_at"),
                "severity": latest_anomaly.get("severity")
            }
            
            # Evaluate ML anomaly alerts
            alert_events = self.trigger_service.evaluate_ml_anomaly_alerts(
                anomaly_score=anomaly_score,
                contributing_features=contributing_features,
                context=context
            )
            
            # Send any triggered alerts
            for event in alert_events:
                await self._send_alert(event, results)
            
            results["checks_performed"].append(f"ml_anomaly_checked_score_{anomaly_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error checking ML anomaly alerts: {e}")
            results["errors"].append(f"ml_anomaly: {e}")
    
    async def _check_system_health_alerts(self, system_health: Dict[str, Any], results: Dict[str, Any]):
        """Check system health alerts for performance issues."""
        try:
            system_info = system_health.get("system", {})
            
            if not system_info:
                results["checks_performed"].append("system_health_skipped_no_data")
                return
            
            cpu_percent = system_info.get("cpu_percent", 0)
            memory_percent = system_info.get("memory_percent", 0)
            
            # Check CPU usage
            if cpu_percent > 90:
                event = AlertEvent(
                    alert_type="system_health",
                    severity="high",
                    title="High CPU Usage Detected",
                    message=f"System CPU usage is at {cpu_percent:.1f}%, exceeding the critical threshold.",
                    value=cpu_percent,
                    threshold=90.0,
                    timestamp=datetime.utcnow().isoformat(),
                    context={"metric": "cpu_percent", "system_info": system_info}
                )
                await self._send_alert(event, results)
            
            # Check memory usage
            if memory_percent > 85:
                event = AlertEvent(
                    alert_type="system_health",
                    severity="high",
                    title="High Memory Usage Detected",
                    message=f"System memory usage is at {memory_percent:.1f}%, exceeding the warning threshold.",
                    value=memory_percent,
                    threshold=85.0,
                    timestamp=datetime.utcnow().isoformat(),
                    context={"metric": "memory_percent", "system_info": system_info}
                )
                await self._send_alert(event, results)
            
            results["checks_performed"].append(f"system_health_checked_cpu_{cpu_percent:.1f}_mem_{memory_percent:.1f}")
            
        except Exception as e:
            logger.error(f"Error checking system health alerts: {e}")
            results["errors"].append(f"system_health: {e}")
    
    async def _send_alert(self, event: AlertEvent, results: Dict[str, Any]):
        """Send an alert event through all configured delivery channels."""
        try:
            # Format alert payload
            alert_payload = {
                "alert_type": event.alert_type,
                "severity": event.severity,
                "title": event.title,
                "message": event.message,
                "value": event.value,
                "threshold": event.threshold,
                "timestamp": event.timestamp,
                "context": event.context,
                "dashboard_url": "https://frontend-1-wvu7.onrender.com/admin"
            }
            
            # Deliver through alert service (handles all subscription channels)
            deliveries = await self.alert_service.deliver_alerts(alert_payload)
            
            results["alerts_triggered"] += 1
            results["alerts_sent"] += len(deliveries)
            
            logger.info(f"Alert sent: {event.title} ({event.severity}) to {len(deliveries)} channels")
            
            # Log alert to admin audit trail
            await self.admin_service.log_admin_action(
                actor="alert_system",
                action=f"alert_triggered_{event.alert_type}",
                payload={
                    "severity": event.severity,
                    "title": event.title,
                    "value": event.value,
                    "threshold": event.threshold,
                    "deliveries_sent": len(deliveries)
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            results["errors"].append(f"send_alert: {e}")


async def run_alert_monitoring():
    """Main entry point for alert monitoring job."""
    logger.info("Starting alert monitoring job...")
    
    job = AlertMonitoringJob()
    
    # Run monitoring cycle
    results = await job.run_monitoring_cycle()
    
    logger.info(f"Alert monitoring completed: {results}")
    return results


async def run_continuous_monitoring(check_interval_minutes: int = 15):
    """Run continuous alert monitoring with specified interval."""
    logger.info(f"Starting continuous alert monitoring (interval: {check_interval_minutes} minutes)...")
    
    job = AlertMonitoringJob()
    
    while True:
        try:
            results = await job.run_monitoring_cycle()
            logger.info(f"Monitoring cycle results: {results['alerts_triggered']} alerts, {len(results['errors'])} errors")
            
            # Wait for next cycle
            await asyncio.sleep(check_interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("Alert monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        asyncio.run(run_continuous_monitoring())
    else:
        asyncio.run(run_alert_monitoring())
