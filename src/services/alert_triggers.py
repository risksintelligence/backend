"""Alert trigger conditions and evaluation logic for production alerts."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    GERI_THRESHOLD = "geri_threshold"
    GERI_CHANGE = "geri_change"
    DATA_STALE = "data_stale"
    SYSTEM_HEALTH = "system_health"
    ML_ANOMALY = "ml_anomaly"
    SCENARIO_EXTREME = "scenario_extreme"


@dataclass
class AlertCondition:
    """Defines when an alert should be triggered."""
    alert_type: AlertType
    severity: AlertSeverity
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte", "eq", "ne"
    time_window_minutes: Optional[int] = None
    cooldown_minutes: int = 60  # Minimum time between similar alerts
    
    def evaluate(self, current_value: float, context: Dict[str, Any] = None) -> bool:
        """Evaluate if the condition is met."""
        context = context or {}
        
        # Check comparison
        if self.comparison == "gt" and current_value <= self.threshold_value:
            return False
        elif self.comparison == "lt" and current_value >= self.threshold_value:
            return False
        elif self.comparison == "gte" and current_value < self.threshold_value:
            return False
        elif self.comparison == "lte" and current_value > self.threshold_value:
            return False
        elif self.comparison == "eq" and current_value != self.threshold_value:
            return False
        elif self.comparison == "ne" and current_value == self.threshold_value:
            return False
            
        return True


@dataclass
class AlertEvent:
    """Represents a triggered alert event."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    value: float
    threshold: float
    timestamp: str
    context: Dict[str, Any]
    
    def to_email_content(self) -> str:
        """Format alert for email delivery."""
        return f"""
RiskSX Intelligence System Alert

Alert Type: {self.alert_type.value.upper()}
Severity: {self.severity.value.upper()}
Time: {self.timestamp}

{self.title}

{self.message}

Current Value: {self.value}
Threshold: {self.threshold}

Additional Context:
{self._format_context()}

---
RiskSX Intelligence System
https://frontend-1-wvu7.onrender.com
        """.strip()
    
    def to_webhook_payload(self) -> Dict[str, Any]:
        """Format alert for webhook delivery."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "context": self.context,
            "source": "RiskSX Intelligence System",
            "dashboard_url": "https://frontend-1-wvu7.onrender.com/admin"
        }
    
    def _format_context(self) -> str:
        """Format context dictionary for human reading."""
        if not self.context:
            return "None"
        
        lines = []
        for key, value in self.context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class AlertTriggerService:
    """Evaluates conditions and triggers alerts for various system states."""
    
    def __init__(self):
        self._last_alert_times: Dict[str, datetime] = {}
        self._default_conditions = self._setup_default_conditions()
    
    def _setup_default_conditions(self) -> List[AlertCondition]:
        """Setup default alert conditions for production monitoring."""
        return [
            # GERI threshold alerts
            AlertCondition(
                alert_type=AlertType.GERI_THRESHOLD,
                severity=AlertSeverity.HIGH,
                threshold_value=20.0,
                comparison="lt",
                cooldown_minutes=30
            ),
            AlertCondition(
                alert_type=AlertType.GERI_THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                threshold_value=10.0,
                comparison="lt",
                cooldown_minutes=15
            ),
            
            # GERI change alerts
            AlertCondition(
                alert_type=AlertType.GERI_CHANGE,
                severity=AlertSeverity.MEDIUM,
                threshold_value=10.0,
                comparison="gt",
                time_window_minutes=60,
                cooldown_minutes=120
            ),
            AlertCondition(
                alert_type=AlertType.GERI_CHANGE,
                severity=AlertSeverity.HIGH,
                threshold_value=20.0,
                comparison="gt",
                time_window_minutes=60,
                cooldown_minutes=60
            ),
            
            # Data freshness alerts
            AlertCondition(
                alert_type=AlertType.DATA_STALE,
                severity=AlertSeverity.MEDIUM,
                threshold_value=6.0,  # hours
                comparison="gt",
                cooldown_minutes=240  # 4 hours
            ),
            AlertCondition(
                alert_type=AlertType.DATA_STALE,
                severity=AlertSeverity.HIGH,
                threshold_value=24.0,  # hours
                comparison="gt",
                cooldown_minutes=180  # 3 hours
            ),
        ]
    
    def evaluate_geri_alerts(self, current_geri: float, previous_geri: Optional[float] = None, 
                           context: Dict[str, Any] = None) -> List[AlertEvent]:
        """Evaluate GERI-related alert conditions."""
        events = []
        context = context or {}
        
        # Check threshold alerts
        for condition in self._default_conditions:
            if condition.alert_type != AlertType.GERI_THRESHOLD:
                continue
                
            if self._should_skip_cooldown(condition):
                continue
                
            if condition.evaluate(current_geri, context):
                event = AlertEvent(
                    alert_type=condition.alert_type,
                    severity=condition.severity,
                    title=f"GERI Below Critical Threshold",
                    message=f"Global Economic Resilience Index has fallen to {current_geri:.2f}, below the {condition.severity.value} threshold of {condition.threshold_value}.",
                    value=current_geri,
                    threshold=condition.threshold_value,
                    timestamp=datetime.utcnow().isoformat(),
                    context=context
                )
                events.append(event)
                self._record_alert_time(condition)
        
        # Check change alerts
        if previous_geri is not None:
            delta = abs(current_geri - previous_geri)
            for condition in self._default_conditions:
                if condition.alert_type != AlertType.GERI_CHANGE:
                    continue
                    
                if self._should_skip_cooldown(condition):
                    continue
                    
                if condition.evaluate(delta, context):
                    direction = "increased" if current_geri > previous_geri else "decreased"
                    event = AlertEvent(
                        alert_type=condition.alert_type,
                        severity=condition.severity,
                        title=f"Significant GERI Change Detected",
                        message=f"GERI has {direction} by {delta:.2f} points from {previous_geri:.2f} to {current_geri:.2f}.",
                        value=delta,
                        threshold=condition.threshold_value,
                        timestamp=datetime.utcnow().isoformat(),
                        context={**context, "previous_geri": previous_geri, "direction": direction}
                    )
                    events.append(event)
                    self._record_alert_time(condition)
        
        return events
    
    def evaluate_data_freshness_alerts(self, series_freshness: List[Dict[str, Any]]) -> List[AlertEvent]:
        """Evaluate data freshness alert conditions."""
        events = []
        
        for series_data in series_freshness:
            hours_stale = series_data.get("hours_stale", 0)
            series_id = series_data.get("series_id", "unknown")
            
            for condition in self._default_conditions:
                if condition.alert_type != AlertType.DATA_STALE:
                    continue
                    
                condition_key = f"{condition.alert_type.value}_{series_id}"
                if self._should_skip_cooldown(condition, condition_key):
                    continue
                    
                if condition.evaluate(hours_stale):
                    event = AlertEvent(
                        alert_type=condition.alert_type,
                        severity=condition.severity,
                        title=f"Stale Data Detected: {series_id}",
                        message=f"Data series {series_id} is {hours_stale:.1f} hours stale, exceeding the {condition.severity.value} threshold.",
                        value=hours_stale,
                        threshold=condition.threshold_value,
                        timestamp=datetime.utcnow().isoformat(),
                        context={"series_id": series_id, "last_update": series_data.get("latest_observation")}
                    )
                    events.append(event)
                    self._record_alert_time(condition, condition_key)
        
        return events
    
    def evaluate_ml_anomaly_alerts(self, anomaly_score: float, contributing_features: List[str], 
                                 context: Dict[str, Any] = None) -> List[AlertEvent]:
        """Evaluate ML anomaly detection alerts."""
        events = []
        context = context or {}
        
        # Anomaly score thresholds (lower scores indicate more anomalous in Isolation Forest)
        if anomaly_score < -0.5:  # Critical anomaly
            severity = AlertSeverity.CRITICAL
            threshold = -0.5
        elif anomaly_score < -0.2:  # High anomaly
            severity = AlertSeverity.HIGH
            threshold = -0.2
        elif anomaly_score < 0:  # Medium anomaly
            severity = AlertSeverity.MEDIUM
            threshold = 0.0
        else:
            return events  # No anomaly detected
        
        condition_key = f"ml_anomaly_{severity.value}"
        if self._should_skip_cooldown_by_key(condition_key, 30):  # 30 min cooldown
            return events
        
        event = AlertEvent(
            alert_type=AlertType.ML_ANOMALY,
            severity=severity,
            title="ML Anomaly Detected",
            message=f"Machine learning anomaly detector flagged unusual market conditions. Anomaly score: {anomaly_score:.3f}. Primary contributing factors: {', '.join(contributing_features[:3])}.",
            value=anomaly_score,
            threshold=threshold,
            timestamp=datetime.utcnow().isoformat(),
            context={**context, "contributing_features": contributing_features}
        )
        events.append(event)
        self._last_alert_times[condition_key] = datetime.utcnow()
        
        return events
    
    def evaluate_scenario_alerts(self, scenario_result: Dict[str, Any]) -> List[AlertEvent]:
        """Evaluate scenario simulation alerts for extreme outcomes."""
        events = []
        
        baseline = scenario_result.get("baseline", 0)
        scenario_value = scenario_result.get("scenario", 0)
        delta = abs(scenario_value - baseline)
        
        # Alert on extreme scenario outcomes
        if delta > 25.0:  # Very large change
            severity = AlertSeverity.HIGH
            threshold = 25.0
        elif delta > 15.0:  # Large change
            severity = AlertSeverity.MEDIUM
            threshold = 15.0
        else:
            return events  # Not extreme enough for alert
        
        condition_key = f"scenario_extreme_{severity.value}"
        if self._should_skip_cooldown_by_key(condition_key, 60):  # 1 hour cooldown
            return events
        
        direction = "increase" if scenario_value > baseline else "decrease"
        event = AlertEvent(
            alert_type=AlertType.SCENARIO_EXTREME,
            severity=severity,
            title="Extreme Scenario Outcome",
            message=f"Scenario simulation shows potential {direction} of {delta:.1f} GERI points from baseline {baseline:.1f} to {scenario_value:.1f}.",
            value=delta,
            threshold=threshold,
            timestamp=datetime.utcnow().isoformat(),
            context={
                "baseline": baseline,
                "scenario_value": scenario_value,
                "shocks": scenario_result.get("shocks", []),
                "direction": direction
            }
        )
        events.append(event)
        self._last_alert_times[condition_key] = datetime.utcnow()
        
        return events
    
    def _should_skip_cooldown(self, condition: AlertCondition, key: Optional[str] = None) -> bool:
        """Check if alert should be skipped due to cooldown period."""
        alert_key = key or f"{condition.alert_type.value}_{condition.severity.value}"
        return self._should_skip_cooldown_by_key(alert_key, condition.cooldown_minutes)
    
    def _should_skip_cooldown_by_key(self, key: str, cooldown_minutes: int) -> bool:
        """Check cooldown by specific key."""
        last_alert = self._last_alert_times.get(key)
        if last_alert is None:
            return False
        
        cooldown_delta = timedelta(minutes=cooldown_minutes)
        return datetime.utcnow() - last_alert < cooldown_delta
    
    def _record_alert_time(self, condition: AlertCondition, key: Optional[str] = None):
        """Record the time an alert was sent."""
        alert_key = key or f"{condition.alert_type.value}_{condition.severity.value}"
        self._last_alert_times[alert_key] = datetime.utcnow()