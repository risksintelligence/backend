"""Alert threshold processing service for Scenario Studio."""
from __future__ import annotations

import asyncpg
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from backend.src.services.auth_service import User
from backend.src.services.alerts_delivery import AlertDeliveryService

@dataclass
class AlertThreshold:
    id: int
    user_id: int
    name: str
    geri_threshold: Optional[float]
    delta_threshold: Optional[float]
    conditions: Dict[str, Any]
    notification_channels: List[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class AlertEvent:
    id: int
    threshold_id: int
    user_id: int
    trigger_type: str  # 'geri_threshold', 'delta_threshold', 'custom'
    trigger_value: float
    scenario_data: Dict[str, Any]
    created_at: datetime

class AlertThresholdService:
    """Production alert threshold service with real processing."""
    
    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self.delivery_service = AlertDeliveryService()
        
    async def create_alert_threshold(
        self,
        user_id: int,
        name: str,
        geri_threshold: Optional[float] = None,
        delta_threshold: Optional[float] = None,
        conditions: Dict[str, Any] = None,
        notification_channels: List[str] = None
    ) -> AlertThreshold:
        """Create a new alert threshold."""
        conditions = conditions or {}
        notification_channels = notification_channels or []
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                INSERT INTO alert_thresholds (user_id, name, geri_threshold, delta_threshold, conditions, notification_channels)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, user_id, name, geri_threshold, delta_threshold, conditions, notification_channels, is_active, created_at, updated_at
            """, user_id, name, geri_threshold, delta_threshold, conditions, notification_channels)
            
            return AlertThreshold(
                id=result["id"],
                user_id=result["user_id"],
                name=result["name"],
                geri_threshold=result["geri_threshold"],
                delta_threshold=result["delta_threshold"],
                conditions=result["conditions"],
                notification_channels=result["notification_channels"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                updated_at=result["updated_at"]
            )
        finally:
            await conn.close()
    
    async def update_alert_threshold(
        self,
        threshold_id: int,
        user_id: int,
        **kwargs
    ) -> Optional[AlertThreshold]:
        """Update an existing alert threshold."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            # Build dynamic update query
            updates = []
            params = [threshold_id, user_id]
            param_counter = 3
            
            for field, value in kwargs.items():
                if field in ['name', 'geri_threshold', 'delta_threshold', 'conditions', 'notification_channels', 'is_active']:
                    updates.append(f"{field} = ${param_counter}")
                    params.append(value)
                    param_counter += 1
            
            if not updates:
                return None
            
            updates.append("updated_at = NOW()")
            
            query = f"""
                UPDATE alert_thresholds 
                SET {', '.join(updates)}
                WHERE id = $1 AND user_id = $2
                RETURNING id, user_id, name, geri_threshold, delta_threshold, conditions, notification_channels, is_active, created_at, updated_at
            """
            
            result = await conn.fetchrow(query, *params)
            
            if not result:
                return None
            
            return AlertThreshold(
                id=result["id"],
                user_id=result["user_id"],
                name=result["name"],
                geri_threshold=result["geri_threshold"],
                delta_threshold=result["delta_threshold"],
                conditions=result["conditions"],
                notification_channels=result["notification_channels"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                updated_at=result["updated_at"]
            )
        finally:
            await conn.close()
    
    async def get_user_thresholds(self, user_id: int) -> List[AlertThreshold]:
        """Get all alert thresholds for a user."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            results = await conn.fetch("""
                SELECT id, user_id, name, geri_threshold, delta_threshold, conditions, notification_channels, is_active, created_at, updated_at
                FROM alert_thresholds
                WHERE user_id = $1
                ORDER BY created_at DESC
            """, user_id)
            
            return [
                AlertThreshold(
                    id=result["id"],
                    user_id=result["user_id"],
                    name=result["name"],
                    geri_threshold=result["geri_threshold"],
                    delta_threshold=result["delta_threshold"],
                    conditions=result["conditions"],
                    notification_channels=result["notification_channels"],
                    is_active=result["is_active"],
                    created_at=result["created_at"],
                    updated_at=result["updated_at"]
                )
                for result in results
            ]
        finally:
            await conn.close()
    
    async def delete_alert_threshold(self, threshold_id: int, user_id: int):
        """Delete an alert threshold."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.execute("""
                DELETE FROM alert_thresholds 
                WHERE id = $1 AND user_id = $2
            """, threshold_id, user_id)
            
            return result != "DELETE 0"
        finally:
            await conn.close()
    
    async def process_scenario_result(
        self,
        user_id: int,
        scenario_data: Dict[str, Any]
    ) -> List[AlertEvent]:
        """Process a scenario result against user's alert thresholds."""
        # Get active thresholds for user
        thresholds = await self.get_user_thresholds(user_id)
        active_thresholds = [t for t in thresholds if t.is_active]
        
        triggered_events = []
        
        for threshold in active_thresholds:
            events = await self._check_threshold(threshold, scenario_data)
            triggered_events.extend(events)
        
        return triggered_events
    
    async def _check_threshold(
        self,
        threshold: AlertThreshold,
        scenario_data: Dict[str, Any]
    ) -> List[AlertEvent]:
        """Check if a scenario result triggers an alert threshold."""
        events = []
        
        geri_value = scenario_data.get("scenario", 0)
        delta_value = scenario_data.get("delta", 0)
        
        # Check GERI threshold
        if threshold.geri_threshold is not None and geri_value < threshold.geri_threshold:
            event = await self._create_alert_event(
                threshold,
                "geri_threshold",
                geri_value,
                scenario_data
            )
            events.append(event)
        
        # Check delta threshold
        if threshold.delta_threshold is not None and abs(delta_value) > threshold.delta_threshold:
            event = await self._create_alert_event(
                threshold,
                "delta_threshold",
                abs(delta_value),
                scenario_data
            )
            events.append(event)
        
        # Check custom conditions
        for condition_name, condition in threshold.conditions.items():
            if await self._check_custom_condition(condition, scenario_data):
                event = await self._create_alert_event(
                    threshold,
                    "custom",
                    scenario_data.get(condition.get("field", ""), 0),
                    scenario_data
                )
                events.append(event)
        
        # Send notifications for triggered events
        for event in events:
            await self._send_alert_notifications(threshold, event, scenario_data)
        
        return events
    
    async def _check_custom_condition(
        self,
        condition: Dict[str, Any],
        scenario_data: Dict[str, Any]
    ) -> bool:
        """Check if custom condition is met."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not all([field, operator, value]):
            return False
        
        scenario_value = scenario_data.get(field)
        if scenario_value is None:
            return False
        
        if operator == "lt":
            return scenario_value < value
        elif operator == "gt":
            return scenario_value > value
        elif operator == "eq":
            return scenario_value == value
        elif operator == "lte":
            return scenario_value <= value
        elif operator == "gte":
            return scenario_value >= value
        
        return False
    
    async def _create_alert_event(
        self,
        threshold: AlertThreshold,
        trigger_type: str,
        trigger_value: float,
        scenario_data: Dict[str, Any]
    ) -> AlertEvent:
        """Create and store an alert event."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                INSERT INTO alert_events (threshold_id, user_id, trigger_type, trigger_value, scenario_data, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id, threshold_id, user_id, trigger_type, trigger_value, scenario_data, created_at
            """, threshold.id, threshold.user_id, trigger_type, trigger_value, scenario_data)
            
            return AlertEvent(
                id=result["id"],
                threshold_id=result["threshold_id"],
                user_id=result["user_id"],
                trigger_type=result["trigger_type"],
                trigger_value=result["trigger_value"],
                scenario_data=result["scenario_data"],
                created_at=result["created_at"]
            )
        finally:
            await conn.close()
    
    async def _send_alert_notifications(
        self,
        threshold: AlertThreshold,
        event: AlertEvent,
        scenario_data: Dict[str, Any]
    ):
        """Send alert notifications through configured channels."""
        notification_payload = {
            "alert_id": event.id,
            "threshold_name": threshold.name,
            "trigger_type": event.trigger_type,
            "trigger_value": event.trigger_value,
            "scenario": {
                "baseline": scenario_data.get("baseline"),
                "scenario": scenario_data.get("scenario"),
                "delta": scenario_data.get("delta"),
                "shocks": scenario_data.get("shocks", [])
            },
            "timestamp": event.created_at.isoformat(),
            "message": self._generate_alert_message(threshold, event, scenario_data)
        }
        
        for channel in threshold.notification_channels:
            if channel.startswith("email:"):
                email_address = channel.split(":", 1)[1]
                self.delivery_service.deliver(
                    subscription_id=threshold.id,
                    channel="email",
                    address=email_address,
                    payload=notification_payload
                )
            elif channel.startswith("webhook:"):
                webhook_url = channel.split(":", 1)[1]
                self.delivery_service.deliver(
                    subscription_id=threshold.id,
                    channel="webhook",
                    address=webhook_url,
                    payload=notification_payload
                )
    
    def _generate_alert_message(
        self,
        threshold: AlertThreshold,
        event: AlertEvent,
        scenario_data: Dict[str, Any]
    ) -> str:
        """Generate a human-readable alert message."""
        if event.trigger_type == "geri_threshold":
            return f"ALERT: '{threshold.name}' - GERI value {scenario_data.get('scenario', 0):.2f} fell below threshold {threshold.geri_threshold:.2f}"
        elif event.trigger_type == "delta_threshold":
            return f"ALERT: '{threshold.name}' - Scenario delta {abs(scenario_data.get('delta', 0)):.2f} exceeded threshold {threshold.delta_threshold:.2f}"
        else:
            return f"ALERT: '{threshold.name}' - Custom condition triggered with value {event.trigger_value:.2f}"
    
    async def get_user_alert_events(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[AlertEvent]:
        """Get recent alert events for a user."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            results = await conn.fetch("""
                SELECT ae.id, ae.threshold_id, ae.user_id, ae.trigger_type, ae.trigger_value, ae.scenario_data, ae.created_at
                FROM alert_events ae
                WHERE ae.user_id = $1
                ORDER BY ae.created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            return [
                AlertEvent(
                    id=result["id"],
                    threshold_id=result["threshold_id"],
                    user_id=result["user_id"],
                    trigger_type=result["trigger_type"],
                    trigger_value=result["trigger_value"],
                    scenario_data=result["scenario_data"],
                    created_at=result["created_at"]
                )
                for result in results
            ]
        finally:
            await conn.close()

# Global service instance
_alert_threshold_service: Optional[AlertThresholdService] = None

def get_alert_threshold_service() -> AlertThresholdService:
    """Dependency injection for alert threshold service."""
    global _alert_threshold_service
    if _alert_threshold_service is None:
        postgres_dsn = os.environ.get("RIS_POSTGRES_DSN")
        if not postgres_dsn:
            raise RuntimeError("RIS_POSTGRES_DSN environment variable not set")
        _alert_threshold_service = AlertThresholdService(postgres_dsn)
    return _alert_threshold_service