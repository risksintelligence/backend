"""Alert delivery service for handling notification dispatch."""
from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertDelivery:
    id: str
    alert_id: str
    channel: AlertChannel
    recipient: str
    status: str
    attempts: int
    last_attempt: Optional[datetime]
    delivered_at: Optional[datetime]
    error_message: Optional[str]

class AlertDeliveryService:
    """Service for managing alert delivery and notifications."""
    
    def __init__(self):
        self.deliveries: Dict[str, AlertDelivery] = {}
        self.delivery_counter = 0
    
    async def deliver_alert(
        self,
        alert_id: str,
        channel: AlertChannel,
        recipient: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM
    ) -> AlertDelivery:
        """Deliver an alert via the specified channel."""
        delivery_id = f"delivery_{self.delivery_counter}"
        self.delivery_counter += 1
        
        delivery = AlertDelivery(
            id=delivery_id,
            alert_id=alert_id,
            channel=channel,
            recipient=recipient,
            status="pending",
            attempts=0,
            last_attempt=None,
            delivered_at=None,
            error_message=None
        )
        
        self.deliveries[delivery_id] = delivery
        
        # Simulate delivery attempt
        try:
            delivery.attempts += 1
            delivery.last_attempt = datetime.now(timezone.utc)
            
            # Mock delivery logic
            if channel == AlertChannel.EMAIL:
                await self._deliver_email(recipient, message)
            elif channel == AlertChannel.SLACK:
                await self._deliver_slack(recipient, message)
            elif channel == AlertChannel.WEBHOOK:
                await self._deliver_webhook(recipient, message)
            elif channel == AlertChannel.SMS:
                await self._deliver_sms(recipient, message)
            
            delivery.status = "delivered"
            delivery.delivered_at = datetime.now(timezone.utc)
            
        except Exception as e:
            delivery.status = "failed"
            delivery.error_message = str(e)
            logger.error(f"Failed to deliver alert {alert_id}: {e}")
        
        return delivery
    
    async def _deliver_email(self, recipient: str, message: str):
        """Mock email delivery."""
        await asyncio.sleep(0.1)  # Simulate network delay
        logger.info(f"Email delivered to {recipient}: {message[:50]}...")
    
    async def _deliver_slack(self, recipient: str, message: str):
        """Mock Slack delivery."""
        await asyncio.sleep(0.1)
        logger.info(f"Slack message delivered to {recipient}: {message[:50]}...")
    
    async def _deliver_webhook(self, recipient: str, message: str):
        """Mock webhook delivery."""
        await asyncio.sleep(0.1)
        logger.info(f"Webhook delivered to {recipient}: {message[:50]}...")
    
    async def _deliver_sms(self, recipient: str, message: str):
        """Mock SMS delivery."""
        await asyncio.sleep(0.1)
        logger.info(f"SMS delivered to {recipient}: {message[:50]}...")
    
    async def get_delivery_status(self, delivery_id: str) -> Optional[AlertDelivery]:
        """Get the status of a specific delivery."""
        return self.deliveries.get(delivery_id)
    
    async def get_alert_deliveries(self, alert_id: str) -> List[AlertDelivery]:
        """Get all deliveries for a specific alert."""
        return [d for d in self.deliveries.values() if d.alert_id == alert_id]
    
    async def retry_failed_delivery(self, delivery_id: str) -> bool:
        """Retry a failed delivery."""
        delivery = self.deliveries.get(delivery_id)
        if not delivery or delivery.status != "failed":
            return False
        
        try:
            delivery.attempts += 1
            delivery.last_attempt = datetime.now(timezone.utc)
            delivery.status = "pending"
            delivery.error_message = None
            
            # Retry delivery logic would go here
            delivery.status = "delivered"
            delivery.delivered_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            delivery.status = "failed"
            delivery.error_message = str(e)
            return False
    
    async def get_delivery_metrics(self) -> Dict[str, Any]:
        """Get delivery metrics and statistics."""
        total_deliveries = len(self.deliveries)
        successful = len([d for d in self.deliveries.values() if d.status == "delivered"])
        failed = len([d for d in self.deliveries.values() if d.status == "failed"])
        pending = len([d for d in self.deliveries.values() if d.status == "pending"])
        
        return {
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "pending_deliveries": pending,
            "success_rate": successful / total_deliveries if total_deliveries > 0 else 0,
            "deliveries_by_channel": self._get_deliveries_by_channel()
        }
    
    def _get_deliveries_by_channel(self) -> Dict[str, int]:
        """Get delivery counts by channel."""
        channel_counts = {}
        for delivery in self.deliveries.values():
            channel = delivery.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        return channel_counts

# Global service instance
_alert_delivery_service: Optional[AlertDeliveryService] = None

def get_alert_delivery_service() -> AlertDeliveryService:
    """Get the global alert delivery service instance."""
    global _alert_delivery_service
    if _alert_delivery_service is None:
        _alert_delivery_service = AlertDeliveryService()
    return _alert_delivery_service