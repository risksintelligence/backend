"""Production alert delivery service with retry logic and comprehensive error handling."""
from __future__ import annotations

import json
import logging
import os
import smtplib
import asyncio
import httpx
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import List, Optional, Dict, Any
from urllib import request as urllib_request


logger = logging.getLogger(__name__)


@dataclass
class DeliveredAlert:
    subscription_id: int
    channel: str
    address: str
    payload: dict
    delivered_at: str
    delivery_status: str = "success"  # success, failed, pending, retrying
    error_message: Optional[str] = None
    retry_count: int = 0


class AlertDeliveryService:
    """Production alert delivery service with retry logic, error handling, and delivery tracking.

    Supports SMTP email and HTTP webhooks with automatic retries, delivery status tracking,
    and comprehensive error handling for production reliability."""

    def __init__(self, postgres_dsn: Optional[str] = None) -> None:
        self._events: List[DeliveredAlert] = []
        self.postgres_dsn = postgres_dsn or os.getenv("RIS_POSTGRES_DSN")
        
        # SMTP Configuration
        self._smtp_host = os.getenv("ALERT_SMTP_HOST")
        self._smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self._smtp_user = os.getenv("ALERT_SMTP_USERNAME")
        self._smtp_pass = os.getenv("ALERT_SMTP_PASSWORD")
        self._smtp_from = os.getenv("ALERT_SMTP_FROM", "alerts@risksx.io")
        self._smtp_tls = os.getenv("ALERT_SMTP_TLS", "true").lower() == "true"
        
        # Webhook Configuration
        self._webhook_token = os.getenv("ALERT_WEBHOOK_TOKEN")
        self._webhook_timeout = int(os.getenv("ALERT_WEBHOOK_TIMEOUT", "30"))
        
        # Retry Configuration
        self._max_retries = int(os.getenv("ALERT_MAX_RETRIES", "3"))
        self._retry_delay = int(os.getenv("ALERT_RETRY_DELAY", "60"))  # seconds

    async def deliver(self, subscription_id: int, channel: str, address: str, payload: dict) -> DeliveredAlert:
        """Deliver alert with retry logic and error handling."""
        delivery_status = "pending"
        error_message = None
        
        try:
            if channel == "email":
                await self._send_email_with_retry(address, payload)
                delivery_status = "success"
            elif channel == "webhook":
                await self._send_webhook_with_retry(address, payload)
                delivery_status = "success"
            else:
                logger.warning("Unsupported alert channel %s", channel)
                delivery_status = "failed"
                error_message = f"Unsupported channel: {channel}"
        except Exception as e:
            logger.error(f"Alert delivery failed for {channel}:{address} - {str(e)}")
            delivery_status = "failed"
            error_message = str(e)
        
        event = DeliveredAlert(
            subscription_id=subscription_id,
            channel=channel,
            address=address,
            payload=payload,
            delivered_at=datetime.utcnow().isoformat(),
            delivery_status=delivery_status,
            error_message=error_message,
            retry_count=0
        )
        
        self._events.append(event)
        return event
    
    async def _send_email_with_retry(self, recipient: str, payload: dict):
        """Send email with retry logic."""
        for attempt in range(self._max_retries + 1):
            try:
                await self._send_email_async(recipient, payload)
                return  # Success
            except Exception as e:
                if attempt == self._max_retries:
                    raise e  # Final attempt failed
                
                logger.warning(f"Email delivery attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
    
    async def _send_webhook_with_retry(self, endpoint: str, payload: dict):
        """Send webhook with retry logic."""
        for attempt in range(self._max_retries + 1):
            try:
                await self._send_webhook_async(endpoint, payload)
                return  # Success
            except Exception as e:
                if attempt == self._max_retries:
                    raise e  # Final attempt failed
                
                logger.warning(f"Webhook delivery attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
    
    async def _send_email_async(self, recipient: str, payload: dict):
        """Send email asynchronously."""
        if not self._smtp_host:
            raise Exception("ALERT_SMTP_HOST not configured")
        
        # Create email message
        msg = EmailMessage()
        msg["Subject"] = f"RiskSX Alert: {payload.get('alert_type', 'System Alert')}"
        msg["From"] = self._smtp_from
        msg["To"] = recipient
        
        # Create HTML email content
        html_content = self._create_email_html(payload)
        msg.set_content(self._create_email_text(payload))  # Text version
        msg.add_alternative(html_content, subtype='html')  # HTML version
        
        # Send email in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp_email, msg)
    
    def _send_smtp_email(self, msg: EmailMessage):
        """Send email via SMTP (blocking operation)."""
        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30) as smtp:
            if self._smtp_tls:
                smtp.starttls()
            if self._smtp_user and self._smtp_pass:
                smtp.login(self._smtp_user, self._smtp_pass)
            smtp.send_message(msg)
    
    async def _send_webhook_async(self, endpoint: str, payload: dict):
        """Send webhook asynchronously."""
        headers = {"Content-Type": "application/json"}
        if self._webhook_token:
            headers["Authorization"] = f"Bearer {self._webhook_token}"
        
        async with httpx.AsyncClient(timeout=self._webhook_timeout) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()  # Raise exception for HTTP errors
    
    def _create_email_text(self, payload: dict) -> str:
        """Create plain text email content."""
        alert_type = payload.get('alert_type', 'Alert')
        threshold_name = payload.get('threshold_name', 'Unnamed Alert')
        message = payload.get('message', 'Alert triggered')
        scenario = payload.get('scenario', {})
        
        return f"""
RiskSX Intelligence System Alert

Alert: {threshold_name}
Type: {alert_type}

{message}

Scenario Details:
- Baseline GERI: {scenario.get('baseline', 'N/A')}
- Scenario GERI: {scenario.get('scenario', 'N/A')}
- Delta: {scenario.get('delta', 'N/A')}

Timestamp: {payload.get('timestamp', 'Unknown')}

---
This is an automated alert from the RiskSX Intelligence System.
Do not reply to this email.
        """.strip()
    
    def _create_email_html(self, payload: dict) -> str:
        """Create HTML email content."""
        alert_type = payload.get('alert_type', 'Alert')
        threshold_name = payload.get('threshold_name', 'Unnamed Alert')
        message = payload.get('message', 'Alert triggered')
        scenario = payload.get('scenario', {})
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'JetBrains Mono', monospace; background-color: #ffffff; color: #1e3a8a; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f8fafc; border-left: 4px solid #dc2626; padding: 20px; margin-bottom: 20px; }}
        .alert-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
        .alert-message {{ font-size: 14px; color: #6b7280; }}
        .scenario-data {{ background-color: #f8fafc; padding: 15px; border: 1px solid #e2e8f0; border-radius: 4px; }}
        .data-row {{ margin: 5px 0; }}
        .label {{ color: #6b7280; font-weight: bold; }}
        .value {{ color: #1e3a8a; }}
        .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #6b7280; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="alert-title">⚠️ RiskSX Alert: {threshold_name}</div>
            <div class="alert-message">{message}</div>
        </div>
        
        <div class="scenario-data">
            <h3>Scenario Details</h3>
            <div class="data-row">
                <span class="label">Baseline GERI:</span>
                <span class="value">{scenario.get('baseline', 'N/A')}</span>
            </div>
            <div class="data-row">
                <span class="label">Scenario GERI:</span>
                <span class="value">{scenario.get('scenario', 'N/A')}</span>
            </div>
            <div class="data-row">
                <span class="label">Delta:</span>
                <span class="value">{scenario.get('delta', 'N/A')}</span>
            </div>
            <div class="data-row">
                <span class="label">Timestamp:</span>
                <span class="value">{payload.get('timestamp', 'Unknown')}</span>
            </div>
        </div>
        
        <div class="footer">
            This is an automated alert from the RiskSX Intelligence System.<br>
            Do not reply to this email.
        </div>
    </div>
</body>
</html>
        """.strip()
    
    def _send_email(self, recipient: str, payload: dict) -> None:
        if not self._smtp_host:
            logger.info("ALERT_SMTP_HOST not set; logging email alert to %s", recipient)
            logger.debug(json.dumps({"recipient": recipient, "payload": payload}, indent=2))
            return
        msg = EmailMessage()
        msg["Subject"] = "RiskSX Alert"
        msg["From"] = self._smtp_from
        msg["To"] = recipient
        msg.set_content(json.dumps(payload, indent=2))
        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
            if self._smtp_tls:
                smtp.starttls()
            if self._smtp_user and self._smtp_pass:
                smtp.login(self._smtp_user, self._smtp_pass)
            smtp.send_message(msg)

    def _send_webhook(self, endpoint: str, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._webhook_token:
            headers["Authorization"] = f"Bearer {self._webhook_token}"
        req = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=10) as response:
                response.read()
        except Exception as exc:  # pragma: no cover - network interaction
            logger.warning("Webhook delivery failed: %s", exc)

    def history(self) -> List[DeliveredAlert]:
        return self._events[-50:]
