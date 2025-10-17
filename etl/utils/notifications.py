"""
ETL Notification System

Provides notification capabilities for ETL pipeline events,
including email notifications, webhooks, and logging integrations.
"""

import asyncio
import logging
import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from src.core.config import get_settings


class NotificationLevel(Enum):
    """Notification severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    LOG = "log"


@dataclass
class NotificationEvent:
    """Represents a notification event"""
    level: NotificationLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "tags": self.tags
        }


class BaseNotifier(ABC):
    """Abstract base class for all notifiers"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"notifier.{name}")
        self.enabled = self.config.get("enabled", True)
        self.min_level = NotificationLevel(self.config.get("min_level", "info"))
    
    @abstractmethod
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send a notification"""
        pass
    
    def should_notify(self, event: NotificationEvent) -> bool:
        """Check if this event should trigger a notification"""
        if not self.enabled:
            return False
        
        # Check level threshold
        level_order = {
            NotificationLevel.DEBUG: 0,
            NotificationLevel.INFO: 1,
            NotificationLevel.WARNING: 2,
            NotificationLevel.ERROR: 3,
            NotificationLevel.CRITICAL: 4
        }
        
        return level_order[event.level] >= level_order[self.min_level]
    
    async def notify(self, event: NotificationEvent) -> bool:
        """Send notification if conditions are met"""
        if not self.should_notify(event):
            return True
        
        try:
            return await self.send_notification(event)
        except Exception as e:
            self.logger.error(f"Notification failed: {str(e)}")
            return False


class EmailNotifier(BaseNotifier):
    """Email notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("email", config)
        self.smtp_server = self.config.get("smtp_server", "localhost")
        self.smtp_port = self.config.get("smtp_port", 587)
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.from_email = self.config.get("from_email", "noreply@riskx.ai")
        self.to_emails = self.config.get("to_emails", [])
        self.use_tls = self.config.get("use_tls", True)
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"[RiskX {event.level.value.upper()}] {event.title}"
            
            # Create email body
            body = self._create_email_body(event)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            if self.config.get("dry_run", False):
                self.logger.info(f"DRY RUN: Would send email to {self.to_emails}")
                return True
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            self.logger.info(f"Email notification sent successfully to {self.to_emails}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def _create_email_body(self, event: NotificationEvent) -> str:
        """Create HTML email body"""
        level_colors = {
            NotificationLevel.DEBUG: "#6B7280",
            NotificationLevel.INFO: "#3B82F6",
            NotificationLevel.WARNING: "#F59E0B",
            NotificationLevel.ERROR: "#EF4444",
            NotificationLevel.CRITICAL: "#DC2626"
        }
        
        color = level_colors.get(event.level, "#6B7280")
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0; }}
                .content {{ border: 1px solid #E5E7EB; padding: 20px; border-radius: 0 0 5px 5px; }}
                .details {{ background-color: #F9FAFB; padding: 15px; margin-top: 15px; border-radius: 5px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #6B7280; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{event.title}</h2>
                <p>Level: {event.level.value.upper()} | Source: {event.source}</p>
            </div>
            <div class="content">
                <p>{event.message}</p>
                
                {self._format_details(event.details) if event.details else ""}
                
                {self._format_tags(event.tags) if event.tags else ""}
            </div>
            <div class="footer">
                <p>Sent by RiskX ETL Pipeline at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """
        return body
    
    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format details section"""
        if not details:
            return ""
        
        details_html = '<div class="details"><h4>Details:</h4><ul>'
        for key, value in details.items():
            details_html += f"<li><strong>{key}:</strong> {value}</li>"
        details_html += "</ul></div>"
        return details_html
    
    def _format_tags(self, tags: List[str]) -> str:
        """Format tags section"""
        if not tags:
            return ""
        
        return f'<div class="details"><h4>Tags:</h4><p>{", ".join(tags)}</p></div>'


class WebhookNotifier(BaseNotifier):
    """Webhook notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("webhook", config)
        self.webhook_url = self.config.get("webhook_url")
        self.timeout = self.config.get("timeout", 30)
        self.headers = self.config.get("headers", {"Content-Type": "application/json"})
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1)
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send webhook notification"""
        if not self.webhook_url:
            self.logger.error("Webhook URL not configured")
            return False
        
        payload = self._create_webhook_payload(event)
        
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status < 400:
                            self.logger.info(f"Webhook notification sent successfully (attempt {attempt + 1})")
                            return True
                        else:
                            self.logger.warning(f"Webhook returned status {response.status} (attempt {attempt + 1})")
                            
            except Exception as e:
                self.logger.error(f"Webhook notification failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
    def _create_webhook_payload(self, event: NotificationEvent) -> Dict[str, Any]:
        """Create webhook payload"""
        return {
            "notification": event.to_dict(),
            "service": "riskx-etl",
            "version": "1.0"
        }


class SlackNotifier(BaseNotifier):
    """Slack notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("slack", config)
        self.webhook_url = self.config.get("webhook_url")
        self.channel = self.config.get("channel", "#alerts")
        self.username = self.config.get("username", "RiskX ETL")
        self.icon_emoji = self.config.get("icon_emoji", ":warning:")
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send Slack notification"""
        if not self.webhook_url:
            self.logger.error("Slack webhook URL not configured")
            return False
        
        payload = self._create_slack_payload(event)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Slack notification sent successfully")
                        return True
                    else:
                        self.logger.error(f"Slack notification failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Slack notification error: {str(e)}")
            return False
    
    def _create_slack_payload(self, event: NotificationEvent) -> Dict[str, Any]:
        """Create Slack message payload"""
        level_emojis = {
            NotificationLevel.DEBUG: ":grey_question:",
            NotificationLevel.INFO: ":information_source:",
            NotificationLevel.WARNING: ":warning:",
            NotificationLevel.ERROR: ":x:",
            NotificationLevel.CRITICAL: ":rotating_light:"
        }
        
        emoji = level_emojis.get(event.level, ":question:")
        
        text = f"{emoji} *{event.title}*\n{event.message}"
        
        if event.details:
            text += "\n\n*Details:*\n"
            for key, value in event.details.items():
                text += f"• {key}: {value}\n"
        
        return {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "text": text,
            "attachments": [
                {
                    "color": self._get_slack_color(event.level),
                    "fields": [
                        {
                            "title": "Source",
                            "value": event.source,
                            "short": True
                        },
                        {
                            "title": "Level",
                            "value": event.level.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "short": False
                        }
                    ]
                }
            ]
        }
    
    def _get_slack_color(self, level: NotificationLevel) -> str:
        """Get Slack color for notification level"""
        colors = {
            NotificationLevel.DEBUG: "#6B7280",
            NotificationLevel.INFO: "#3B82F6",
            NotificationLevel.WARNING: "#F59E0B",
            NotificationLevel.ERROR: "#EF4444",
            NotificationLevel.CRITICAL: "#DC2626"
        }
        return colors.get(level, "#6B7280")


class LogNotifier(BaseNotifier):
    """Log-based notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("log", config)
        self.log_level_mapping = {
            NotificationLevel.DEBUG: logging.DEBUG,
            NotificationLevel.INFO: logging.INFO,
            NotificationLevel.WARNING: logging.WARNING,
            NotificationLevel.ERROR: logging.ERROR,
            NotificationLevel.CRITICAL: logging.CRITICAL
        }
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Log notification event"""
        try:
            log_level = self.log_level_mapping.get(event.level, logging.INFO)
            
            message = f"[{event.source}] {event.title}: {event.message}"
            
            if event.details:
                details_str = json.dumps(event.details, default=str)
                message += f" | Details: {details_str}"
            
            if event.tags:
                message += f" | Tags: {', '.join(event.tags)}"
            
            self.logger.log(log_level, message)
            return True
            
        except Exception as e:
            self.logger.error(f"Log notification error: {str(e)}")
            return False


class NotificationManager:
    """Central notification management"""
    
    def __init__(self):
        self.notifiers: Dict[str, BaseNotifier] = {}
        self.logger = logging.getLogger("notification_manager")
        self.settings = get_settings()
        self._initialize_notifiers()
    
    def _initialize_notifiers(self):
        """Initialize notifiers based on configuration"""
        notification_config = getattr(self.settings, 'notifications', {})
        
        # Email notifier
        if notification_config.get('email', {}).get('enabled', False):
            email_config = notification_config['email']
            self.add_notifier('email', EmailNotifier(email_config))
        
        # Webhook notifier
        if notification_config.get('webhook', {}).get('enabled', False):
            webhook_config = notification_config['webhook']
            self.add_notifier('webhook', WebhookNotifier(webhook_config))
        
        # Slack notifier
        if notification_config.get('slack', {}).get('enabled', False):
            slack_config = notification_config['slack']
            self.add_notifier('slack', SlackNotifier(slack_config))
        
        # Log notifier (always enabled)
        log_config = notification_config.get('log', {'enabled': True})
        self.add_notifier('log', LogNotifier(log_config))
    
    def add_notifier(self, name: str, notifier: BaseNotifier):
        """Add a notifier"""
        self.notifiers[name] = notifier
        self.logger.info(f"Added notifier: {name}")
    
    def remove_notifier(self, name: str):
        """Remove a notifier"""
        if name in self.notifiers:
            del self.notifiers[name]
            self.logger.info(f"Removed notifier: {name}")
    
    async def notify(self, event: NotificationEvent) -> Dict[str, bool]:
        """Send notification through all configured notifiers"""
        results = {}
        
        for name, notifier in self.notifiers.items():
            try:
                results[name] = await notifier.notify(event)
            except Exception as e:
                self.logger.error(f"Notifier {name} failed: {str(e)}")
                results[name] = False
        
        return results
    
    async def notify_pipeline_start(self, pipeline_name: str, details: Dict[str, Any] = None):
        """Notify pipeline start"""
        event = NotificationEvent(
            level=NotificationLevel.INFO,
            title=f"Pipeline Started: {pipeline_name}",
            message=f"ETL pipeline '{pipeline_name}' has started execution",
            source="etl_pipeline",
            details=details or {},
            tags=["pipeline", "start"]
        )
        return await self.notify(event)
    
    async def notify_pipeline_success(self, pipeline_name: str, details: Dict[str, Any] = None):
        """Notify pipeline success"""
        event = NotificationEvent(
            level=NotificationLevel.INFO,
            title=f"Pipeline Completed: {pipeline_name}",
            message=f"ETL pipeline '{pipeline_name}' completed successfully",
            source="etl_pipeline",
            details=details or {},
            tags=["pipeline", "success"]
        )
        return await self.notify(event)
    
    async def notify_pipeline_failure(self, pipeline_name: str, error: str, details: Dict[str, Any] = None):
        """Notify pipeline failure"""
        event = NotificationEvent(
            level=NotificationLevel.ERROR,
            title=f"Pipeline Failed: {pipeline_name}",
            message=f"ETL pipeline '{pipeline_name}' failed: {error}",
            source="etl_pipeline",
            details=details or {},
            tags=["pipeline", "failure"]
        )
        return await self.notify(event)
    
    async def notify_data_quality_issue(self, source_name: str, issue: str, details: Dict[str, Any] = None):
        """Notify data quality issue"""
        event = NotificationEvent(
            level=NotificationLevel.WARNING,
            title=f"Data Quality Issue: {source_name}",
            message=f"Data quality issue detected in '{source_name}': {issue}",
            source="data_validation",
            details=details or {},
            tags=["data_quality", "warning"]
        )
        return await self.notify(event)
    
    async def notify_source_unavailable(self, source_name: str, error: str, details: Dict[str, Any] = None):
        """Notify data source unavailable"""
        event = NotificationEvent(
            level=NotificationLevel.ERROR,
            title=f"Data Source Unavailable: {source_name}",
            message=f"Data source '{source_name}' is unavailable: {error}",
            source="data_extraction",
            details=details or {},
            tags=["source_unavailable", "error"]
        )
        return await self.notify(event)
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Health check all notifiers"""
        health_status = {}
        
        for name, notifier in self.notifiers.items():
            try:
                # Send a test notification if supported
                test_event = NotificationEvent(
                    level=NotificationLevel.DEBUG,
                    title="Health Check",
                    message="Testing notification system",
                    source="health_check",
                    tags=["test", "health_check"]
                )
                
                # Override min_level temporarily for health check
                original_min_level = notifier.min_level
                notifier.min_level = NotificationLevel.DEBUG
                
                success = await notifier.notify(test_event)
                
                # Restore original min_level
                notifier.min_level = original_min_level
                
                health_status[name] = {
                    "status": "healthy" if success else "unhealthy",
                    "enabled": notifier.enabled,
                    "min_level": notifier.min_level.value
                }
                
            except Exception as e:
                health_status[name] = {
                    "status": "error",
                    "error": str(e),
                    "enabled": notifier.enabled
                }
        
        return health_status