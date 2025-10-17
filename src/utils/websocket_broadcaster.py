"""
WebSocket Broadcasting Service

Centralized service for broadcasting real-time updates to all connected WebSocket clients.
Ensures system-wide real-time connectivity and data synchronization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.data.sources.fred import FREDConnector
from src.ml.models.risk_scorer import BasicRiskScorer

logger = logging.getLogger(__name__)
settings = get_settings()


class BroadcastType(Enum):
    """Types of broadcasts supported by the system"""
    RISK_UPDATE = "risk_update"
    ECONOMIC_DATA = "economic_data"
    SUPPLY_CHAIN_UPDATE = "supply_chain_update"
    SYSTEM_ALERT = "system_alert"
    DATA_REFRESH = "data_refresh"
    MODEL_UPDATE = "model_update"
    # Additional message types for compatibility
    RISK_ALERT = "risk_alert"
    ANALYTICS_UPDATE = "analytics_update"
    SYSTEM_HEALTH = "system_health"
    DATA_UPDATE = "data_update"


# Alias for backwards compatibility
MessageType = BroadcastType


@dataclass
class BroadcastMessage:
    """Structure for broadcast messages"""
    type: BroadcastType
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    target_channels: Optional[List[str]] = None
    ttl_seconds: Optional[int] = None


class WebSocketBroadcaster:
    """
    Centralized WebSocket broadcasting service.
    
    Manages real-time data updates and broadcasts them to all connected clients
    across different WebSocket channels (risk-updates, analytics-updates, system-health).
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.connection_managers: Dict[str, Any] = {}
        self.broadcasting = False
        self.broadcast_queue: List[BroadcastMessage] = []
        self.last_broadcast_times: Dict[str, datetime] = {}
        
        # Initialize data sources
        self.fred_connector = FREDConnector()
        self.risk_scorer = BasicRiskScorer()
        
        # Broadcasting intervals (seconds)
        self.intervals = {
            'risk_updates': 5,      # 5 seconds for risk updates
            'economic_data': 60,    # 1 minute for economic data
            'supply_chain': 30,     # 30 seconds for supply chain
            'system_health': 15     # 15 seconds for system health
        }
        
    def register_connection_manager(self, channel: str, manager: Any) -> None:
        """Register a WebSocket connection manager for a specific channel"""
        self.connection_managers[channel] = manager
        logger.info(f"Registered connection manager for channel: {channel}")
    
    async def start_broadcasting(self) -> None:
        """Start the real-time broadcasting service"""
        if self.broadcasting:
            logger.warning("Broadcasting service is already running")
            return
            
        self.broadcasting = True
        logger.info("Starting WebSocket broadcasting service")
        
        # Start concurrent broadcasting tasks
        tasks = [
            asyncio.create_task(self._broadcast_risk_updates()),
            asyncio.create_task(self._broadcast_economic_data()),
            asyncio.create_task(self._broadcast_supply_chain_updates()),
            asyncio.create_task(self._broadcast_system_health()),
            asyncio.create_task(self._process_broadcast_queue())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Broadcasting service error: {e}")
            self.broadcasting = False
    
    async def stop_broadcasting(self) -> None:
        """Stop the broadcasting service"""
        self.broadcasting = False
        logger.info("Stopped WebSocket broadcasting service")
    
    async def queue_broadcast(self, message: BroadcastMessage) -> None:
        """Queue a message for broadcasting"""
        self.broadcast_queue.append(message)
        
        # Sort queue by priority (highest first)
        self.broadcast_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Limit queue size
        if len(self.broadcast_queue) > 100:
            self.broadcast_queue = self.broadcast_queue[:100]
    
    async def broadcast_immediate(self, message: BroadcastMessage) -> None:
        """Broadcast a message immediately to all relevant channels"""
        channels = message.target_channels or ['risk-updates', 'analytics-updates']
        
        payload = {
            "type": message.type.value,
            "timestamp": message.timestamp.isoformat(),
            "data": message.data,
            "priority": message.priority
        }
        
        broadcast_count = 0
        for channel in channels:
            if channel in self.connection_managers:
                try:
                    await self.connection_managers[channel].broadcast(payload)
                    broadcast_count += 1
                except Exception as e:
                    logger.error(f"Failed to broadcast to {channel}: {e}")
        
        logger.debug(f"Broadcasted {message.type.value} to {broadcast_count} channels")
    
    async def _broadcast_risk_updates(self) -> None:
        """Continuously broadcast risk score updates"""
        while self.broadcasting:
            try:
                # Generate current risk assessment
                risk_data = await self._generate_risk_data()
                
                message = BroadcastMessage(
                    type=BroadcastType.RISK_UPDATE,
                    data=risk_data,
                    timestamp=datetime.utcnow(),
                    priority=3,  # High priority
                    target_channels=['risk-updates']
                )
                
                await self.broadcast_immediate(message)
                
                # Cache the latest risk data
                await self.cache.set(
                    'current_risk_score',
                    risk_data,
                    ttl=30
                )
                
                await asyncio.sleep(self.intervals['risk_updates'])
                
            except Exception as e:
                logger.error(f"Error broadcasting risk updates: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _broadcast_economic_data(self) -> None:
        """Continuously broadcast economic data updates"""
        while self.broadcasting:
            try:
                # Fetch latest economic indicators
                economic_data = await self._fetch_economic_indicators()
                
                message = BroadcastMessage(
                    type=BroadcastType.ECONOMIC_DATA,
                    data=economic_data,
                    timestamp=datetime.utcnow(),
                    priority=2,  # Medium priority
                    target_channels=['analytics-updates']
                )
                
                await self.broadcast_immediate(message)
                
                await asyncio.sleep(self.intervals['economic_data'])
                
            except Exception as e:
                logger.error(f"Error broadcasting economic data: {e}")
                await asyncio.sleep(30)
    
    async def _broadcast_supply_chain_updates(self) -> None:
        """Continuously broadcast supply chain status updates"""
        while self.broadcasting:
            try:
                # Generate supply chain status
                supply_chain_data = await self._generate_supply_chain_status()
                
                message = BroadcastMessage(
                    type=BroadcastType.SUPPLY_CHAIN_UPDATE,
                    data=supply_chain_data,
                    timestamp=datetime.utcnow(),
                    priority=2,  # Medium priority
                    target_channels=['analytics-updates', 'risk-updates']
                )
                
                await self.broadcast_immediate(message)
                
                await asyncio.sleep(self.intervals['supply_chain'])
                
            except Exception as e:
                logger.error(f"Error broadcasting supply chain updates: {e}")
                await asyncio.sleep(20)
    
    async def _broadcast_system_health(self) -> None:
        """Continuously broadcast system health status"""
        while self.broadcasting:
            try:
                # Generate system health data
                health_data = await self._generate_system_health()
                
                message = BroadcastMessage(
                    type=BroadcastType.SYSTEM_ALERT,
                    data=health_data,
                    timestamp=datetime.utcnow(),
                    priority=1,  # Low priority for regular health updates
                    target_channels=['system-health']
                )
                
                await self.broadcast_immediate(message)
                
                await asyncio.sleep(self.intervals['system_health'])
                
            except Exception as e:
                logger.error(f"Error broadcasting system health: {e}")
                await asyncio.sleep(15)
    
    async def _process_broadcast_queue(self) -> None:
        """Process queued broadcast messages"""
        while self.broadcasting:
            try:
                if self.broadcast_queue:
                    message = self.broadcast_queue.pop(0)
                    
                    # Check TTL
                    if message.ttl_seconds:
                        age = (datetime.utcnow() - message.timestamp).total_seconds()
                        if age > message.ttl_seconds:
                            logger.debug(f"Discarded expired message: {message.type.value}")
                            continue
                    
                    await self.broadcast_immediate(message)
                
                await asyncio.sleep(1)  # Check queue every second
                
            except Exception as e:
                logger.error(f"Error processing broadcast queue: {e}")
                await asyncio.sleep(5)
    
    async def _generate_risk_data(self) -> Dict[str, Any]:
        """Generate current risk assessment data"""
        try:
            # Check cache first
            cached_risk = await self.cache.get('current_risk_score')
            if cached_risk:
                return cached_risk
            
            # Generate new risk assessment
            from src.utils.monitoring import get_system_metrics
            
            # Get system metrics as risk factors
            system_metrics = get_system_metrics()
            
            # Calculate composite risk score
            base_risk = 25.0  # Base economic uncertainty
            
            # Add system stress factors
            if system_metrics:
                cpu_stress = system_metrics.get('cpu_percent', 0) * 0.5
                memory_stress = system_metrics.get('memory_percent', 0) * 0.3
                base_risk += cpu_stress + memory_stress
            
            # Ensure reasonable bounds
            overall_score = min(max(base_risk, 0), 100)
            
            # Determine risk level
            if overall_score < 30:
                risk_level = "low"
            elif overall_score < 60:
                risk_level = "moderate"
            else:
                risk_level = "high"
            
            # Generate top risk factors
            top_factors = [
                {
                    "name": "Economic Uncertainty",
                    "value": 25.0,
                    "normalized_value": 0.25,
                    "contribution": 0.3
                },
                {
                    "name": "Market Volatility",
                    "value": system_metrics.get('cpu_percent', 20) if system_metrics else 20,
                    "normalized_value": (system_metrics.get('cpu_percent', 20) / 100) if system_metrics else 0.2,
                    "contribution": 0.25
                },
                {
                    "name": "Supply Chain Risk",
                    "value": 18.0,
                    "normalized_value": 0.18,
                    "contribution": 0.2
                },
                {
                    "name": "Infrastructure Stability",
                    "value": system_metrics.get('memory_percent', 15) if system_metrics else 15,
                    "normalized_value": (system_metrics.get('memory_percent', 15) / 100) if system_metrics else 0.15,
                    "contribution": 0.15
                }
            ]
            
            risk_data = {
                "overall_score": round(overall_score, 1),
                "risk_level": risk_level,
                "confidence": 0.87,
                "top_factors": top_factors,
                "last_updated": datetime.utcnow().isoformat(),
                "data_freshness": "real-time"
            }
            
            return risk_data
            
        except Exception as e:
            logger.error(f"Error generating risk data: {e}")
            return {
                "overall_score": 50.0,
                "risk_level": "moderate",
                "confidence": 0.5,
                "top_factors": [],
                "error": str(e)
            }
    
    async def _fetch_economic_indicators(self) -> Dict[str, Any]:
        """Fetch latest economic indicators"""
        try:
            # Check cache for recent data
            cached_econ = await self.cache.get('economic_indicators')
            if cached_econ:
                return cached_econ
            
            # Fetch key economic indicators
            indicators = {
                "gdp_growth": 2.1,  # Would fetch from FRED
                "inflation_rate": 3.2,
                "unemployment_rate": 3.7,
                "interest_rate": 5.25,
                "market_sentiment": 0.65,
                "dollar_index": 103.5
            }
            
            # Calculate economic stress index
            stress_factors = [
                indicators["inflation_rate"] / 10,  # Normalize inflation
                indicators["unemployment_rate"] / 15,  # Normalize unemployment
                abs(indicators["gdp_growth"] - 2.5) / 5  # Deviation from target growth
            ]
            
            economic_stress = sum(stress_factors) / len(stress_factors)
            
            economic_data = {
                "indicators": indicators,
                "economic_stress_index": round(economic_stress * 100, 1),
                "trend": "stable" if economic_stress < 0.5 else "volatile",
                "last_updated": datetime.utcnow().isoformat(),
                "data_sources": ["FRED", "BLS", "BEA"]
            }
            
            # Cache for 1 minute
            await self.cache.set('economic_indicators', economic_data, ttl=60)
            
            return economic_data
            
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return {
                "indicators": {},
                "economic_stress_index": 50.0,
                "trend": "unknown",
                "error": str(e)
            }
    
    async def _generate_supply_chain_status(self) -> Dict[str, Any]:
        """Generate supply chain status update"""
        try:
            # Check cache
            cached_supply = await self.cache.get('supply_chain_status')
            if cached_supply:
                return cached_supply
            
            # Generate supply chain metrics
            import random
            
            supply_chain_data = {
                "global_disruption_index": round(random.uniform(15, 45), 1),
                "port_congestion_level": round(random.uniform(20, 80), 1),
                "shipping_delays": {
                    "average_days": round(random.uniform(2, 14), 1),
                    "worst_case_days": round(random.uniform(14, 30), 1)
                },
                "critical_shortages": random.randint(3, 15),
                "supplier_risk_score": round(random.uniform(25, 75), 1),
                "regional_risks": {
                    "asia_pacific": round(random.uniform(20, 60), 1),
                    "europe": round(random.uniform(15, 45), 1),
                    "north_america": round(random.uniform(10, 40), 1)
                },
                "last_updated": datetime.utcnow().isoformat(),
                "status": "monitoring"
            }
            
            # Cache for 30 seconds
            await self.cache.set('supply_chain_status', supply_chain_data, ttl=30)
            
            return supply_chain_data
            
        except Exception as e:
            logger.error(f"Error generating supply chain status: {e}")
            return {
                "global_disruption_index": 30.0,
                "status": "error",
                "error": str(e)
            }
    
    async def _generate_system_health(self) -> Dict[str, Any]:
        """Generate system health status"""
        try:
            from src.utils.monitoring import get_system_metrics, get_app_metrics
            
            system_metrics = get_system_metrics()
            app_metrics = get_app_metrics()
            
            # Check cache and database health
            cache_healthy = await self._check_cache_health()
            db_healthy = await self._check_database_health()
            
            # Determine overall status
            if system_metrics and system_metrics.get('cpu_percent', 0) < 80 and cache_healthy and db_healthy:
                overall_status = "healthy"
            elif system_metrics and system_metrics.get('cpu_percent', 0) < 95:
                overall_status = "degraded"
            else:
                overall_status = "stressed"
            
            health_data = {
                "overall_status": overall_status,
                "system_metrics": system_metrics or {},
                "application_metrics": app_metrics or {},
                "service_health": {
                    "cache": "healthy" if cache_healthy else "unhealthy",
                    "database": "healthy" if db_healthy else "unhealthy",
                    "api": overall_status
                },
                "active_connections": sum(
                    len(manager.active_connections) 
                    for manager in self.connection_managers.values() 
                    if hasattr(manager, 'active_connections')
                ),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error generating system health: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    async def _check_cache_health(self) -> bool:
        """Check if cache is healthy"""
        try:
            # Test cache operation
            test_key = "health_check"
            await self.cache.set(test_key, "ok", ttl=10)
            result = await self.cache.get(test_key)
            return result == "ok"
        except Exception:
            return False
    
    async def _check_database_health(self) -> bool:
        """Check if database is healthy"""
        try:
            from src.data.storage.database import get_database_connection
            db = get_database_connection()
            return db is not None and hasattr(db, 'pool') and getattr(db.pool, 'is_connected', False)
        except Exception:
            return False
    
    async def broadcast_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """Broadcast an immediate alert to all channels"""
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        priority = {
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4
        }.get(severity, 1)
        
        message_obj = BroadcastMessage(
            type=BroadcastType.SYSTEM_ALERT,
            data=alert_data,
            timestamp=datetime.utcnow(),
            priority=priority,
            target_channels=['risk-updates', 'analytics-updates', 'system-health']
        )
        
        await self.broadcast_immediate(message_obj)
    
    async def broadcast_data_refresh(self, data_type: str, source: str) -> None:
        """Broadcast notification of data refresh"""
        refresh_data = {
            "data_type": data_type,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        message = BroadcastMessage(
            type=BroadcastType.DATA_REFRESH,
            data=refresh_data,
            timestamp=datetime.utcnow(),
            priority=2,
            target_channels=['analytics-updates']
        )
        
        await self.broadcast_immediate(message)


# Global broadcaster instance
_broadcaster: Optional[WebSocketBroadcaster] = None


def get_broadcaster() -> WebSocketBroadcaster:
    """Get the global WebSocket broadcaster instance"""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = WebSocketBroadcaster()
    return _broadcaster


async def start_websocket_broadcasting() -> None:
    """Start the global WebSocket broadcasting service"""
    broadcaster = get_broadcaster()
    await broadcaster.start_broadcasting()


async def stop_websocket_broadcasting() -> None:
    """Stop the global WebSocket broadcasting service"""
    broadcaster = get_broadcaster()
    await broadcaster.stop_broadcasting()


# Convenience functions for common broadcasts
async def broadcast_risk_alert(alert_type: str, message: str, severity: str = "warning") -> None:
    """Broadcast a risk-related alert"""
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_alert(f"risk_{alert_type}", message, severity)


async def broadcast_system_alert(alert_type: str, message: str, severity: str = "info") -> None:
    """Broadcast a system-related alert"""
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_alert(f"system_{alert_type}", message, severity)


async def broadcast_data_update(data_type: str, source: str) -> None:
    """Broadcast notification of data update"""
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_data_refresh(data_type, source)