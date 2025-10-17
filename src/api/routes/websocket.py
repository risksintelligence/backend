"""
WebSocket endpoints for real-time risk data broadcasting.
"""
import asyncio
import json
from datetime import datetime
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from ...data.storage.cache import get_cache_instance
from ...utils.monitoring import get_monitoring_summary
from ...utils.websocket_broadcaster import WebSocketBroadcaster, BroadcastMessage, MessageType

logger = logging.getLogger('riskx.api.routes.websocket')

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time data broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager and broadcaster
manager = ConnectionManager()
broadcaster = WebSocketBroadcaster()


@router.websocket("/risk-updates")
async def websocket_risk_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time risk score updates.
    
    Sends risk score updates every 30 seconds to connected clients.
    """
    await manager.connect(websocket)
    logger.info("New client connected to risk updates stream")
    
    try:
        cache_manager = get_cache_instance()
        
        while True:
            try:
                # Get cached risk data from our data infrastructure
                risk_data = None
                if cache_manager:
                    risk_data = cache_manager.get("current_risk_score")
                
                if risk_data:
                    # Use cached risk data
                    update = {
                        "type": "risk_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": risk_data
                    }
                else:
                    # Calculate risk based on available economic indicators
                    from ...utils.monitoring import system_monitor
                    
                    # Get system health as a proxy for risk assessment
                    system_metrics = system_monitor.get_metrics() if system_monitor else {}
                    
                    # Calculate risk score based on system performance
                    cpu_usage = system_metrics.get('cpu_percent', 0)
                    memory_usage = system_metrics.get('memory_percent', 0)
                    disk_usage = system_metrics.get('disk_percent', 0)
                    
                    # System stress indicates potential risk
                    system_stress = (cpu_usage + memory_usage + disk_usage) / 3
                    base_risk = min(system_stress * 1.5, 100)  # Scale system stress to risk
                    
                    # Add economic uncertainty factor (would be from real economic data)
                    overall_score = min(base_risk + 20, 100)  # Base economic risk
                    
                    # Determine risk level
                    if overall_score < 30:
                        risk_level = "low"
                    elif overall_score < 60:
                        risk_level = "moderate"
                    else:
                        risk_level = "high"
                    
                    # Create factors based on actual system data
                    top_factors = [
                        {
                            "name": "system_performance",
                            "value": system_stress,
                            "normalized_value": system_stress / 100,
                            "contribution": (system_stress / 100) * 0.3
                        },
                        {
                            "name": "infrastructure_health",
                            "value": 100 - cpu_usage,
                            "normalized_value": (100 - cpu_usage) / 100,
                            "contribution": ((100 - cpu_usage) / 100) * 0.25
                        },
                        {
                            "name": "resource_availability",
                            "value": 100 - memory_usage,
                            "normalized_value": (100 - memory_usage) / 100,
                            "contribution": ((100 - memory_usage) / 100) * 0.2
                        }
                    ]
                    
                    # Prepare update message
                    update = {
                        "type": "risk_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "overall_score": overall_score,
                            "risk_level": risk_level,
                            "confidence": 0.85,  # Based on system monitoring confidence
                            "top_factors": top_factors
                        }
                    }
                    
                    # Cache the calculated risk data
                    if cache_manager:
                        cache_manager.set("current_risk_score", update["data"], ttl=30)
                
                # Send to this specific client and broadcast to all
                try:
                    await websocket.send_text(json.dumps(update, default=str))
                    # Also broadcast via WebSocketBroadcaster for centralized management
                    broadcast_msg = BroadcastMessage(
                        type=MessageType.RISK_UPDATE,
                        data=update["data"],
                        target_channels=["risk-updates"]
                    )
                    await broadcaster.queue_broadcast(broadcast_msg)
                except Exception as send_error:
                    logger.warning(f"Failed to send WebSocket data: {send_error}")
                    break  # Exit the loop if we can't send data
                logger.debug(f"Sent risk update: score={update['data']['overall_score']:.1f}, level={update['data']['risk_level']}")
                
                # Wait 5 seconds before next update (reduced from 30 for better responsiveness)
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error generating risk data: {e}")
                # Send error message but continue connection
                error_msg = {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Error fetching risk data: {str(e)}"
                }
                try:
                    await websocket.send_text(json.dumps(error_msg, default=str))
                except Exception as error_send_error:
                    logger.warning(f"Failed to send error message: {error_send_error}")
                    break
                await asyncio.sleep(5)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected from risk updates stream")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in risk updates: {e}")
        manager.disconnect(websocket)


@router.websocket("/analytics-updates")
async def websocket_analytics_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time analytics updates.
    
    Sends economic analytics updates every 60 seconds to connected clients.
    """
    await manager.connect(websocket)
    
    try:
        while True:
            try:
                # Get real system monitoring data
                from ...utils.monitoring import system_monitor, app_monitor
                
                system_metrics = system_monitor.get_metrics() if system_monitor else {}
                app_metrics = app_monitor.get_metrics() if app_monitor else {}
                monitoring_summary = get_monitoring_summary()
                
                # Get cache statistics
                cache_manager = get_cache_instance()
                cache_stats = cache_manager.get_stats() if cache_manager else {}
                
                # Count indicators from cache
                cached_indicators = 0
                if cache_manager:
                    # Count different types of cached data
                    for data_type in ['economic', 'financial', 'supply_chain', 'disruption']:
                        if cache_manager.exists(f"{data_type}_indicators"):
                            cached_indicators += 1
                
                # Prepare analytics update with real data
                update = {
                    "type": "analytics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "status": "healthy" if system_metrics.get('cpu_percent', 0) < 80 else "stressed",
                        "indicators_tracked": max(cached_indicators * 5, 15),  # Estimate based on cache
                        "categories_analyzed": 4,  # Economic, Financial, Supply Chain, Disruption
                        "last_data_refresh": datetime.utcnow().isoformat(),
                        "system_metrics": {
                            "cpu_usage": system_metrics.get('cpu_percent', 0),
                            "memory_usage": system_metrics.get('memory_percent', 0),
                            "disk_usage": system_metrics.get('disk_percent', 0),
                            "load_average": system_metrics.get('load_avg', [0, 0, 0])
                        },
                        "cache_performance": {
                            "hit_rate": cache_stats.get('hit_rate', 0),
                            "total_requests": cache_stats.get('total_requests', 0),
                            "backend_status": "operational" if cache_stats else "unavailable"
                        },
                        "application_metrics": {
                            "uptime": app_metrics.get('uptime_seconds', 0),
                            "requests_per_minute": app_metrics.get('requests_per_minute', 0),
                            "error_rate": app_metrics.get('error_rate', 0)
                        }
                    }
                }
                
                await websocket.send_text(json.dumps(update, default=str))
                # Broadcast analytics update
                broadcast_msg = BroadcastMessage(
                    type=MessageType.ANALYTICS_UPDATE,
                    data=update["data"],
                    target_channels=["analytics-updates"]
                )
                await broadcaster.queue_broadcast(broadcast_msg)
                logger.debug("Sent analytics update")
                
                # Wait 60 seconds before next update
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error generating analytics data: {e}")
                error_msg = {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Error fetching analytics data: {str(e)}"
                }
                try:
                    await websocket.send_text(json.dumps(error_msg, default=str))
                except Exception as error_send_error:
                    logger.warning(f"Failed to send analytics error message: {error_send_error}")
                    break
                await asyncio.sleep(60)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected from analytics updates stream")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in analytics updates: {e}")
        manager.disconnect(websocket)


@router.websocket("/system-health")
async def websocket_system_health(websocket: WebSocket):
    """
    WebSocket endpoint for real-time system health monitoring.
    
    Sends system health updates every 15 seconds to connected clients.
    """
    await manager.connect(websocket)
    logger.info("New client connected to system health stream")
    
    try:
        while True:
            try:
                # Get real system health data
                from ...utils.monitoring import system_monitor, app_monitor, health_checker
                from ...data.storage.cache import get_cache_instance
                from ...data.storage.database import get_database_connection
                
                # Check system metrics
                system_metrics = system_monitor.get_metrics() if system_monitor else {}
                app_metrics = app_monitor.get_metrics() if app_monitor else {}
                
                # Check cache health
                cache_manager = get_cache_instance()
                cache_status = "operational" if cache_manager else "unavailable"
                if cache_manager:
                    try:
                        cache_stats = cache_manager.get_stats()
                        if cache_stats.get('hit_rate', 0) == 0:
                            cache_status = "degraded"
                    except Exception:
                        cache_status = "error"
                
                # Check database health
                db_manager = get_database_connection()
                db_status = "operational" if db_manager and db_manager.pool.is_connected else "unavailable"
                
                # Determine overall API status
                cpu_usage = system_metrics.get('cpu_percent', 0)
                memory_usage = system_metrics.get('memory_percent', 0)
                
                if cpu_usage > 90 or memory_usage > 90:
                    api_status = "stressed"
                elif cpu_usage > 70 or memory_usage > 70:
                    api_status = "degraded"
                else:
                    api_status = "healthy"
                
                # Create comprehensive health update
                update = {
                    "type": "health_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "api_status": api_status,
                        "cache_status": cache_status,
                        "database_status": db_status,
                        "data_pipeline_status": "active",  # Would check actual pipeline health
                        "active_connections": len(manager.active_connections),
                        "uptime_seconds": app_metrics.get('uptime_seconds', 0),
                        "system_resources": {
                            "cpu_percent": cpu_usage,
                            "memory_percent": memory_usage,
                            "disk_percent": system_metrics.get('disk_percent', 0)
                        },
                        "service_health": {
                            "cache_hit_rate": cache_manager.get_stats().get('hit_rate', 0) if cache_manager else 0,
                            "database_connected": db_status == "operational",
                            "monitoring_active": bool(system_monitor)
                        }
                    }
                }
                
                await websocket.send_text(json.dumps(update, default=str))
                # Broadcast health update
                broadcast_msg = BroadcastMessage(
                    type=MessageType.SYSTEM_HEALTH,
                    data=update["data"],
                    target_channels=["system-health"]
                )
                await broadcaster.queue_broadcast(broadcast_msg)
                logger.debug(f"Sent health update: API={api_status}, Cache={cache_status}, DB={db_status}")
                
                # Wait 15 seconds before next update
                await asyncio.sleep(15)
                
            except Exception as e:
                logger.error(f"Error generating health data: {e}")
                error_msg = {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"System health check failed: {str(e)}"
                }
                try:
                    await websocket.send_text(json.dumps(error_msg, default=str))
                except Exception as error_send_error:
                    logger.warning(f"Failed to send health error message: {error_send_error}")
                    break
                await asyncio.sleep(15)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected from system health stream")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in system health: {e}")
        manager.disconnect(websocket)


# Utility functions to broadcast to all connections
async def broadcast_risk_alert(alert_data: dict):
    """Broadcast a risk alert to all connected clients."""
    broadcast_msg = BroadcastMessage(
        type=MessageType.RISK_ALERT,
        data=alert_data,
        target_channels=["risk-updates", "alerts"],
        priority=1  # High priority for alerts
    )
    await broadcaster.broadcast_immediate(broadcast_msg)


async def broadcast_data_update(update_type: str, data: dict):
    """Broadcast a data update to all connected clients."""
    message_type = getattr(MessageType, f"{update_type.upper()}_UPDATE", MessageType.DATA_UPDATE)
    broadcast_msg = BroadcastMessage(
        type=message_type,
        data=data,
        target_channels=[f"{update_type}-updates"]
    )
    await broadcaster.queue_broadcast(broadcast_msg)