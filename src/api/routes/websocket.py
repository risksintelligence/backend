"""
WebSocket endpoints for real-time risk data broadcasting.
"""
import asyncio
import json
from datetime import datetime
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from src.cache.cache_manager import CacheManager
from src.ml.models.risk_scorer import BasicRiskScorer

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


# Global connection manager
manager = ConnectionManager()


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
        
        # Initialize risk scoring components
        cache_manager = CacheManager()
        risk_scorer = BasicRiskScorer(cache_manager)
        
        while True:
            try:
                # Get current risk score using real FRED data and economic indicators
                risk_score = await risk_scorer.calculate_risk_score(use_cache=True)
                
                if risk_score and risk_score.factors:
                    # Create real-time risk data from actual calculations
                    risk_data = {
                        "overall_score": round(risk_score.overall_score, 2),
                        "risk_level": risk_score.risk_level,
                        "confidence": round(risk_score.confidence, 3),
                        "timestamp": risk_score.timestamp.isoformat(),
                        "methodology_version": risk_score.methodology_version,
                        "factors": [
                            {
                                "name": factor.name,
                                "category": factor.category,
                                "value": round(factor.value, 4),
                                "normalized_value": round(factor.normalized_value, 4),
                                "weight": round(factor.weight, 3),
                                "description": factor.description,
                                "confidence": round(factor.confidence, 3)
                            }
                            for factor in risk_score.factors
                        ]
                    }
                    
                    # Create WebSocket update message
                    update = {
                        "type": "risk_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": risk_data,
                        "source": "fred_economic_data",
                        "update_interval": 30
                    }
                    
                    logger.debug(f"Broadcasting risk update: score={risk_score.overall_score}, level={risk_score.risk_level}")
                    
                else:
                    logger.warning("Risk score calculation returned no data")
                    continue
                
                # Send real-time update to connected client
                try:
                    await websocket.send_text(json.dumps(update, default=str))
                    logger.debug(f"Sent WebSocket update to client: {len(update['data']['factors'])} factors")
                    
                    # Cache the current risk data for other consumers
                    cache_manager.set("websocket_risk_data", risk_data, ttl=60)
                    
                except Exception as send_error:
                    logger.error(f"Failed to send WebSocket update: {send_error}")
                    break
                
                # Wait 30 seconds before next update (FRED data updates every 30 minutes, 
                # but we provide more frequent updates for better user experience)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in risk data generation: {e}")
                await asyncio.sleep(10)  # Shorter wait on error
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