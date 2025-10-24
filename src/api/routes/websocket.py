"""
WebSocket endpoints for real-time data streaming and live updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Set, Any, Optional
import asyncio
import json
import logging
from datetime import datetime
import uuid

from src.core.dependencies import get_cache_manager
from src.cache.cache_manager import IntelligentCacheManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of subscribed topics
        self.topic_connections: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
        
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        connection_id = client_id or str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "subscriptions": set()
        }
        self.subscriptions[connection_id] = set()
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_topics": list(self.get_available_topics())
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            # Unsubscribe from all topics
            await self.unsubscribe_all(connection_id)
            
            # Remove connection
            del self.active_connections[connection_id]
            del self.connection_metadata[connection_id]
            del self.subscriptions[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to a topic."""
        if connection_id not in self.active_connections:
            return False
        
        # Add to subscriptions
        self.subscriptions[connection_id].add(topic)
        self.connection_metadata[connection_id]["subscriptions"].add(topic)
        
        # Add to topic connections
        if topic not in self.topic_connections:
            self.topic_connections[topic] = set()
        self.topic_connections[topic].add(connection_id)
        
        logger.info(f"Connection {connection_id} subscribed to topic: {topic}")
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "subscription_confirmed",
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from a topic."""
        if connection_id not in self.active_connections:
            return False
        
        # Remove from subscriptions
        self.subscriptions[connection_id].discard(topic)
        self.connection_metadata[connection_id]["subscriptions"].discard(topic)
        
        # Remove from topic connections
        if topic in self.topic_connections:
            self.topic_connections[topic].discard(connection_id)
            if not self.topic_connections[topic]:
                del self.topic_connections[topic]
        
        logger.info(f"Connection {connection_id} unsubscribed from topic: {topic}")
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "unsubscription_confirmed", 
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    async def unsubscribe_all(self, connection_id: str):
        """Unsubscribe connection from all topics."""
        if connection_id in self.subscriptions:
            topics = self.subscriptions[connection_id].copy()
            for topic in topics:
                await self.unsubscribe(connection_id, topic)
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all subscribers of a topic."""
        if topic not in self.topic_connections:
            return
        
        message["topic"] = topic
        message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = []
        for connection_id in self.topic_connections[topic].copy():
            success = await self.send_to_connection(connection_id, message)
            if not success:
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            if connection_id in self.topic_connections[topic]:
                self.topic_connections[topic].remove(connection_id)
        
        logger.info(f"Broadcasted to topic {topic}: {len(self.topic_connections[topic])} connections")
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections."""
        message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = []
        for connection_id in list(self.active_connections.keys()):
            success = await self.send_to_connection(connection_id, message)
            if not success:
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
        
        logger.info(f"Broadcasted to all connections: {len(self.active_connections)}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)
    
    def get_topic_subscribers(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self.topic_connections.get(topic, set()))
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict]:
        """Get connection metadata."""
        return self.connection_metadata.get(connection_id)
    
    def get_available_topics(self) -> Set[str]:
        """Get list of available subscription topics."""
        return {
            "risk_alerts",
            "risk_scores", 
            "market_data",
            "economic_indicators",
            "network_status",
            "system_health",
            "cache_metrics",
            "api_metrics",
            "simulation_updates",
            "prediction_updates"
        }


# Global connection manager
connection_manager = ConnectionManager()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = None
):
    """
    Main WebSocket endpoint for real-time connections.
    
    Supports:
    - Real-time risk alerts
    - Live market data streaming 
    - System health updates
    - Cache performance metrics
    """
    connection_id = await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(connection_id, message)
            except json.JSONDecodeError:
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling message from {connection_id}: {e}")
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error", 
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        await connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        await connection_manager.disconnect(connection_id)


async def handle_client_message(connection_id: str, message: Dict[str, Any]):
    """Handle incoming messages from WebSocket clients."""
    message_type = message.get("type")
    
    if message_type == "subscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.subscribe(connection_id, topic)
        else:
            await connection_manager.send_to_connection(connection_id, {
                "type": "error",
                "message": "Topic required for subscription"
            })
    
    elif message_type == "unsubscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.unsubscribe(connection_id, topic)
        else:
            await connection_manager.send_to_connection(connection_id, {
                "type": "error",
                "message": "Topic required for unsubscription"
            })
    
    elif message_type == "ping":
        await connection_manager.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif message_type == "get_status":
        await send_connection_status(connection_id)
    
    elif message_type == "get_topics":
        await connection_manager.send_to_connection(connection_id, {
            "type": "available_topics",
            "topics": list(connection_manager.get_available_topics())
        })
    
    else:
        await connection_manager.send_to_connection(connection_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


async def send_connection_status(connection_id: str):
    """Send connection status to client."""
    info = connection_manager.get_connection_info(connection_id)
    
    status = {
        "type": "connection_status",
        "connection_id": connection_id,
        "connected_at": info.get("connected_at") if info else None,
        "subscriptions": list(info.get("subscriptions", [])) if info else [],
        "total_connections": connection_manager.get_connection_count(),
        "server_time": datetime.utcnow().isoformat()
    }
    
    await connection_manager.send_to_connection(connection_id, status)


# Real-time data streaming functions

async def stream_risk_alerts(alert_data: Dict[str, Any]):
    """Stream risk alerts to subscribers."""
    await connection_manager.broadcast_to_topic("risk_alerts", {
        "type": "risk_alert",
        "alert": alert_data,
        "severity": alert_data.get("severity", "medium"),
        "alert_id": alert_data.get("id", str(uuid.uuid4()))
    })


async def stream_risk_scores(risk_data: Dict[str, Any]):
    """Stream updated risk scores to subscribers."""
    await connection_manager.broadcast_to_topic("risk_scores", {
        "type": "risk_score_update",
        "data": risk_data,
        "overall_score": risk_data.get("overall_score"),
        "trend": risk_data.get("trend", "stable")
    })


async def stream_market_data(market_data: Dict[str, Any]):
    """Stream market data updates to subscribers."""
    await connection_manager.broadcast_to_topic("market_data", {
        "type": "market_update",
        "data": market_data,
        "source": market_data.get("source", "market_feed")
    })


async def stream_economic_indicators(indicator_data: Dict[str, Any]):
    """Stream economic indicator updates to subscribers."""
    await connection_manager.broadcast_to_topic("economic_indicators", {
        "type": "economic_update",
        "data": indicator_data,
        "indicator": indicator_data.get("series_id"),
        "value": indicator_data.get("value")
    })


async def stream_network_status(network_data: Dict[str, Any]):
    """Stream network status updates to subscribers."""
    await connection_manager.broadcast_to_topic("network_status", {
        "type": "network_status_update",
        "data": network_data,
        "health": network_data.get("health", "unknown")
    })


async def stream_system_health(health_data: Dict[str, Any]):
    """Stream system health updates to subscribers."""
    await connection_manager.broadcast_to_topic("system_health", {
        "type": "system_health_update",
        "data": health_data,
        "status": health_data.get("status", "unknown")
    })


async def stream_cache_metrics(cache_data: Dict[str, Any]):
    """Stream cache performance metrics to subscribers."""
    await connection_manager.broadcast_to_topic("cache_metrics", {
        "type": "cache_metrics_update",
        "data": cache_data,
        "hit_rate": cache_data.get("hit_rate_percent", 0)
    })


async def stream_api_metrics(api_data: Dict[str, Any]):
    """Stream API performance metrics to subscribers."""
    await connection_manager.broadcast_to_topic("api_metrics", {
        "type": "api_metrics_update",
        "data": api_data,
        "requests_per_minute": api_data.get("requests_per_minute", 0)
    })


async def stream_simulation_updates(simulation_data: Dict[str, Any]):
    """Stream simulation progress updates to subscribers."""
    await connection_manager.broadcast_to_topic("simulation_updates", {
        "type": "simulation_update",
        "data": simulation_data,
        "simulation_id": simulation_data.get("simulation_id"),
        "progress": simulation_data.get("progress", 0)
    })


async def stream_prediction_updates(prediction_data: Dict[str, Any]):
    """Stream financial prediction updates to subscribers."""
    await connection_manager.broadcast_to_topic("prediction_updates", {
        "type": "prediction_update",
        "data": prediction_data,
        "model": prediction_data.get("model_name"),
        "confidence": prediction_data.get("confidence", 0)
    })


# Administrative endpoints

@router.get("/status")
async def get_websocket_status():
    """Get WebSocket server status and connection statistics."""
    topic_stats = {}
    for topic in connection_manager.get_available_topics():
        topic_stats[topic] = connection_manager.get_topic_subscribers(topic)
    
    return {
        "status": "active",
        "total_connections": connection_manager.get_connection_count(),
        "available_topics": list(connection_manager.get_available_topics()),
        "topic_subscribers": topic_stats,
        "server_time": datetime.utcnow().isoformat()
    }


@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """
    Administrative endpoint to broadcast messages to all connections.
    
    Body:
        - type: Message type
        - data: Message data
        - topic: Optional topic filter
    """
    topic = message.get("topic")
    
    if topic:
        await connection_manager.broadcast_to_topic(topic, message)
        return {
            "status": "success",
            "message": f"Broadcasted to topic: {topic}",
            "subscribers": connection_manager.get_topic_subscribers(topic)
        }
    else:
        await connection_manager.broadcast_to_all(message)
        return {
            "status": "success", 
            "message": "Broadcasted to all connections",
            "total_connections": connection_manager.get_connection_count()
        }


# Background task for periodic updates

class RealTimeDataStreamer:
    """Background service for streaming real-time data updates."""
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache_manager = cache_manager
        self.is_running = False
        
    async def start_streaming(self):
        """Start background streaming tasks."""
        self.is_running = True
        
        # Create multiple streaming tasks
        tasks = [
            asyncio.create_task(self._stream_risk_scores()),
            asyncio.create_task(self._stream_system_health()),
            asyncio.create_task(self._stream_cache_metrics()),
            asyncio.create_task(self._stream_market_data())
        ]
        
        logger.info("Real-time data streaming started")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.is_running = False
    
    async def stop_streaming(self):
        """Stop background streaming."""
        self.is_running = False
        logger.info("Real-time data streaming stopped")
    
    async def _stream_risk_scores(self):
        """Stream risk score updates every 30 seconds."""
        while self.is_running:
            try:
                # Get latest risk data from cache
                risk_data = await self.cache_manager.get("risk:overview")
                
                if risk_data:
                    await stream_risk_scores(risk_data)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error streaming risk scores: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _stream_system_health(self):
        """Stream system health updates every 60 seconds."""
        while self.is_running:
            try:
                # Get system health metrics
                health_data = {
                    "api_status": "healthy",
                    "cache_status": "healthy", 
                    "database_status": "healthy",
                    "websocket_connections": connection_manager.get_connection_count(),
                    "last_updated": datetime.utcnow().isoformat()
                }
                
                await stream_system_health(health_data)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error streaming system health: {e}")
                await asyncio.sleep(120)
    
    async def _stream_cache_metrics(self):
        """Stream cache performance metrics every 45 seconds."""
        while self.is_running:
            try:
                # Get cache metrics
                cache_metrics = self.cache_manager.get_metrics()
                
                await stream_cache_metrics(cache_metrics)
                
                await asyncio.sleep(45)  # Update every 45 seconds
                
            except Exception as e:
                logger.error(f"Error streaming cache metrics: {e}")
                await asyncio.sleep(90)
    
    async def _stream_market_data(self):
        """Stream market data updates every 20 seconds."""
        while self.is_running:
            try:
                # Get market data from cache
                market_data = await self.cache_manager.get("market:overview")
                
                if not market_data:
                    # Get real market data from external APIs
                    from src.data.sources import fred
                    try:
                        real_market_data = await fred.get_market_overview()
                        if real_market_data:
                            market_data = real_market_data
                        else:
                            # No real data available - skip this update
                            logger.warning("No real market data available - skipping WebSocket update")
                            await asyncio.sleep(20)
                            continue
                    except Exception as e:
                        logger.error(f"Failed to get real market data: {e}")
                        await asyncio.sleep(20)
                        continue
                
                await stream_market_data(market_data)
                
                await asyncio.sleep(20)  # Update every 20 seconds
                
            except Exception as e:
                logger.error(f"Error streaming market data: {e}")
                await asyncio.sleep(60)


# Global streaming service
real_time_streamer = None


async def initialize_real_time_streaming(cache_manager: IntelligentCacheManager):
    """Initialize real-time streaming service."""
    global real_time_streamer
    real_time_streamer = RealTimeDataStreamer(cache_manager)
    
    # Start streaming in background
    asyncio.create_task(real_time_streamer.start_streaming())
    
    logger.info("Real-time streaming service initialized")


async def shutdown_real_time_streaming():
    """Shutdown real-time streaming service."""
    global real_time_streamer
    if real_time_streamer:
        await real_time_streamer.stop_streaming()
    
    logger.info("Real-time streaming service shutdown")


# Export functions for use in main application
__all__ = [
    'router',
    'connection_manager', 
    'stream_risk_alerts',
    'stream_risk_scores',
    'stream_market_data',
    'stream_economic_indicators',
    'stream_network_status',
    'stream_system_health',
    'initialize_real_time_streaming',
    'shutdown_real_time_streaming'
]