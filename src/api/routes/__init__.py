"""API routes package."""
from .health import router as health_router
from .risk import router as risk_router
from .analytics import router as analytics_router
from .network import router as network_router
from .websocket import router as websocket_router
from .prediction import router as prediction_router
from .simulation import router as simulation_router
from .data import router as data_router

__all__ = [
    "health_router",
    "risk_router", 
    "analytics_router",
    "network_router",
    "websocket_router",
    "prediction_router",
    "simulation_router",
    "data_router"
]