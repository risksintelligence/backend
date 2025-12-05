"""
Geopolitical Intelligence API Endpoints
Provides access to geopolitical events, conflicts, and supply chain disruptions
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import logging

from app.services.geopolitical_intelligence import GeopoliticalIntelligenceService
from app.core.error_logging import error_logger

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/disruptions")
async def get_geopolitical_disruptions(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back for disruptions")
) -> List[Dict[str, Any]]:
    """
    Get supply chain disruptions based on geopolitical events
    
    Args:
        days: Number of days to look back (1-365)
        
    Returns:
        List of supply chain disruptions with impact analysis
    """
    try:
        service = GeopoliticalIntelligenceService()
        async with service:
            disruptions = await service.get_supply_chain_disruptions(days=days)
            
            # Convert dataclasses to dicts for JSON serialization
            result = []
            for disruption in disruptions:
                # Map SupplyChainDisruption dataclass fields to expected API format
                disruption_dict = {
                    "event_id": getattr(disruption, 'disruption_id', ''),
                    "title": getattr(disruption, 'description', 'Unknown Event')[:100],
                    "location": f"{disruption.location[0]:.3f},{disruption.location[1]:.3f}" if hasattr(disruption, 'location') and disruption.location else 'Unknown',
                    "severity": getattr(disruption, 'severity', 'low'),
                    "impact_score": getattr(disruption, 'economic_impact_usd', 0) or 0,
                    "confidence": 75,  # Default confidence for GDELT data
                    "timestamp": getattr(disruption, 'start_date', ''),
                    "duration_days": getattr(disruption, 'estimated_duration_days', 0),
                    "affected_routes": getattr(disruption, 'affected_trade_routes', []),
                    "affected_commodities": getattr(disruption, 'affected_commodities', []),
                    "source": getattr(disruption, 'source', 'gdelt'),
                    "description": getattr(disruption, 'description', ''),
                    "event_type": getattr(disruption, 'event_type', 'unknown'),
                    "mitigation_strategies": getattr(disruption, 'mitigation_strategies', [])
                }
                
                # Ensure timestamp is string for JSON serialization
                timestamp = disruption_dict.get('timestamp')
                if hasattr(timestamp, 'isoformat'):
                    disruption_dict['timestamp'] = timestamp.isoformat()
                elif timestamp:
                    disruption_dict['timestamp'] = str(timestamp)
                else:
                    disruption_dict['timestamp'] = ''
                    
                result.append(disruption_dict)
            
            logger.info(f"Retrieved {len(result)} geopolitical disruptions for {days} days")
            return result
            
    except Exception as e:
        error_logger.log_error(e, {
            "endpoint": "/api/v1/geopolitical/disruptions",
            "days": days
        })
        logger.error(f"Error getting geopolitical disruptions: {str(e)}")
        
        # Return empty list with success status to prevent frontend crashes
        return []

@router.get("/events/recent")
async def get_recent_geopolitical_events(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of events to return")
) -> List[Dict[str, Any]]:
    """
    Get recent geopolitical events from GDELT
    
    Args:
        limit: Maximum number of events to return
        
    Returns:
        List of recent geopolitical events
    """
    try:
        service = GeopoliticalIntelligenceService()
        async with service:
            events = await service.get_recent_events(limit=limit)
            
            # Ensure proper JSON serialization
            result = []
            for event in events[:limit]:
                if isinstance(event, dict):
                    result.append(event)
                else:
                    result.append({
                        "event_id": str(getattr(event, 'event_id', '')),
                        "title": str(getattr(event, 'title', 'Unknown Event')),
                        "location": str(getattr(event, 'location', 'Unknown')),
                        "date": str(getattr(event, 'date', '')),
                        "confidence": float(getattr(event, 'confidence', 0)),
                    })
            
            logger.info(f"Retrieved {len(result)} recent geopolitical events")
            return result
            
    except Exception as e:
        error_logger.log_error(e, {
            "endpoint": "/api/v1/geopolitical/events/recent",
            "limit": limit
        })
        logger.error(f"Error getting recent events: {str(e)}")
        
        # Return empty list to prevent crashes
        return []

@router.get("/health")
async def geopolitical_health_check() -> Dict[str, Any]:
    """
    Health check for geopolitical intelligence service
    """
    try:
        service = GeopoliticalIntelligenceService()
        async with service:
            # Test basic connectivity
            test_disruptions = await service.get_supply_chain_disruptions(days=1)
            
            return {
                "status": "healthy",
                "service": "geopolitical_intelligence",
                "last_check": "ok",
                "data_sources": ["gdelt", "free_apis"],
                "test_results": {
                    "disruptions_available": len(test_disruptions) > 0 if test_disruptions else False
                }
            }
    except Exception as e:
        logger.warning(f"Geopolitical health check failed: {str(e)}")
        return {
            "status": "degraded",
            "service": "geopolitical_intelligence", 
            "error": str(e),
            "data_sources": ["gdelt", "free_apis"]
        }