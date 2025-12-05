"""
Maritime Intelligence API

Free maritime data from AISHub, NOAA Marine Cadastre, and OpenSeaMap
for supply chain risk analysis and port congestion monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging

from app.core.security import require_system_rate_limit
from app.services.maritime_intelligence import maritime_intelligence

router = APIRouter(prefix="/api/v1/maritime", tags=["maritime"])
logger = logging.getLogger(__name__)

@router.get("/health")
async def get_maritime_health(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get health status of maritime data providers"""
    
    try:
        provider_health = await maritime_intelligence.health_check()
        
        # Calculate overall health score
        healthy_providers = sum(1 for status in provider_health.values() if status)
        total_providers = len(provider_health)
        health_score = (healthy_providers / total_providers) * 100 if total_providers > 0 else 0
        
        return {
            "overall_health": "healthy" if health_score >= 50 else "degraded" if health_score > 0 else "critical",
            "health_score": health_score,
            "providers": provider_health,
            "healthy_providers": healthy_providers,
            "total_providers": total_providers
        }
        
    except Exception as e:
        logger.error(f"Failed to check maritime health: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/ports/congestion")
async def get_port_congestion(
    ports: Optional[str] = Query(None, description="Comma-separated port codes (e.g. SGSIN,CNSHA)"),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get port congestion data for critical ports"""
    
    try:
        port_codes = None
        if ports:
            port_codes = [code.strip().upper() for code in ports.split(",")]
        
        congestion_data = await maritime_intelligence.get_port_congestion(port_codes)
        
        # Format response
        ports_list = []
        for port_code, data in congestion_data.items():
            ports_list.append({
                "port_code": port_code,
                "port_name": data.port_name,
                "congestion_level": data.congestion_level,
                "vessels_at_anchor": data.vessels_at_anchor,
                "vessels_at_berth": data.vessels_at_berth,
                "average_wait_time_hours": data.average_wait_time_hours,
                "source_breakdown": data.source_breakdown,
                "last_updated": data.last_updated.isoformat()
            })
        
        return {
            "ports": ports_list,
            "summary": {
                "total_ports_monitored": len(ports_list),
                "high_congestion_ports": len([p for p in ports_list if p["congestion_level"] in ["high", "severe"]]),
                "average_vessels_at_anchor": sum(p["vessels_at_anchor"] for p in ports_list) / len(ports_list) if ports_list else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get port congestion: {e}")
        raise HTTPException(status_code=500, detail="Port congestion data unavailable")

@router.get("/shipping/delays")
async def get_shipping_delays(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get shipping delays across major trade routes"""
    
    try:
        delays = await maritime_intelligence.get_shipping_delays()
        
        # Format response
        delays_list = []
        for delay in delays:
            delays_list.append({
                "route_name": delay.route_name,
                "origin_port": delay.origin_port,
                "destination_port": delay.destination_port,
                "typical_transit_days": delay.typical_transit_days,
                "current_delay_days": delay.current_delay_days,
                "delay_reasons": delay.delay_reasons,
                "severity": delay.severity,
                "affected_vessels": delay.affected_vessels
            })
        
        return {
            "delays": delays_list,
            "summary": {
                "total_routes_monitored": len(delays_list),
                "routes_with_delays": len([d for d in delays_list if d["current_delay_days"] > 0]),
                "critical_delays": len([d for d in delays_list if d["severity"] in ["major", "critical"]]),
                "average_delay_days": sum(d["current_delay_days"] for d in delays_list) / len(delays_list) if delays_list else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get shipping delays: {e}")
        raise HTTPException(status_code=500, detail="Shipping delays data unavailable")

@router.get("/risk-assessment")
async def get_supply_chain_risk_assessment(
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get comprehensive supply chain risk assessment based on maritime data"""
    
    try:
        risk_assessment = await maritime_intelligence.get_supply_chain_risk_assessment()
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Failed to get risk assessment: {e}")
        raise HTTPException(status_code=500, detail="Risk assessment unavailable")

@router.get("/vessels/near-port")
async def get_vessels_near_port(
    lat: float = Query(..., description="Port latitude"),
    lng: float = Query(..., description="Port longitude"), 
    radius_km: float = Query(50, description="Search radius in kilometers"),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get vessels near a specific port location"""
    
    try:
        from app.services.maritime_intelligence import FreeMaritimeIntelligence
        
        # Create temporary instance for this request
        maritime_service = FreeMaritimeIntelligence()
        vessels = await maritime_service._get_vessels_near_port(lat, lng, radius_km)
        
        # Format response
        vessels_list = []
        for vessel in vessels:
            vessels_list.append({
                "mmsi": vessel.mmsi,
                "vessel_name": vessel.vessel_name,
                "vessel_type": vessel.vessel_type,
                "lat": vessel.lat,
                "lng": vessel.lng,
                "speed": vessel.speed,
                "course": vessel.course,
                "timestamp": vessel.timestamp.isoformat(),
                "source": vessel.source
            })
        
        return {
            "search_params": {
                "lat": lat,
                "lng": lng,
                "radius_km": radius_km
            },
            "vessels": vessels_list,
            "summary": {
                "total_vessels": len(vessels_list),
                "vessels_at_anchor": len([v for v in vessels_list if v["speed"] is not None and v["speed"] < 1]),
                "vessels_moving": len([v for v in vessels_list if v["speed"] is not None and v["speed"] >= 1]),
                "data_sources": list(set([v["source"] for v in vessels_list]))
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get vessels near port: {e}")
        raise HTTPException(status_code=500, detail="Vessel data unavailable")

@router.get("/providers")
async def get_maritime_providers() -> Dict[str, Any]:
    """Get information about maritime data providers"""
    
    from app.services.maritime_intelligence import MARITIME_PROVIDERS
    
    providers_info = []
    for provider_id, config in MARITIME_PROVIDERS.items():
        providers_info.append({
            "id": provider_id,
            "name": config["name"],
            "coverage": config["coverage"],
            "data_types": config["data_types"],
            "rate_limit": config["rate_limit"],
            "requires_auth": config["requires_auth"]
        })
    
    return {
        "providers": providers_info,
        "total_providers": len(providers_info),
        "coverage_areas": list(set([p["coverage"] for p in providers_info])),
        "advantages": [
            "100% free data sources",
            "No API key requirements", 
            "Multiple provider redundancy",
            "Global coverage with US focus",
            "Real-time AIS data",
            "Government-backed reliability (NOAA)"
        ]
    }