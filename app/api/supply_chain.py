from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
from app.core.supply_chain_cache import get_supply_chain_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/supply-chain", tags=["supply-chain"])


def _cached_or_unavailable(data_type: str, identifier: str, detail: str):
    """Return cached value when available or raise if nothing real can be served."""
    cache = get_supply_chain_cache()
    cached, _ = cache.get(data_type, identifier)
    if cached:
        return cached
    raise HTTPException(status_code=503, detail=detail)


# Mock functions removed - real-data-first approach implemented


@router.get("/supply-cascade")
async def get_supply_cascade():
    """Get current supply chain cascade snapshot using real data"""

    cache = get_supply_chain_cache()
    cached_snapshot, _ = cache.get("cascade_events", "snapshot")

    try:
        from ..db import SessionLocal
        from ..models import SupplyChainNode, CascadeEvent
        from datetime import datetime, timedelta
        
        db = SessionLocal()
        
        # Get recent cascade events from database
        recent_events = db.query(CascadeEvent).filter(
            CascadeEvent.event_start >= datetime.utcnow() - timedelta(days=30)
        ).limit(20).all()
        
        # Get supply chain nodes summary
        total_nodes = db.query(SupplyChainNode).count()
        high_risk_nodes = db.query(SupplyChainNode).filter(
            SupplyChainNode.overall_risk_score > 0.7
        ).count()
        
        db.close()
        
        # Build real cascade data from database
        cascades = []
        for event in recent_events:
            cascades.append({
                "id": event.id,
                "severity": event.severity,
                "affected_nodes": len(event.affected_nodes or []),
                "cascade_depth": event.cascade_depth or 1,
                "recovery_time_days": event.recovery_time_days,
                "estimated_cost_usd": event.estimated_cost_usd,
                "status": event.status,
                "event_start": event.event_start.isoformat() if event.event_start else None
            })
        
        result = {
            "network_status": "active" if len(recent_events) > 0 else "stable",
            "total_nodes": total_nodes,
            "high_risk_nodes": high_risk_nodes,
            "active_cascades": len([e for e in recent_events if e.status == "active"]),
            "recent_cascades": cascades,
            "risk_score": (high_risk_nodes / max(1, total_nodes)) * 100,
            "last_updated": datetime.utcnow().isoformat()
        }

        cache.set("cascade_events", "snapshot", result, source="db_supply_chain")
        return result
        
    except Exception as e:
        logger.error("Error getting real cascade data", exc_info=e)
        if cached_snapshot:
            return cached_snapshot
        raise HTTPException(status_code=503, detail="Supply cascade data unavailable")


@router.get("/cascade/impacts")
async def get_cascade_impacts():
    """Get supply chain cascade impact analysis using real data"""
    
    return _cached_or_unavailable(
        "cascade_impacts", 
        "current", 
        "Cascade impacts unavailable - real impact analysis pending"
    )


@router.get("/cascade/history")
async def get_cascade_history():
    """Get cascade history using real data"""
    
    return _cached_or_unavailable(
        "cascade_history", 
        "30_days", 
        "Cascade history unavailable - real historical data pending"
    )


@router.get("/resilience-metrics") 
async def get_resilience_metrics():
    """Get resilience metrics using real data"""
    
    return _cached_or_unavailable(
        "resilience_metrics", 
        "current", 
        "Resilience metrics unavailable - real resilience analysis pending"
    )


@router.get("/timeline-cascade")
async def get_timeline_cascade():
    """Get timeline cascade data using real data"""
    
    return _cached_or_unavailable(
        "timeline_cascade", 
        "12_months", 
        "Timeline cascade data unavailable - real timeline analysis pending"
    )


@router.get("/vulnerability-assessment")
async def get_vulnerability_assessment(sector: Optional[str] = Query(None)):
    """Get sector vulnerability assessment using real data"""
    
    identifier = f"sector_{sector}" if sector else "all_sectors"
    return _cached_or_unavailable(
        "vulnerability_assessment", 
        identifier, 
        "Vulnerability assessment unavailable - real analysis pending"
    )