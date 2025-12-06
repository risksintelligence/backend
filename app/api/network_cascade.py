"""
Supply Chain Cascade API

Provides snapshot, history, and impact rollups for supply-chain cascades.
Now integrated with World Bank WITS API for real trade flow data.
"""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

from app.api.schemas import (
    CascadeSnapshotResponse,
    CascadeHistoryResponse,
    CascadeImpactsResponse,
)
from app.core.security import require_system_rate_limit
from app.services.worldbank_wits_integration import wb_wits as get_wits_integration
from app.services.geopolitical_intelligence import geopolitical_intelligence
from app.services.maritime_intelligence import maritime_intelligence

router = APIRouter(prefix="/api/v1/network", tags=["network"])


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@router.get("/supply-cascade", response_model=CascadeSnapshotResponse)
async def get_supply_cascade_snapshot(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return real-time cascade snapshot with World Bank WITS trade flow data."""
    as_of = _now_iso()
    
    try:
        # Get real trade data from World Bank WITS
        wits = get_wits_integration()
        nodes, edges = await wits.build_supply_chain_network()
        
        # Get real geopolitical disruption events from free sources
        geopolitical_disruptions = await geopolitical_intelligence.get_supply_chain_disruptions(days=30)
        
        # Get real maritime disruptions from free maritime intelligence
        shipping_delays = await maritime_intelligence.get_shipping_delays()
        maritime_disruptions = []
        for delay in shipping_delays:
            if delay.severity in ["major", "critical"]:
                maritime_disruptions.append({
                    'id': f"maritime_{delay.route_name.lower().replace(' ', '_')}",
                    'severity': delay.severity,
                    'description': f"Shipping delays on {delay.route_name}: {delay.delay_reasons}",
                    'economic_impact_usd': delay.affected_vessels * 100000,  # Rough estimate
                    'affected_ports': [delay.origin_port, delay.destination_port],
                    'affected_routes': [delay.route_name]
                })
        
        # Combine all disruptions into cascade format
        disruptions = []
        
        # Add geopolitical disruptions from free sources
        for disruption in geopolitical_disruptions[:8]:  # Top 8 geopolitical disruptions
            disruptions.append({
                "id": disruption.disruption_id,
                "type": disruption.event_type,
                "severity": disruption.severity,
                "location": list(disruption.location),  # Convert tuple to list
                "description": disruption.description,
                "source": disruption.source,
                "economic_impact_usd": disruption.economic_impact_usd,
                "affected_commodities": disruption.affected_commodities,
                "affected_trade_routes": disruption.affected_trade_routes,
                "estimated_duration_days": disruption.estimated_duration_days,
                "mitigation_strategies": disruption.mitigation_strategies
            })
        
        # Add Free Maritime Intelligence disruptions
        for disruption in maritime_disruptions[:7]:  # Top 7 maritime disruptions
            # Get port location for disruption
            location = [0.0, 0.0]  # Default
            if disruption.get('affected_ports'):
                from app.services.maritime_intelligence import CRITICAL_PORTS
                port_code = disruption['affected_ports'][0]
                if port_code in CRITICAL_PORTS:
                    port_info = CRITICAL_PORTS[port_code]
                    location = [port_info["lat"], port_info["lng"]]
            
            disruptions.append({
                "id": disruption['id'],
                "type": "shipping_delay", 
                "severity": disruption['severity'],
                "location": location,
                "description": disruption['description'],
                "source": "Free Maritime Intelligence",
                "economic_impact_usd": disruption['economic_impact_usd'],
                "affected_ports": disruption['affected_ports'],
                "affected_trade_routes": disruption['affected_routes'],
                "vessels_impacted": 0,  # Not available from free sources
                "mitigation_strategies": ["Diversify shipping routes", "Use alternative ports"]
            })
        
        # Build critical paths from trade flow data
        critical_paths = []
        if len(nodes) >= 2:
            # Create paths based on highest trade volume flows
            high_value_edges = sorted(edges, key=lambda x: x.get("trade_value_usd", 0), reverse=True)
            if high_value_edges:
                # Build simple path from highest volume trade flows
                path = []
                for edge in high_value_edges[:3]:  # Top 3 trade flows
                    if edge["from"] not in path:
                        path.append(edge["from"])
                    if edge["to"] not in path:
                        path.append(edge["to"])
                if len(path) >= 2:
                    critical_paths.append(path)
        
        # Fallback critical path if no trade data available
        if not critical_paths:
            critical_paths = [[node["id"] for node in nodes[:3]]]
        
        logger.info(f"Supply cascade snapshot: {len(nodes)} nodes, {len(edges)} edges, {len(disruptions)} disruptions")
        
        return {
            "as_of": as_of,
            "nodes": nodes,
            "edges": edges,
            "critical_paths": critical_paths,
            "disruptions": disruptions,
        }
        
    except Exception as e:
        logger.error(f"Failed to get real trade data, using fallback: {e}")
        raise HTTPException(status_code=503, detail="Supply cascade snapshot unavailable")


@router.get("/cascade/history", response_model=CascadeHistoryResponse)
async def get_cascade_history(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return cascade history when available; otherwise use cached real data."""
    try:
        # Try to get real cascade history from timeline service
        from app.services.timeline_cascade_service import get_timeline_cascade_service
        timeline_service = get_timeline_cascade_service()
        cascades = await timeline_service.get_cascade_history(time_range_days=365, limit=20)
        
        # Convert to expected format
        series = []
        if cascades:
            # Build time series data from historical cascades
            for cascade in cascades[:10]:  # Top 10 most recent
                for event in cascade.events:
                    series.append({
                        "t": event.timestamp.isoformat(),
                        "v": 1.0,  # Event occurrence
                        "metric": f"{cascade.title.lower().replace(' ', '_')}"
                    })
        
        # Group by metric
        metric_series = {}
        for point in series:
            metric = point["metric"]
            if metric not in metric_series:
                metric_series[metric] = []
            metric_series[metric].append({"t": point["t"], "v": point["v"]})
        
        history_series = [
            {"metric": metric, "points": points}
            for metric, points in metric_series.items()
        ]
        
        logger.info(f"Generated cascade history with {len(history_series)} series from {len(cascades)} historical events")
        
        return {
            "as_of": _now_iso(),
            "series": history_series,
            "metadata": {
                "total_cascades": len(cascades),
                "time_range_days": 365,
                "data_source": "timeline_cascade_service"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get cascade history, attempting cache fallback: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_cascade")
        cached_result, cache_meta = cache.get("cascade_history")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached cascade history as fallback")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.error("No valid cached cascade history available")
            raise HTTPException(status_code=503, detail="Cascade history service unavailable and no cached data")


@router.get("/cascade/impacts", response_model=CascadeImpactsResponse)
async def get_cascade_impacts(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return real-time impact rollups derived from free geopolitical data sources and trade data."""
    
    try:
        import asyncio

        async def get_external_data():
            disruptions = await geopolitical_intelligence.get_supply_chain_disruptions(days=30)
            
            port_statuses = await maritime_intelligence.get_port_congestion()
            shipping_delays = await maritime_intelligence.get_shipping_delays()
            maritime_disruptions = [delay.__dict__ for delay in shipping_delays if delay.severity in ["major", "critical"]]
            
            return disruptions, port_statuses, maritime_disruptions
        
        # Give external APIs only 5 seconds to respond
        disruptions, port_statuses, maritime_disruptions = await asyncio.wait_for(
            get_external_data(), timeout=5.0
        )
        
        # Calculate aggregate financial impacts (combine geopolitical + maritime)
        total_economic_impact = sum(
            d.economic_impact_usd for d in disruptions if d.economic_impact_usd
        ) + sum(
            d.economic_impact_usd for d in maritime_disruptions if d.economic_impact_usd
        )
        
        # Add port congestion impacts
        port_congestion_impact = 0
        port_impacts = {}
        for port in port_statuses:
            if port.congestion_level in ["high", "critical"]:
                # Rough economic impact calculation
                congestion_multiplier = {"high": 0.15, "critical": 0.25}[port.congestion_level]
                daily_port_value = 50_000_000  # Rough estimate per major port
                daily_impact = daily_port_value * congestion_multiplier
                port_congestion_impact += daily_impact
                
                port_impacts[port.port_code] = {
                    "congestion_level": port.congestion_level,
                    "congestion_score": port.congestion_score,
                    "wait_time_hours": port.avg_wait_time_hours,
                    "vessels_affected": port.total_vessels,
                    "daily_impact_usd": daily_impact
                }
        
        total_economic_impact += port_congestion_impact
        
        # Aggregate affected commodities
        commodity_impacts = {}
        for disruption in disruptions:
            for commodity in disruption.affected_commodities:
                if commodity not in commodity_impacts:
                    commodity_impacts[commodity] = {
                        "disruption_count": 0,
                        "total_impact_usd": 0,
                        "severity_score": 0
                    }
                
                commodity_impacts[commodity]["disruption_count"] += 1
                if disruption.economic_impact_usd:
                    commodity_impacts[commodity]["total_impact_usd"] += disruption.economic_impact_usd
                
                # Add severity score
                severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                commodity_impacts[commodity]["severity_score"] += severity_weights.get(disruption.severity, 1)
        
        # Calculate commodity price deltas based on disruptions
        financial_commodities = {}
        for commodity, impact_data in commodity_impacts.items():
            # Estimate price delta based on disruption severity and economic impact
            base_delta = min(5.0, impact_data["severity_score"] * 0.3)  # Max 5% impact
            economic_factor = min(2.0, (impact_data["total_impact_usd"] / 1_000_000_000))  # Scale by billions
            price_delta = base_delta * economic_factor
            
            if price_delta > 0.1:  # Only include significant impacts
                financial_commodities[commodity] = {
                    "delta_pct": round(price_delta, 2),
                    "disruption_count": impact_data["disruption_count"],
                    "impact_usd": impact_data["total_impact_usd"]
                }
        
        # Aggregate policy impacts
        policy_events = [d for d in disruptions if d.event_type in ["policy_change", "civil_unrest"]]
        policy_risk_score = min(1.0, len(policy_events) * 0.1)
        
        # Calculate trade route risks
        route_risks = {}
        for disruption in disruptions:
            for route in disruption.affected_trade_routes:
                if route not in route_risks:
                    route_risks[route] = {"risk_score": 0, "disruption_count": 0}
                
                severity_impact = {"critical": 0.8, "high": 0.6, "medium": 0.4, "low": 0.2}
                route_risks[route]["risk_score"] += severity_impact.get(disruption.severity, 0.2)
                route_risks[route]["disruption_count"] += 1
        
        # Normalize route risks
        for route in route_risks:
            route_risks[route]["risk_score"] = min(1.0, route_risks[route]["risk_score"])
        
        # Calculate industry lead time impacts
        industry_impacts = {}
        
        # Map commodities to industries and calculate lead time increases
        commodity_industry_mapping = {
            "electronics": "tech",
            "semiconductors": "tech", 
            "crude_oil": "energy",
            "natural_gas": "energy",
            "steel": "autos",
            "textiles": "retail",
            "consumer_goods": "retail"
        }
        
        industry_base_lead_times = {"tech": 7.0, "autos": 5.0, "retail": 3.0, "energy": 2.0}
        
        for commodity, impact_data in commodity_impacts.items():
            industry = commodity_industry_mapping.get(commodity)
            if industry and impact_data["disruption_count"] > 0:
                base_lead_time = industry_base_lead_times.get(industry, 4.0)
                disruption_factor = min(2.0, 1 + (impact_data["disruption_count"] * 0.2))
                new_lead_time = base_lead_time * disruption_factor
                
                if industry not in industry_impacts:
                    industry_impacts[industry] = {"lead_time_days": new_lead_time, "disruptions": 0}
                else:
                    industry_impacts[industry]["lead_time_days"] = max(
                        industry_impacts[industry]["lead_time_days"], new_lead_time
                    )
                
                industry_impacts[industry]["disruptions"] += impact_data["disruption_count"]
        
        # Add default values for industries without disruptions
        for industry, base_time in industry_base_lead_times.items():
            if industry not in industry_impacts:
                industry_impacts[industry] = {"lead_time_days": base_time, "disruptions": 0}
        
        logger.info(f"Calculated impacts from {len(disruptions)} geopolitical + {len(maritime_disruptions)} maritime disruptions: ${total_economic_impact:,.0f} total impact")
        
        return {
            "financial": {
                "commodities": financial_commodities,
                "credit_spreads": {
                    "em": {"bp": min(50, int(policy_risk_score * 40))}, 
                    "high_yield": {"bp": min(30, int(policy_risk_score * 25))}
                },
                "total_disruption_impact_usd": total_economic_impact,
                "active_disruptions": len(disruptions),
                "port_congestion_impact_usd": port_congestion_impact,
                "affected_ports": port_impacts
            },
            "policy": {
                "trade_routes": route_risks,
                "overall_policy_risk": round(policy_risk_score, 3),
                "policy_events": len(policy_events),
                "note": f"Analysis based on {len(disruptions)} recent geopolitical events"
            },
            "industry": {
                "lead_time_days": {k: round(v["lead_time_days"], 1) for k, v in industry_impacts.items()},
                "capacity": {
                    "global_supply_chain": max(0.5, 1.0 - (len(disruptions) * 0.05)),  # Reduced capacity with more disruptions
                    "logistics": max(0.6, 1.0 - (policy_risk_score * 0.3))
                },
                "disruption_summary": {k: v["disruptions"] for k, v in industry_impacts.items()}
            },
        }
        
    except Exception as e:
        logger.error(f"Failed to get real impact data from geopolitical sources: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_cascade")
        cached_result, cache_meta = cache.get("cascade_impacts")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached cascade impacts as fallback")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.error("No valid cached cascade impacts available")
            raise HTTPException(status_code=503, detail="Cascade impacts service unavailable and no cached data")
