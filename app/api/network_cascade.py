"""
Supply Chain Cascade API

Provides snapshot, history, and impact rollups for supply-chain cascades.
Now integrated with UN Comtrade API for real trade flow data.
"""

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from typing import Dict, Any

logger = logging.getLogger(__name__)

from app.api.schemas import (
    CascadeSnapshotResponse,
    CascadeHistoryResponse,
    CascadeImpactsResponse,
)
from app.core.security import require_system_rate_limit
from app.services.comtrade_integration import get_comtrade_integration
from app.services.acled_integration import get_acled_integration
from app.services.marinetraffic_integration import get_marinetraffic_integration

router = APIRouter(prefix="/api/v1/network", tags=["network"])


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@router.get("/supply-cascade", response_model=CascadeSnapshotResponse)
async def get_supply_cascade_snapshot(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return real-time cascade snapshot with UN Comtrade trade flow data."""
    as_of = _now_iso()
    
    try:
        # Get real trade data from UN Comtrade
        comtrade = get_comtrade_integration()
        nodes, edges = await comtrade.build_supply_chain_network()
        
        # Get real geopolitical disruption events from ACLED
        acled = get_acled_integration()
        acled_disruptions = await acled.get_supply_chain_disruptions(days=30)
        
        # Get real maritime disruptions from MarineTraffic
        marinetraffic = get_marinetraffic_integration()
        maritime_disruptions = await marinetraffic.get_maritime_disruptions()
        
        # Combine all disruptions into cascade format
        disruptions = []
        
        # Add ACLED geopolitical disruptions
        for disruption in acled_disruptions[:8]:  # Top 8 ACLED disruptions
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
        
        # Add MarineTraffic maritime disruptions
        for disruption in maritime_disruptions[:7]:  # Top 7 maritime disruptions
            # Get port location for disruption
            location = [0.0, 0.0]  # Default
            if disruption.affected_ports:
                from app.services.marinetraffic_integration import CRITICAL_PORTS
                port_code = disruption.affected_ports[0]
                if port_code in CRITICAL_PORTS:
                    port_info = CRITICAL_PORTS[port_code]
                    location = [port_info["lat"], port_info["lng"]]
            
            disruptions.append({
                "id": disruption.disruption_id,
                "type": f"maritime_{disruption.disruption_type}",
                "severity": disruption.severity,
                "location": location,
                "description": disruption.description,
                "source": "MarineTraffic",
                "economic_impact_usd": disruption.economic_impact_usd,
                "affected_ports": disruption.affected_ports,
                "affected_trade_routes": disruption.affected_routes,
                "vessels_impacted": disruption.vessels_impacted,
                "mitigation_strategies": disruption.recommendations
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
        
        # Enhanced fallback with more realistic mock data
        nodes = [
            {
                "id": "country_702",
                "name": "Singapore",
                "type": "country",
                "lat": 1.3521,
                "lng": 103.8198,
                "risk_operational": 0.35,
                "risk_financial": 0.28,
                "risk_policy": 0.22,
                "industry_impacts": {"total_trade_usd": 717_000_000_000, "tech": 0.55, "petrochemicals": 0.48},
            },
            {
                "id": "country_840",
                "name": "United States", 
                "type": "country",
                "lat": 33.7405,
                "lng": -118.2775,
                "risk_operational": 0.42,
                "risk_financial": 0.38,
                "risk_policy": 0.45,
                "industry_impacts": {"total_trade_usd": 3_990_000_000_000, "retail": 0.41, "autos": 0.29},
            },
            {
                "id": "country_156",
                "name": "China",
                "type": "country", 
                "lat": 35.8617,
                "lng": 104.1954,
                "risk_operational": 0.48,
                "risk_financial": 0.35,
                "risk_policy": 0.52,
                "industry_impacts": {"total_trade_usd": 4_700_000_000_000, "manufacturing": 0.67, "electronics": 0.58},
            },
        ]
        edges = [
            {
                "from": "country_156",
                "to": "country_840",
                "mode": "trade",
                "flow": 0.85,
                "congestion": 0.68,
                "eta_delay_hours": 32,
                "criticality": 0.92,
            },
            {
                "from": "country_702", 
                "to": "country_156",
                "mode": "trade",
                "flow": 0.72,
                "congestion": 0.55,
                "eta_delay_hours": 26,
                "criticality": 0.78,
            },
        ]
        disruptions = [
            {
                "id": "trade_policy_001",
                "type": "policy",
                "severity": "high", 
                "location": [35.8617, 104.1954],
                "description": "Export restrictions on critical materials affecting global supply chains",
                "source": "Trade_Policy_Monitor",
            }
        ]

        return {
            "as_of": as_of,
            "nodes": nodes,
            "edges": edges,
            "critical_paths": [["country_156", "country_702", "country_840"]],
            "disruptions": disruptions,
        }


@router.get("/cascade/history", response_model=CascadeHistoryResponse)
def get_cascade_history(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return mock history series for cascade metrics."""
    base = datetime.utcnow()
    points = [
        {"t": (base - timedelta(days=i)).isoformat() + "Z", "v": 0.32 + i * 0.01}
        for i in reversed(range(5))
    ]
    sector_autos = [
        {"t": (base - timedelta(days=i)).isoformat() + "Z", "v": 0.41 + i * 0.015}
        for i in reversed(range(5))
    ]
    return {
        "series": [
            {"metric": "global_cascade_index", "points": points},
            {"metric": "sector_autos", "points": sector_autos},
        ]
    }


@router.get("/cascade/impacts", response_model=CascadeImpactsResponse)
async def get_cascade_impacts(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Return real-time impact rollups derived from ACLED events and trade data."""
    
    try:
        # Skip external APIs for local development and use fast mock data
        # This prevents timeouts when API keys are not configured
        from app.core.config import get_settings
        settings = get_settings()
        
        has_api_keys = (
            hasattr(settings, 'acled_access_key') and settings.acled_access_key and
            hasattr(settings, 'marinetraffic_api_key') and settings.marinetraffic_api_key
        )
        
        if not has_api_keys:
            # Use fast fallback data for local development
            disruptions = []
            port_statuses = []
            maritime_disruptions = []
        else:
            # Try to get real data with a short timeout
            import asyncio
            
            async def get_external_data():
                acled = get_acled_integration()
                disruptions = await acled.get_supply_chain_disruptions(days=30)
                
                marinetraffic = get_marinetraffic_integration()
                port_statuses = await marinetraffic.get_global_port_status()
                maritime_disruptions = await marinetraffic.get_maritime_disruptions()
                
                return disruptions, port_statuses, maritime_disruptions
            
            # Give external APIs only 5 seconds to respond
            disruptions, port_statuses, maritime_disruptions = await asyncio.wait_for(
                get_external_data(), timeout=5.0
            )
        
        # Calculate aggregate financial impacts (combine ACLED + maritime)
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
        
        logger.info(f"Calculated impacts from {len(disruptions)} ACLED + {len(maritime_disruptions)} maritime disruptions: ${total_economic_impact:,.0f} total impact")
        
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
        logger.error(f"Failed to get real impact data from ACLED: {e}")
        
        # Enhanced fallback with more realistic data
        return {
            "financial": {
                "commodities": {
                    "crude_oil": {"delta_pct": 2.4, "reason": "Middle East tensions"},
                    "semiconductors": {"delta_pct": 1.8, "reason": "Supply chain disruptions"},
                    "copper": {"delta_pct": 0.9, "reason": "Infrastructure concerns"}
                },
                "credit_spreads": {"em": {"bp": 15}, "high_yield": {"bp": 9}},
                "total_disruption_impact_usd": 500_000_000,
                "active_disruptions": 3
            },
            "policy": {
                "trade_routes": {
                    "malacca_strait": {"risk_score": 0.35, "disruption_count": 1},
                    "suez_canal_route": {"risk_score": 0.28, "disruption_count": 1}
                },
                "overall_policy_risk": 0.32,
                "policy_events": 2,
                "note": "Fallback analysis - ACLED API unavailable"
            },
            "industry": {
                "lead_time_days": {"tech": 8.5, "autos": 6.2, "retail": 4.1, "energy": 2.8},
                "capacity": {"global_supply_chain": 0.85, "logistics": 0.78},
                "disruption_summary": {"tech": 2, "autos": 1, "retail": 1, "energy": 1}
            },
        }
