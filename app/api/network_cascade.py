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
from app.services.worldbank_wits_integration import get_wits_integration
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
        from app.services.maritime_intelligence import CRITICAL_PORTS
        for disruption in maritime_disruptions[:7]:  # Top 7 maritime disruptions
            # Get port location for disruption
            location = [0.0, 0.0]  # Default
            if disruption.get('affected_ports'):
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
        
        result = {
            "as_of": as_of,
            "summary": {
                "total_nodes": len(nodes),
                "total_disruptions": len(disruptions),
                "high_risk_nodes": len([n for n in nodes if n.get("risk_score", 0) > 0.7]),
                "cascade_probability": round(len(disruptions) / max(len(nodes), 1) * 0.3, 2)
            },
            "top_disruptions": disruptions,
            "network_overview": {
                "nodes": nodes,
                "edges": edges
            },
            "critical_paths": critical_paths,
            "data_freshness": {
                "trade_data": "Live API data",
                "geopolitical_data": "Live API data", 
                "maritime_data": "Live API data",
                "cache_status": "Fresh data from APIs"
            }
        }
        
        # Cache successful result for future use
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_cascade")
        cache.set("supply_cascade_snapshot", result, source="supply_chain_apis", 
                  source_url="world_bank_wits+geopolitical+maritime", soft_ttl=1800, hard_ttl=7200)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get real trade data, using cached fallback: {e}")
        
        # Use cached data instead of failing
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_cascade")
        
        cached_data, metadata = cache.get("supply_cascade_snapshot")
        if cached_data:
            logger.info("Returning cached supply cascade data due to API rate limits/issues")
            return cached_data
        
        # If no cache, return minimal fallback data so frontend doesn't break
        logger.warning("No cached data available, returning minimal fallback")
        return {
            "as_of": as_of,
            "summary": {
                "total_nodes": 8,
                "total_disruptions": 2,
                "high_risk_nodes": 1,
                "cascade_probability": 0.25
            },
            "top_disruptions": [
                {
                    "id": "api_rate_limit_fallback",
                    "type": "system_notice", 
                    "severity": "info",
                    "location": ["Global"],
                    "description": "External APIs temporarily rate limited - using cached/fallback data",
                    "source": "Cache System",
                    "economic_impact_usd": 0,
                    "affected_commodities": [],
                    "affected_trade_routes": [],
                    "estimated_duration_days": 1,
                    "mitigation_strategies": ["API rate limits will reset automatically"]
                }
            ],
            "network_overview": {
                "nodes": [
                    {"id": "node_usa", "country": "USA", "risk_score": 0.2, "type": "major_hub"},
                    {"id": "node_chn", "country": "China", "risk_score": 0.3, "type": "major_hub"},
                    {"id": "node_deu", "country": "Germany", "risk_score": 0.15, "type": "regional_hub"}
                ],
                "edges": [
                    {"from": "node_usa", "to": "node_chn", "weight": 0.9, "risk": 0.2},
                    {"from": "node_chn", "to": "node_deu", "weight": 0.7, "risk": 0.15}
                ]
            },
            "data_freshness": {
                "trade_data": "Cached/Fallback mode", 
                "geopolitical_data": "Cached/Fallback mode",
                "maritime_data": "Cached/Fallback mode",
                "cache_status": "Using fallback data due to API limits"
            }
        }


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
            logger.warning("No valid cached cascade history available, returning minimal fallback")
            # Return minimal fallback data so frontend doesn't break
            return {
                "as_of": _now_iso(),
                "series": [
                    {
                        "metric": "supply_disruptions",
                        "points": [
                            {"t": (_now_dt() - timedelta(days=30)).isoformat(), "v": 0.2},
                            {"t": (_now_dt() - timedelta(days=15)).isoformat(), "v": 0.4},
                            {"t": _now_iso(), "v": 0.3}
                        ]
                    },
                    {
                        "metric": "trade_flow_volatility",
                        "points": [
                            {"t": (_now_dt() - timedelta(days=30)).isoformat(), "v": 0.1},
                            {"t": (_now_dt() - timedelta(days=15)).isoformat(), "v": 0.25},
                            {"t": _now_iso(), "v": 0.15}
                        ]
                    }
                ],
                "metadata": {
                    "total_cascades": 2,
                    "time_range_days": 30,
                    "data_source": "fallback_synthetic",
                    "fallback_reason": "Service and cache unavailable"
                },
                "fallback_data": True
            }


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
            try:
                # Check if port is an object with the expected attributes
                if hasattr(port, 'congestion_level') and hasattr(port, 'port_code'):
                    if port.congestion_level in ["high", "critical"]:
                        # Rough economic impact calculation
                        congestion_multiplier = {"high": 0.15, "critical": 0.25}[port.congestion_level]
                        daily_port_value = 50_000_000  # Rough estimate per major port
                        daily_impact = daily_port_value * congestion_multiplier
                        port_congestion_impact += daily_impact
                        
                        port_impacts[port.port_code] = {
                            "congestion_level": port.congestion_level,
                            "congestion_score": getattr(port, 'congestion_score', 0.5),
                            "wait_time_hours": getattr(port, 'avg_wait_time_hours', 24),
                            "vessels_affected": getattr(port, 'total_vessels', 50),
                            "daily_impact_usd": daily_impact
                        }
            except (AttributeError, TypeError) as e:
                # Skip invalid port data
                logger.warning(f"Skipping invalid port data: {e}")
                continue
        
        total_economic_impact += port_congestion_impact
        
        # Aggregate affected commodities
        commodity_impacts = {}
        for disruption in disruptions:
            try:
                # Check if disruption has the expected attributes
                affected_commodities = getattr(disruption, 'affected_commodities', [])
                if not affected_commodities:
                    continue
                    
                for commodity in affected_commodities:
                    if commodity not in commodity_impacts:
                        commodity_impacts[commodity] = {
                            "disruption_count": 0,
                            "total_impact_usd": 0,
                            "severity_score": 0
                        }
                    
                    commodity_impacts[commodity]["disruption_count"] += 1
                    economic_impact = getattr(disruption, 'economic_impact_usd', None)
                    if economic_impact:
                        commodity_impacts[commodity]["total_impact_usd"] += economic_impact
                    
                    # Add severity score
                    severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                    severity = getattr(disruption, 'severity', 'medium')
                    commodity_impacts[commodity]["severity_score"] += severity_weights.get(severity, 1)
            except (AttributeError, TypeError) as e:
                # Skip invalid disruption data
                logger.warning(f"Skipping invalid disruption data: {e}")
                continue
        
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
        policy_events = [d for d in disruptions if hasattr(d, 'event_type') and getattr(d, 'event_type', '') in ["policy_change", "civil_unrest"]]
        policy_risk_score = min(1.0, len(policy_events) * 0.1)
        
        # Calculate trade route risks
        route_risks = {}
        for disruption in disruptions:
            try:
                affected_routes = getattr(disruption, 'affected_trade_routes', [])
                for route in affected_routes:
                    if route not in route_risks:
                        route_risks[route] = {"risk_score": 0, "disruption_count": 0}
                    
                    severity_impact = {"critical": 0.8, "high": 0.6, "medium": 0.4, "low": 0.2}
                    severity = getattr(disruption, 'severity', 'medium')
                    route_risks[route]["risk_score"] += severity_impact.get(severity, 0.2)
                    route_risks[route]["disruption_count"] += 1
            except (AttributeError, TypeError) as e:
                # Skip invalid disruption data
                logger.warning(f"Skipping invalid disruption for route risks: {e}")
                continue
        
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
        
        # Cache successful data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_cascade")
        
        # Return new consistent structure that matches frontend expectations
        result = {
            "as_of": _now_iso(),
            "total_economic_impact": total_economic_impact,
            "commodity_impacts": commodity_impacts,
            "financial_commodities": financial_commodities,
            "port_impacts": port_impacts,
            "summary": {
                "total_disruptions": len(disruptions) + len(maritime_disruptions),
                "critical_disruptions": len([d for d in disruptions if getattr(d, 'severity', 'medium') in ['critical', 'high']]),
                "affected_trade_routes": len(route_risks),
                "avg_severity": "medium" if len(disruptions) > 0 else "low"
            },
            "policy_analysis": {
                "trade_routes": route_risks,
                "overall_policy_risk": round(policy_risk_score, 3),
                "policy_events": len(policy_events)
            },
            "industry_analysis": {
                "lead_time_days": {k: round(v["lead_time_days"], 1) for k, v in industry_impacts.items()},
                "capacity_utilization": max(0.5, 1.0 - (len(disruptions) * 0.05)),
                "disruption_summary": {k: v["disruptions"] for k, v in industry_impacts.items()}
            },
            "metadata": {
                "data_sources": ["geopolitical_intelligence", "maritime_intelligence"],
                "coverage": "global",
                "analysis_type": "real_time"
            },
            "last_updated": _now_iso()
        }
        
        # Cache the result for future fallback
        cache.set("cascade_impacts", result, "geopolitical_maritime_intelligence", 
                 derivation_flag="analyzed", soft_ttl=1800, hard_ttl=7200)
        
        return result
        
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
            logger.warning("No valid cached cascade impacts available, returning minimal fallback")
            # Return minimal fallback data so frontend doesn't break
            return {
                "as_of": _now_iso(),
                "total_economic_impact": 2_500_000_000,  # $2.5B fallback
                "commodity_impacts": {
                    "crude_oil": {"delta_pct": 1.2, "disruption_count": 3, "total_impact_usd": 800_000_000, "severity_score": 8},
                    "wheat": {"delta_pct": 0.8, "disruption_count": 2, "total_impact_usd": 300_000_000, "severity_score": 4},
                    "semiconductors": {"delta_pct": 2.1, "disruption_count": 1, "total_impact_usd": 1_200_000_000, "severity_score": 4}
                },
                "financial_commodities": {
                    "WTI_crude": {"price_usd": 78.50, "delta_pct": 1.2, "disruption_count": 3, "impact_usd": 800_000_000},
                    "wheat_futures": {"price_usd": 6.80, "delta_pct": 0.8, "disruption_count": 2, "impact_usd": 300_000_000},
                    "copper": {"price_usd": 8750.0, "delta_pct": 0.5, "disruption_count": 1, "impact_usd": 200_000_000}
                },
                "port_impacts": {
                    "USLAX": {
                        "congestion_level": "moderate",
                        "congestion_score": 0.6,
                        "wait_time_hours": 24,
                        "vessels_affected": 85,
                        "daily_impact_usd": 15_000_000
                    },
                    "USNYC": {
                        "congestion_level": "high", 
                        "congestion_score": 0.8,
                        "wait_time_hours": 36,
                        "vessels_affected": 120,
                        "daily_impact_usd": 25_000_000
                    }
                },
                "summary": {
                    "total_disruptions": 6,
                    "critical_disruptions": 1,
                    "affected_trade_routes": 12,
                    "avg_severity": "medium"
                },
                "policy_analysis": {
                    "trade_routes": {
                        "asia_pacific_us": {"risk_score": 0.6, "disruption_count": 2},
                        "europe_asia": {"risk_score": 0.4, "disruption_count": 1},
                        "middle_east_global": {"risk_score": 0.8, "disruption_count": 3}
                    },
                    "overall_policy_risk": 0.65,
                    "policy_events": 3
                },
                "industry_analysis": {
                    "lead_time_days": {
                        "tech": 45.5,
                        "autos": 62.3, 
                        "energy": 38.7,
                        "retail": 28.9
                    },
                    "capacity_utilization": 0.75,
                    "disruption_summary": {
                        "tech": 2,
                        "autos": 1,
                        "energy": 2,
                        "retail": 1
                    }
                },
                "metadata": {
                    "data_sources": ["fallback_synthetic"],
                    "coverage": "global",
                    "fallback_reason": "Service and cache unavailable",
                    "analysis_type": "synthetic"
                },
                "last_updated": _now_iso(),
                "fallback_data": True
            }


@router.get("/vulnerability-assessment/{sector}")
async def get_network_vulnerability_assessment(
    sector: str,
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get sector vulnerability assessment for network analysis."""
    
    try:
        # Try to get real vulnerability data from existing supply chain endpoint  
        from app.api.supply_chain import get_vulnerability_assessment as supply_chain_vuln
        vulnerability_data = await supply_chain_vuln(sector)
        
        if vulnerability_data and not isinstance(vulnerability_data, dict) or "error" not in vulnerability_data:
            # Cache successful data
            from app.core.unified_cache import UnifiedCache
            cache = UnifiedCache("network_vulnerability")
            cache.set(f"sector_{sector}", vulnerability_data, "supply_chain_analyzer", 
                     derivation_flag="analyzed", soft_ttl=1800, hard_ttl=7200)
            return vulnerability_data
        
        raise Exception("Supply chain vulnerability endpoint returned error")
        
    except Exception as e:
        logger.error(f"Failed to get vulnerability assessment for {sector}: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_vulnerability")
        cached_result, cache_meta = cache.get(f"sector_{sector}")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info(f"Using cached vulnerability data for {sector}")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.warning(f"No valid cached vulnerability data for {sector}, returning fallback")
            # Return sector-specific fallback data
            return {
                "sector_name": sector,
                "overall_score": 65.0 + hash(sector) % 20,  # Deterministic but varied by sector
                "risk_level": "medium",
                "vulnerability_metrics": [
                    {
                        "metric_name": "supply_chain_concentration",
                        "current_value": 0.7 + (hash(sector + "conc") % 100) / 300,
                        "threshold": 0.8,
                        "risk_level": "medium",
                        "impact_description": f"High concentration risk in {sector} supply chains"
                    },
                    {
                        "metric_name": "geopolitical_exposure",
                        "current_value": 0.4 + (hash(sector + "geo") % 100) / 250,
                        "threshold": 0.6,
                        "risk_level": "low",
                        "impact_description": f"Moderate geopolitical risk exposure for {sector}"
                    },
                    {
                        "metric_name": "cyber_vulnerability",
                        "current_value": 0.5 + (hash(sector + "cyber") % 100) / 200,
                        "threshold": 0.7,
                        "risk_level": "medium",
                        "impact_description": f"Cybersecurity risks present in {sector} infrastructure"
                    },
                    {
                        "metric_name": "financial_resilience",
                        "current_value": 0.6 + (hash(sector + "fin") % 100) / 166,
                        "threshold": 0.5,
                        "risk_level": "good",
                        "impact_description": f"Financial resilience indicators for {sector} sector"
                    }
                ],
                "recommendations": [
                    f"Diversify {sector} supply chain sources across multiple regions",
                    f"Implement enhanced cybersecurity measures for {sector} infrastructure",
                    f"Develop contingency plans for {sector}-specific disruptions",
                    f"Monitor key {sector} suppliers for financial stability"
                ],
                "last_updated": _now_iso(),
                "metadata": {
                    "data_source": "fallback_synthetic",
                    "coverage": sector,
                    "fallback_reason": "Vulnerability service and cache unavailable"
                },
                "fallback_data": True
            }


@router.get("/vulnerability-assessment")  
async def get_network_vulnerability_assessment_all(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get overall network vulnerability assessment."""
    
    try:
        # Try to get real vulnerability data from supply chain endpoint
        from app.api.supply_chain import get_vulnerability_assessment as supply_chain_vuln
        vulnerability_data = await supply_chain_vuln(None)
        
        if vulnerability_data and not isinstance(vulnerability_data, dict) or "error" not in vulnerability_data:
            return vulnerability_data
        
        raise Exception("Supply chain vulnerability endpoint returned error")
        
    except Exception as e:
        logger.error(f"Failed to get overall vulnerability assessment: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_vulnerability")
        cached_result, cache_meta = cache.get("all_sectors")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached overall vulnerability data")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.warning("No valid cached overall vulnerability data, returning fallback")
            # Return aggregated fallback data
            return {
                "sector_name": "All Sectors",
                "overall_score": 68.5,
                "risk_level": "medium",
                "vulnerability_metrics": [
                    {
                        "metric_name": "global_supply_concentration",
                        "current_value": 0.72,
                        "threshold": 0.8,
                        "risk_level": "medium",
                        "impact_description": "High concentration in global supply chains"
                    },
                    {
                        "metric_name": "cross_sector_dependencies",
                        "current_value": 0.65,
                        "threshold": 0.7,
                        "risk_level": "medium", 
                        "impact_description": "Moderate cross-sector interdependencies"
                    },
                    {
                        "metric_name": "systemic_risk_indicators",
                        "current_value": 0.58,
                        "threshold": 0.6,
                        "risk_level": "low",
                        "impact_description": "Systemic risks within acceptable bounds"
                    }
                ],
                "recommendations": [
                    "Enhance cross-sector resilience coordination",
                    "Develop sector-agnostic risk monitoring systems",
                    "Strengthen critical infrastructure dependencies",
                    "Implement systematic early warning mechanisms"
                ],
                "last_updated": _now_iso(),
                "metadata": {
                    "data_source": "fallback_synthetic",
                    "coverage": "global",
                    "fallback_reason": "Vulnerability service and cache unavailable"
                },
                "fallback_data": True
            }


@router.get("/resilience-metrics")
async def get_network_resilience_metrics(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get network resilience metrics."""
    
    try:
        # Try to get real resilience data from supply chain endpoint
        from app.api.supply_chain import get_resilience_metrics as supply_chain_resilience
        resilience_data = await supply_chain_resilience()
        
        if resilience_data and not isinstance(resilience_data, dict) or "error" not in resilience_data:
            return resilience_data
        
        raise Exception("Supply chain resilience endpoint returned error")
        
    except Exception as e:
        logger.error(f"Failed to get resilience metrics: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_resilience")
        cached_result, cache_meta = cache.get("resilience_metrics")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached resilience metrics")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.warning("No valid cached resilience metrics, returning fallback")
            # Return resilience metrics fallback data
            return {
                "resilience_metrics": {
                    "overall_score": 74.5,
                    "sector_scores": {
                        "technology": 78.2,
                        "manufacturing": 71.8,
                        "energy": 76.5,
                        "finance": 82.1,
                        "healthcare": 69.3,
                        "transportation": 73.7,
                        "agriculture": 70.2,
                        "telecommunications": 79.8,
                        "retail": 68.9,
                        "automotive": 75.4
                    },
                    "redundancy_score": 67.0,
                    "adaptability_score": 81.9,
                    "avg_recovery_time": 96,
                    "critical_nodes_count": 7,
                    "recommendations": [
                        "Enhance redundancy in manufacturing supply chains",
                        "Improve healthcare sector resilience planning",
                        "Strengthen retail distribution network backup systems",
                        "Develop cross-sector cooperation frameworks",
                        "Implement advanced early warning systems"
                    ]
                },
                "last_updated": _now_iso(),
                "metadata": {
                    "data_source": "fallback_synthetic",
                    "coverage": "global",
                    "fallback_reason": "Resilience service and cache unavailable"
                },
                "fallback_data": True
            }


@router.get("/timeline-cascade")
async def get_network_timeline_cascade(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get timeline cascade data."""
    
    try:
        # Try to get real timeline cascade data
        from app.api.supply_chain import get_timeline_cascade as supply_chain_timeline
        timeline_data = await supply_chain_timeline()
        
        if timeline_data and not isinstance(timeline_data, dict) or "error" not in timeline_data:
            return timeline_data
        
        raise Exception("Supply chain timeline endpoint returned error")
        
    except Exception as e:
        logger.error(f"Failed to get timeline cascade: {e}")
        
        # Try cached data
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("network_timeline")
        cached_result, cache_meta = cache.get("timeline_cascade")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached timeline cascade")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            logger.warning("No valid cached timeline cascade, returning fallback")
            # Return timeline cascade fallback data
            
            from datetime import timedelta
            
            # Generate synthetic timeline events
            events = []
            base_date = datetime.utcnow() - timedelta(days=90)
            
            event_types = [
                ("supply_disruption", "Supply Chain Disruption", "high"),
                ("geopolitical_event", "Geopolitical Event", "medium"), 
                ("natural_disaster", "Natural Disaster", "critical"),
                ("cyber_incident", "Cyber Security Incident", "high"),
                ("trade_dispute", "Trade Dispute", "medium"),
                ("port_congestion", "Port Congestion", "low"),
                ("energy_crisis", "Energy Crisis", "high"),
                ("financial_shock", "Financial Market Shock", "medium")
            ]
            
            for i in range(15):
                event_date = base_date + timedelta(days=i*6 + hash(f"event_{i}") % 5)
                event_type, title, severity = event_types[i % len(event_types)]
                
                events.append({
                    "timestamp": event_date.isoformat(),
                    "title": f"{title} {i+1}",
                    "description": f"Simulated {event_type} affecting global supply chains",
                    "severity": severity,
                    "affected_sectors": ["manufacturing", "technology", "energy"][:(i % 3) + 1],
                    "impact_score": 0.3 + (hash(f"impact_{i}") % 50) / 100,
                    "duration_hours": 24 + (hash(f"duration_{i}") % 120),
                    "event_type": event_type
                })
            
            return {
                "timeline_cascade": {
                    "events": events,
                    "total_events": len(events),
                    "time_range_days": 90,
                    "avg_impact_score": 0.65,
                    "most_affected_sector": "manufacturing",
                    "recovery_metrics": {
                        "avg_recovery_time": 72,
                        "successful_recoveries": 12,
                        "ongoing_incidents": 3
                    }
                },
                "last_updated": _now_iso(),
                "metadata": {
                    "data_source": "fallback_synthetic",
                    "coverage": "90_days",
                    "fallback_reason": "Timeline service and cache unavailable"
                },
                "fallback_data": True
            }
