"""
Free Maritime Intelligence Integration for Supply Chain Risk Analysis

Combines three free data sources:
- AISHub: Free AIS data exchange network
- OpenSeaMap: Open source marine navigation data  
- NOAA Marine Cadastre: US government marine data

Provides real-time port congestion, vessel tracking, and shipping delays
for supply chain risk analysis and cascade modeling.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import httpx
import json
from functools import lru_cache
import xml.etree.ElementTree as ET
import sentry_sdk
from opentelemetry import trace

from app.core.unified_cache import UnifiedCache
from app.core.config import get_settings
from app.core.error_logging import error_logger

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)

# Free Maritime Data Providers
MARITIME_PROVIDERS = {
    "aishub": {
        "name": "AISHub",
        "base_url": "https://www.aishub.net/api",
        "endpoints": {
            "vessels": "/vesselsdata/v1/index.php",
            "tracks": "/tracks/v1/index.php"
        },
        "rate_limit": 1000,  # requests per day
        "requires_auth": False,
        "coverage": "global",
        "data_types": ["vessel_positions", "vessel_tracks"]
    },
    "noaa_marine": {
        "name": "NOAA Marine Cadastre", 
        "base_url": "https://marinecadastre.gov/api",
        "endpoints": {
            "ais": "/ais/v1/data",
            "ports": "/ports/v1/data",
            "anchorages": "/anchorages/v1/data"
        },
        "rate_limit": 10000,  # generous government API
        "requires_auth": False,
        "coverage": "us_waters",
        "data_types": ["us_port_data", "vessel_density", "anchorage_usage"]
    },
    "openseamap": {
        "name": "OpenSeaMap",
        "base_url": "https://api.openseamap.org",
        "endpoints": {
            "harbours": "/harbours",
            "buoys": "/buoys", 
            "seamarksdata": "/seamarksdata"
        },
        "rate_limit": 5000,
        "requires_auth": False,
        "coverage": "global",
        "data_types": ["port_infrastructure", "navigation_aids", "maritime_boundaries"]
    }
}

MARITIME_CACHE_TTL = 3600 * 2  # 2 hours cache

# Critical global ports for supply chain monitoring (same as before)
CRITICAL_PORTS = {
    "SGSIN": {  # Singapore
        "name": "Port of Singapore",
        "country": "Singapore",
        "lat": 1.2966, "lng": 103.8547,
        "annual_teu": 37200000,
        "strategic_importance": 0.95,
        "primary_trades": ["trans_pacific", "asia_europe", "intra_asia"]
    },
    "CNSHA": {  # Shanghai
        "name": "Port of Shanghai",
        "country": "China", 
        "lat": 31.2304, "lng": 121.4737,
        "annual_teu": 47030000,
        "strategic_importance": 0.98,
        "primary_trades": ["trans_pacific", "asia_europe", "china_domestic"]
    },
    "NLRTM": {  # Rotterdam
        "name": "Port of Rotterdam",
        "country": "Netherlands",
        "lat": 51.9225, "lng": 4.4792,
        "annual_teu": 15290000,
        "strategic_importance": 0.92,
        "primary_trades": ["europe_asia", "transatlantic", "europe_africa"]
    },
    "USNYC": {  # New York/New Jersey
        "name": "Port of New York/New Jersey", 
        "country": "United States",
        "lat": 40.6700, "lng": -74.0401,
        "annual_teu": 8900000,
        "strategic_importance": 0.88,
        "primary_trades": ["transatlantic", "trans_pacific", "caribbean"]
    },
    "USLAX": {  # Los Angeles
        "name": "Port of Los Angeles",
        "country": "United States", 
        "lat": 33.7353, "lng": -118.2644,
        "annual_teu": 10700000,
        "strategic_importance": 0.90,
        "primary_trades": ["trans_pacific", "asia_us_west_coast"]
    },
    "AEDXB": {  # Dubai
        "name": "Port of Dubai",
        "country": "UAE",
        "lat": 25.2697, "lng": 55.3094,
        "annual_teu": 15300000,
        "strategic_importance": 0.85,
        "primary_trades": ["middle_east_asia", "europe_middle_east", "africa_asia"]
    },
    "DEHAM": {  # Hamburg
        "name": "Port of Hamburg",
        "country": "Germany",
        "lat": 53.5453, "lng": 9.9068,
        "annual_teu": 8500000,
        "strategic_importance": 0.82,
        "primary_trades": ["europe_asia", "baltic_trade", "north_sea"]
    }
}

@dataclass
class VesselInfo:
    """Unified vessel information from multiple sources"""
    mmsi: int
    vessel_name: str
    vessel_type: str
    lat: float
    lng: float
    speed: Optional[float]
    course: Optional[float]
    timestamp: datetime
    source: str  # which provider gave us this data
    port_destination: Optional[str] = None
    eta: Optional[datetime] = None
    cargo_type: Optional[str] = None

@dataclass
class PortCongestion:
    """Port congestion metrics from multiple sources"""
    port_code: str
    port_name: str
    vessels_at_anchor: int
    vessels_at_berth: int
    average_wait_time_hours: Optional[float]
    congestion_level: str  # "low", "medium", "high", "severe"
    last_updated: datetime
    source_breakdown: Dict[str, int]  # which sources contributed data

@dataclass
class ShippingDelay:
    """Shipping delay information"""
    route_name: str
    origin_port: str
    destination_port: str
    typical_transit_days: int
    current_delay_days: int
    delay_reasons: List[str]
    severity: str  # "minor", "moderate", "major", "critical"
    affected_vessels: int

class FreeMaritimeIntelligence:
    """Unified maritime intelligence using free data sources"""
    
    def __init__(self):
        self.cache = UnifiedCache("maritime_intel")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.provider_health = {provider: True for provider in MARITIME_PROVIDERS.keys()}
    
    async def get_port_congestion(self, port_codes: Optional[List[str]] = None) -> Dict[str, PortCongestion]:
        """Get port congestion data from multiple free sources"""
        
        with tracer.start_as_current_span("maritime_port_congestion") as span:
            span.set_attribute("service.name", "maritime_intelligence")
            span.set_attribute("api.operation", "get_port_congestion")
            
            try:
                if port_codes is None:
                    port_codes = list(CRITICAL_PORTS.keys())
                
                span.set_attribute("ports.requested", len(port_codes))
                congestion_data = {}
                
                for port_code in port_codes:
                    cache_key = f"port_congestion_{port_code}"
                    cached_data, metadata = self.cache.get(cache_key)
                    
                    if cached_data and metadata and not metadata.is_stale_soft:
                        congestion_data[port_code] = PortCongestion(**cached_data)
                        continue
                    
                    # Gather data from multiple sources
                    port_info = CRITICAL_PORTS.get(port_code)
                    if not port_info:
                        continue
                    
                    # Combine data from all available sources
                    vessels_data = await self._get_vessels_near_port(port_info['lat'], port_info['lng'])
                    congestion = self._calculate_congestion(port_code, port_info, vessels_data)
                    
                    congestion_data[port_code] = congestion
                    
                    # Cache for 30 minutes
                    self.cache.set(cache_key, congestion.__dict__, source="maritime_intelligence", soft_ttl=1800)
                
                span.set_attribute("ports.analyzed", len(congestion_data))
                span.set_attribute("api.success", True)
                return congestion_data
                
            except Exception as e:
                span.set_attribute("api.success", False)
                span.set_attribute("error.message", str(e))
                sentry_sdk.capture_exception(e)
                logger.error(f"Maritime intelligence error: {e}")
                return {}
    
    async def _get_vessels_near_port(self, lat: float, lng: float, radius_km: float = 50) -> List[VesselInfo]:
        """Get vessels near a port from multiple sources"""
        all_vessels = []
        
        # AISHub data
        aishub_vessels = await self._fetch_aishub_vessels(lat, lng, radius_km)
        all_vessels.extend(aishub_vessels)
        
        # NOAA data (for US ports)
        if -180 <= lng <= -60 and 20 <= lat <= 70:  # Rough US boundaries
            noaa_vessels = await self._fetch_noaa_vessels(lat, lng, radius_km)
            all_vessels.extend(noaa_vessels)
        
        # OpenSeaMap doesn't provide real-time vessel data, but provides port infrastructure
        
        return self._deduplicate_vessels(all_vessels)
    
    async def _fetch_aishub_vessels(self, lat: float, lng: float, radius_km: float) -> List[VesselInfo]:
        """Fetch vessel data from AISHub"""
        if not self.provider_health.get("aishub", False):
            return []
        
        try:
            # AISHub API format
            params = {
                "format": "1",  # JSON format
                "latmin": lat - (radius_km / 111),
                "latmax": lat + (radius_km / 111), 
                "lonmin": lng - (radius_km / (111 * abs(lat / 90))),
                "lonmax": lng + (radius_km / (111 * abs(lat / 90))),
                "output": "json"
            }
            
            provider = MARITIME_PROVIDERS["aishub"]
            url = f"{provider['base_url']}{provider['endpoints']['vessels']}"
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            vessels = []
            data = response.json()
            
            for vessel in data.get("vessels", []):
                vessels.append(VesselInfo(
                    mmsi=int(vessel.get("MMSI", 0)),
                    vessel_name=vessel.get("SHIPNAME", "Unknown"),
                    vessel_type=vessel.get("TYPE", "Unknown"),
                    lat=float(vessel.get("LAT", 0)),
                    lng=float(vessel.get("LON", 0)),
                    speed=float(vessel.get("SPEED", 0)) if vessel.get("SPEED") else None,
                    course=float(vessel.get("COURSE", 0)) if vessel.get("COURSE") else None,
                    timestamp=datetime.utcnow(),
                    source="aishub"
                ))
            
            logger.info(f"AISHub: Found {len(vessels)} vessels near {lat},{lng}")
            return vessels
            
        except Exception as e:
            error_logger.log_api_error(
                service="aishub",
                endpoint="vessels",
                error_message=str(e),
                context={"lat": lat, "lng": lng, "radius_km": radius_km}
            )
            self.provider_health["aishub"] = False
            return []
    
    async def _fetch_noaa_vessels(self, lat: float, lng: float, radius_km: float) -> List[VesselInfo]:
        """Fetch vessel data from NOAA Marine Cadastre"""
        if not self.provider_health.get("noaa_marine", False):
            return []
        
        try:
            # NOAA AIS data API
            provider = MARITIME_PROVIDERS["noaa_marine"]
            url = f"{provider['base_url']}{provider['endpoints']['ais']}"
            
            params = {
                "bbox": f"{lng-0.5},{lat-0.5},{lng+0.5},{lat+0.5}",  # Bounding box
                "format": "json",
                "limit": 1000
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            vessels = []
            data = response.json()
            
            for vessel in data.get("features", []):
                props = vessel.get("properties", {})
                coords = vessel.get("geometry", {}).get("coordinates", [0, 0])
                
                vessels.append(VesselInfo(
                    mmsi=int(props.get("mmsi", 0)),
                    vessel_name=props.get("vesselname", "Unknown"),
                    vessel_type=props.get("vesseltype", "Unknown"),
                    lat=float(coords[1]) if len(coords) > 1 else 0,
                    lng=float(coords[0]) if len(coords) > 0 else 0,
                    speed=float(props.get("sog", 0)) if props.get("sog") else None,
                    course=float(props.get("cog", 0)) if props.get("cog") else None,
                    timestamp=datetime.utcnow(),
                    source="noaa_marine"
                ))
            
            logger.info(f"NOAA: Found {len(vessels)} vessels near {lat},{lng}")
            return vessels
            
        except Exception as e:
            error_logger.log_api_error(
                service="noaa_marine", 
                endpoint="ais",
                error_message=str(e),
                context={"lat": lat, "lng": lng, "radius_km": radius_km}
            )
            self.provider_health["noaa_marine"] = False
            return []
    
    def _deduplicate_vessels(self, vessels: List[VesselInfo]) -> List[VesselInfo]:
        """Remove duplicate vessels based on MMSI"""
        seen_mmsi = set()
        unique_vessels = []
        
        for vessel in vessels:
            if vessel.mmsi not in seen_mmsi:
                seen_mmsi.add(vessel.mmsi)
                unique_vessels.append(vessel)
        
        return unique_vessels
    
    def _calculate_congestion(self, port_code: str, port_info: Dict, vessels: List[VesselInfo]) -> PortCongestion:
        """Calculate port congestion based on vessel positions and speeds"""
        
        # Simple congestion calculation
        vessels_at_anchor = len([v for v in vessels if v.speed is not None and v.speed < 1])
        vessels_at_berth = len([v for v in vessels if v.speed is not None and v.speed < 0.5])
        vessels_moving = len([v for v in vessels if v.speed is not None and v.speed > 1])
        
        # Congestion level based on vessel count and port capacity
        total_vessels = len(vessels)
        
        if total_vessels < 10:
            congestion_level = "low"
        elif total_vessels < 25:
            congestion_level = "medium"
        elif total_vessels < 50:
            congestion_level = "high"
        else:
            congestion_level = "severe"
        
        # Source breakdown
        source_breakdown = {}
        for vessel in vessels:
            source_breakdown[vessel.source] = source_breakdown.get(vessel.source, 0) + 1
        
        return PortCongestion(
            port_code=port_code,
            port_name=port_info["name"],
            vessels_at_anchor=vessels_at_anchor,
            vessels_at_berth=vessels_at_berth,
            average_wait_time_hours=None,  # Would need historical data
            congestion_level=congestion_level,
            last_updated=datetime.utcnow(),
            source_breakdown=source_breakdown
        )
    
    async def get_shipping_delays(self) -> List[ShippingDelay]:
        """Analyze shipping delays across major trade routes"""
        
        cache_key = "shipping_delays_global"
        cached_data, metadata = self.cache.get(cache_key)
        
        if cached_data and metadata and not metadata.is_stale_soft:
            return [ShippingDelay(**delay) for delay in cached_data]
        
        delays = []
        
        # Major trade routes to monitor
        major_routes = [
            {"name": "Trans-Pacific", "origin": "CNSHA", "destination": "USLAX", "typical_days": 14},
            {"name": "Asia-Europe", "origin": "SGSIN", "destination": "NLRTM", "typical_days": 24},
            {"name": "Transatlantic", "origin": "NLRTM", "destination": "USNYC", "typical_days": 8},
            {"name": "Asia-Middle East", "origin": "CNSHA", "destination": "AEDXB", "typical_days": 16}
        ]
        
        # Get congestion data for all relevant ports
        all_port_codes = list(set([route["origin"] for route in major_routes] + 
                                 [route["destination"] for route in major_routes]))
        congestion_data = await self.get_port_congestion(all_port_codes)
        
        for route in major_routes:
            origin_congestion = congestion_data.get(route["origin"])
            dest_congestion = congestion_data.get(route["destination"])
            
            # Calculate delay based on port congestion
            delay_days = 0
            delay_reasons = []
            
            if origin_congestion and origin_congestion.congestion_level in ["high", "severe"]:
                delay_days += 2 if origin_congestion.congestion_level == "high" else 4
                delay_reasons.append(f"Origin port {origin_congestion.port_name} congestion")
            
            if dest_congestion and dest_congestion.congestion_level in ["high", "severe"]:
                delay_days += 1 if dest_congestion.congestion_level == "high" else 3
                delay_reasons.append(f"Destination port {dest_congestion.port_name} congestion")
            
            # Determine severity
            if delay_days == 0:
                severity = "minor"
            elif delay_days < 3:
                severity = "moderate" 
            elif delay_days < 7:
                severity = "major"
            else:
                severity = "critical"
            
            delays.append(ShippingDelay(
                route_name=route["name"],
                origin_port=route["origin"],
                destination_port=route["destination"],
                typical_transit_days=route["typical_days"],
                current_delay_days=delay_days,
                delay_reasons=delay_reasons,
                severity=severity,
                affected_vessels=origin_congestion.vessels_at_anchor + dest_congestion.vessels_at_anchor
                if origin_congestion and dest_congestion else 0
            ))
        
        # Cache for 1 hour
        self.cache.set(cache_key, [delay.__dict__ for delay in delays], source="maritime_intelligence", soft_ttl=3600)
        
        return delays
    
    async def get_supply_chain_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive supply chain risk assessment"""
        
        congestion_data = await self.get_port_congestion()
        shipping_delays = await self.get_shipping_delays()
        
        # Calculate overall risk metrics
        high_risk_ports = [port for port, data in congestion_data.items() 
                          if data.congestion_level in ["high", "severe"]]
        
        critical_delays = [delay for delay in shipping_delays if delay.severity in ["major", "critical"]]
        
        # Risk scoring
        congestion_risk_score = min(len(high_risk_ports) * 15, 100)
        delay_risk_score = min(len(critical_delays) * 20, 100)
        overall_risk_score = (congestion_risk_score + delay_risk_score) / 2
        
        return {
            "overall_risk_score": overall_risk_score,
            "risk_level": "low" if overall_risk_score < 30 else "medium" if overall_risk_score < 60 else "high",
            "high_risk_ports": high_risk_ports,
            "critical_delays": len(critical_delays),
            "port_congestion_summary": {
                port: {
                    "name": data.port_name,
                    "congestion_level": data.congestion_level,
                    "vessels_waiting": data.vessels_at_anchor
                }
                for port, data in congestion_data.items()
            },
            "shipping_delays_summary": [
                {
                    "route": delay.route_name,
                    "delay_days": delay.current_delay_days,
                    "severity": delay.severity,
                    "reasons": delay.delay_reasons
                }
                for delay in shipping_delays
            ],
            "data_sources": list(self.provider_health.keys()),
            "provider_health": self.provider_health,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all maritime data providers"""
        health_results = {}
        
        for provider_name, config in MARITIME_PROVIDERS.items():
            try:
                # Simple health check - try to reach the API
                url = config["base_url"]
                response = await self.client.get(url, timeout=10)
                health_results[provider_name] = response.status_code < 400
            except Exception:
                health_results[provider_name] = False
            
            self.provider_health[provider_name] = health_results[provider_name]
        
        return health_results

# Global instance
maritime_intelligence = FreeMaritimeIntelligence()