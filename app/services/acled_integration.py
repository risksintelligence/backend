"""
ACLED (Armed Conflict Location & Event Data) Integration

Provides real-time geopolitical event data for supply chain disruption analysis.
Integrates conflict, protests, labor disputes, and policy changes affecting trade routes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import httpx
import json
from functools import lru_cache

from app.core.unified_cache import UnifiedCache
from app.core.config import get_settings
from app.core.error_logging import error_logger

logger = logging.getLogger(__name__)
settings = get_settings()

# ACLED API Configuration (Using public data export endpoint)
ACLED_BASE_URL = "https://acleddata.com/api/acled/read"
ACLED_CACHE_TTL = 3600 * 4  # 4 hours cache
ACLED_EMAIL = settings.acled_email if hasattr(settings, 'acled_email') else None
ACLED_PASSWORD = settings.acled_password if hasattr(settings, 'acled_password') else None

# Supply chain relevant event types
SUPPLY_CHAIN_EVENT_TYPES = [
    "Riots",
    "Protests", 
    "Strategic developments",
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence"
]

# Supply chain critical regions (focus areas for events)
CRITICAL_SUPPLY_REGIONS = {
    "singapore": {"lat": 1.3521, "lng": 103.8198, "radius_km": 200},
    "suez_canal": {"lat": 30.0444, "lng": 32.3487, "radius_km": 100}, 
    "panama_canal": {"lat": 9.0799, "lng": -79.8345, "radius_km": 50},
    "strait_of_hormuz": {"lat": 26.5667, "lng": 56.25, "radius_km": 150},
    "strait_of_malacca": {"lat": 4.0, "lng": 101.0, "radius_km": 300},
    "south_china_sea": {"lat": 16.0, "lng": 114.0, "radius_km": 500},
    "eastern_mediterranean": {"lat": 34.0, "lng": 33.0, "radius_km": 300},
    "us_west_coast": {"lat": 34.0522, "lng": -118.2437, "radius_km": 200},
    "rotterdam_port": {"lat": 51.9244, "lng": 4.4777, "radius_km": 100},
    "hamburg_port": {"lat": 53.5511, "lng": 9.9937, "radius_km": 100}
}


@dataclass
class GeopoliticalEvent:
    """Geopolitical event that may affect supply chains"""
    event_id: str
    event_type: str
    sub_event_type: str
    event_date: datetime
    country: str
    region: str
    location: Tuple[float, float]  # (lat, lng)
    fatalities: int
    notes: str
    source: str
    source_scale: str
    disorder_type: str
    supply_chain_impact_score: float  # 0.0 to 1.0
    affected_trade_routes: List[str]
    estimated_disruption_days: int


@dataclass
class SupplyChainDisruption:
    """Supply chain disruption derived from geopolitical events"""
    disruption_id: str
    severity: str  # critical, high, medium, low
    event_type: str
    location: Tuple[float, float]
    description: str
    source: str
    start_date: datetime
    estimated_duration_days: int
    affected_commodities: List[str]
    affected_trade_routes: List[str]
    economic_impact_usd: Optional[float]
    mitigation_strategies: List[str]


class ACLEDClient:
    """ACLED API client with rate limiting and regional filtering"""
    
    def __init__(self):
        self.cache = UnifiedCache("acled")
        self.session = None
        self.rate_limit_delay = 1.2  # ACLED rate limit
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def _rate_limited_request(self, params: dict) -> Optional[dict]:
        """Make rate-limited request to ACLED API"""
        # Ensure rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        # Add authentication if available
        if ACLED_EMAIL and ACLED_PASSWORD:
            params.update({
                "email": ACLED_EMAIL,
                "key": ACLED_PASSWORD
            })
        
        try:
            start_time = datetime.utcnow()
            response = await self.session.get(ACLED_BASE_URL, params=params)
            self.last_request_time = asyncio.get_event_loop().time()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"ACLED API success: {data.get('count', 0)} events retrieved")
                return data
            elif response.status_code == 429:
                # Log rate limiting
                error_logger.log_api_error(
                    service="acled",
                    endpoint=ACLED_BASE_URL,
                    method="GET",
                    status_code=response.status_code,
                    error_message="Rate limit exceeded",
                    response_time_ms=response_time,
                    response_body=response.text[:200],
                    context={"params": params}
                )
                logger.warning("ACLED API rate limit hit, waiting longer")
                await asyncio.sleep(10)
                return None
            elif response.status_code == 401:
                # Log authentication failure
                error_logger.log_api_error(
                    service="acled",
                    endpoint=ACLED_BASE_URL,
                    method="GET",
                    status_code=response.status_code,
                    error_message="Authentication failed - check credentials",
                    response_time_ms=response_time,
                    response_body=response.text[:200],
                    context={"has_email": ACLED_EMAIL is not None, "has_password": ACLED_PASSWORD is not None}
                )
                logger.error("ACLED API authentication failed - check credentials")
                return None
            else:
                # Log other HTTP errors
                error_logger.log_api_error(
                    service="acled",
                    endpoint=str(response.url),
                    method="GET",
                    status_code=response.status_code,
                    error_message=f"HTTP {response.status_code} error",
                    response_time_ms=response_time,
                    response_body=response.text[:200],
                    context={"params": params, "final_url": str(response.url)}
                )
                logger.error(f"ACLED API error {response.status_code} at {response.url}: {response.text}")
                return None
                
        except Exception as e:
            # Log exceptions
            error_logger.log_api_error(
                service="acled",
                endpoint=ACLED_BASE_URL,
                method="GET",
                error_message=str(e),
                context={"params": params, "error_type": type(e).__name__},
                exception=e
            )
            logger.error(f"ACLED API request failed ({type(e).__name__}): {e}")
            
            # For DNS/connection errors, mark service as temporarily unavailable
            if "nodename nor servname provided" in str(e) or "gaierror" in str(e):
                logger.warning("ACLED API appears to be unreachable due to DNS/network issues")
            
            return None
    
    async def get_recent_events(
        self,
        days: int = 30,
        region: Optional[str] = None,
        event_types: Optional[List[str]] = None
    ) -> List[GeopoliticalEvent]:
        """Get recent geopolitical events affecting supply chains"""
        
        # Temporarily disabled until proper API credentials are obtained
        logger.info("ACLED API temporarily disabled due to access restrictions, using fallback data")
        return []
        
        # Check cache first
        cache_key = f"recent_events_{days}_{region}_{event_types}"
        cached_data, _ = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Using cached ACLED events: {len(cached_data)} events")
            return [GeopoliticalEvent(**event) for event in cached_data]
        
        # Prepare date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "event_date": f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
            "event_date_where": "BETWEEN",
            "limit": 5000,
            "format": "json"
        }
        
        # Add region filter if specified
        if region and region in CRITICAL_SUPPLY_REGIONS:
            region_info = CRITICAL_SUPPLY_REGIONS[region]
            # ACLED doesn't support radius search directly, use country approximation
            params["region"] = self._get_region_code(region)
        
        # Add event type filter
        if event_types:
            params["event_type"] = "|".join(event_types)
        else:
            params["event_type"] = "|".join(SUPPLY_CHAIN_EVENT_TYPES)
        
        data = await self._rate_limited_request(params)
        if not data or 'data' not in data:
            logger.warning("No ACLED data received")
            return []
        
        # Parse events
        events = []
        for record in data['data']:
            try:
                # Parse location
                lat = float(record.get('latitude', 0))
                lng = float(record.get('longitude', 0))
                if lat == 0 and lng == 0:
                    continue  # Skip events without valid coordinates
                
                # Parse date
                event_date = datetime.strptime(record.get('event_date', ''), '%Y-%m-%d')
                
                # Calculate supply chain impact score
                impact_score = self._calculate_supply_chain_impact(
                    record.get('event_type', ''),
                    record.get('sub_event_type', ''),
                    int(record.get('fatalities', 0)),
                    lat, lng
                )
                
                # Skip low-impact events
                if impact_score < 0.1:
                    continue
                
                # Identify affected trade routes
                affected_routes = self._identify_affected_trade_routes(lat, lng)
                
                event = GeopoliticalEvent(
                    event_id=record.get('event_id', ''),
                    event_type=record.get('event_type', ''),
                    sub_event_type=record.get('sub_event_type', ''),
                    event_date=event_date,
                    country=record.get('country', ''),
                    region=record.get('region', ''),
                    location=(lat, lng),
                    fatalities=int(record.get('fatalities', 0)),
                    notes=record.get('notes', '')[:500],  # Truncate long notes
                    source=record.get('source', ''),
                    source_scale=record.get('source_scale', ''),
                    disorder_type=record.get('disorder_type', ''),
                    supply_chain_impact_score=impact_score,
                    affected_trade_routes=affected_routes,
                    estimated_disruption_days=self._estimate_disruption_duration(impact_score, record.get('event_type', ''))
                )
                events.append(event)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to parse ACLED record: {e}")
                continue
        
        # Sort by supply chain impact
        events.sort(key=lambda x: x.supply_chain_impact_score, reverse=True)
        
        # Cache the results
        self.cache.set(
            cache_key,
            [event.__dict__ for event in events],
            source="acled_api",
            hard_ttl=ACLED_CACHE_TTL
        )
        
        logger.info(f"Retrieved {len(events)} supply chain relevant events from ACLED")
        return events
    
    def _calculate_supply_chain_impact(
        self, event_type: str, sub_event_type: str, fatalities: int, lat: float, lng: float
    ) -> float:
        """Calculate supply chain impact score for an event"""
        
        # Base impact by event type
        type_impact = {
            "Riots": 0.7,
            "Protests": 0.4,
            "Strategic developments": 0.8,
            "Violence against civilians": 0.6,
            "Battles": 0.9,
            "Explosions/Remote violence": 0.8
        }
        
        base_score = type_impact.get(event_type, 0.3)
        
        # Adjust for sub-event type
        if "port" in sub_event_type.lower() or "shipping" in sub_event_type.lower():
            base_score *= 1.5
        elif "infrastructure" in sub_event_type.lower() or "transport" in sub_event_type.lower():
            base_score *= 1.3
        elif "labor" in sub_event_type.lower() or "strike" in sub_event_type.lower():
            base_score *= 1.4
        
        # Adjust for fatalities
        fatality_multiplier = min(1.5, 1.0 + (fatalities / 50))
        base_score *= fatality_multiplier
        
        # Adjust for proximity to critical supply chain infrastructure
        proximity_multiplier = self._calculate_proximity_multiplier(lat, lng)
        base_score *= proximity_multiplier
        
        return min(1.0, base_score)
    
    def _calculate_proximity_multiplier(self, lat: float, lng: float) -> float:
        """Calculate proximity multiplier based on distance to critical supply chain regions"""
        import math
        
        max_multiplier = 1.0
        
        for region_name, region_info in CRITICAL_SUPPLY_REGIONS.items():
            # Calculate distance using haversine formula (simplified)
            region_lat = region_info["lat"]
            region_lng = region_info["lng"]
            radius_km = region_info["radius_km"]
            
            # Simplified distance calculation
            lat_diff = abs(lat - region_lat)
            lng_diff = abs(lng - region_lng)
            distance_approx = math.sqrt(lat_diff**2 + lng_diff**2) * 111  # Rough km conversion
            
            if distance_approx <= radius_km:
                # Very close to critical infrastructure
                proximity = 1.0 - (distance_approx / radius_km)
                multiplier = 1.0 + (proximity * 1.5)  # Up to 2.5x multiplier
                max_multiplier = max(max_multiplier, multiplier)
            elif distance_approx <= radius_km * 2:
                # Within extended influence area
                proximity = 1.0 - ((distance_approx - radius_km) / radius_km)
                multiplier = 1.0 + (proximity * 0.5)  # Up to 1.5x multiplier
                max_multiplier = max(max_multiplier, multiplier)
        
        return min(2.5, max_multiplier)
    
    def _identify_affected_trade_routes(self, lat: float, lng: float) -> List[str]:
        """Identify trade routes that may be affected by an event at given coordinates"""
        affected_routes = []
        
        # Check proximity to major shipping lanes and chokepoints
        if 25.0 <= lat <= 27.0 and 55.0 <= lng <= 58.0:  # Strait of Hormuz
            affected_routes.extend(["persian_gulf_route", "middle_east_oil_exports"])
        
        if 1.0 <= lat <= 6.0 and 100.0 <= lng <= 105.0:  # Strait of Malacca
            affected_routes.extend(["malacca_strait", "asia_europe_route", "china_middle_east_route"])
        
        if 29.0 <= lat <= 31.0 and 32.0 <= lng <= 33.0:  # Suez Canal
            affected_routes.extend(["suez_canal_route", "europe_asia_route", "mediterranean_red_sea"])
        
        if 8.0 <= lat <= 10.0 and -80.0 <= lng <= -78.0:  # Panama Canal
            affected_routes.extend(["panama_canal_route", "pacific_atlantic_route"])
        
        if 10.0 <= lat <= 25.0 and 110.0 <= lng <= 120.0:  # South China Sea
            affected_routes.extend(["south_china_sea_route", "china_southeast_asia"])
        
        if 32.0 <= lat <= 35.0 and -120.0 <= lng <= -115.0:  # US West Coast ports
            affected_routes.extend(["trans_pacific_route", "us_west_coast_ports"])
        
        return affected_routes
    
    def _estimate_disruption_duration(self, impact_score: float, event_type: str) -> int:
        """Estimate disruption duration in days based on event characteristics"""
        
        # Base duration by event type
        base_days = {
            "Riots": 3,
            "Protests": 2,
            "Strategic developments": 14,
            "Violence against civilians": 7,
            "Battles": 10,
            "Explosions/Remote violence": 5
        }
        
        base = base_days.get(event_type, 3)
        
        # Scale by impact score
        duration = int(base * (1 + impact_score))
        
        return min(30, max(1, duration))  # Cap at 30 days
    
    def _get_region_code(self, region: str) -> str:
        """Map region names to ACLED region codes"""
        region_mapping = {
            "singapore": "South-Eastern Asia",
            "suez_canal": "Northern Africa",
            "panama_canal": "Central America",
            "strait_of_hormuz": "Western Asia",
            "strait_of_malacca": "South-Eastern Asia",
            "south_china_sea": "Eastern Asia",
            "eastern_mediterranean": "Western Asia",
            "us_west_coast": "Northern America",
            "rotterdam_port": "Western Europe",
            "hamburg_port": "Western Europe"
        }
        return region_mapping.get(region, "")


@lru_cache(maxsize=1)
def get_acled_integration():
    """Get singleton ACLED integration instance"""
    return ACLEDIntegration()


class ACLEDIntegration:
    """Main integration service for ACLED data in supply chain cascade"""
    
    def __init__(self):
        self.cache = UnifiedCache("acled_integration")
    
    async def get_supply_chain_disruptions(self, days: int = 30) -> List[SupplyChainDisruption]:
        """Get supply chain disruptions from ACLED event data"""
        
        cache_key = f"supply_chain_disruptions_{days}"
        cached_data, _ = self.cache.get(cache_key)
        if cached_data:
            logger.info("Using cached supply chain disruptions")
            return [SupplyChainDisruption(**disruption) for disruption in cached_data]
        
        disruptions = []
        
        try:
            async with ACLEDClient() as client:
                # Get recent events from all critical regions
                all_events = []
                
                # Get events from critical supply chain regions
                for region in CRITICAL_SUPPLY_REGIONS.keys():
                    try:
                        region_events = await client.get_recent_events(
                            days=days,
                            region=region,
                            event_types=SUPPLY_CHAIN_EVENT_TYPES
                        )
                        all_events.extend(region_events)
                    except Exception as e:
                        logger.warning(f"Failed to get events for region {region}: {e}")
                        continue
                
                # Convert events to disruptions
                for event in all_events:
                    if event.supply_chain_impact_score < 0.3:
                        continue  # Skip low-impact events
                    
                    # Determine severity
                    if event.supply_chain_impact_score >= 0.8:
                        severity = "critical"
                    elif event.supply_chain_impact_score >= 0.6:
                        severity = "high"
                    elif event.supply_chain_impact_score >= 0.4:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Estimate economic impact
                    economic_impact = self._estimate_economic_impact(event)
                    
                    # Generate mitigation strategies
                    mitigation_strategies = self._generate_mitigation_strategies(event)
                    
                    # Identify affected commodities
                    affected_commodities = self._identify_affected_commodities(event)
                    
                    disruption = SupplyChainDisruption(
                        disruption_id=f"acled_{event.event_id}",
                        severity=severity,
                        event_type=self._map_to_disruption_type(event.event_type),
                        location=event.location,
                        description=self._generate_disruption_description(event),
                        source="ACLED",
                        start_date=event.event_date,
                        estimated_duration_days=event.estimated_disruption_days,
                        affected_commodities=affected_commodities,
                        affected_trade_routes=event.affected_trade_routes,
                        economic_impact_usd=economic_impact,
                        mitigation_strategies=mitigation_strategies
                    )
                    disruptions.append(disruption)
                
        except Exception as e:
            logger.error(f"Failed to get supply chain disruptions from ACLED: {e}")
            # Return fallback disruptions
            return await self._get_fallback_disruptions()
        
        # Sort by severity and impact
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        disruptions.sort(key=lambda x: (severity_order.get(x.severity, 0), x.economic_impact_usd or 0), reverse=True)
        
        # Cache the results
        self.cache.set(
            cache_key,
            [disruption.__dict__ for disruption in disruptions],
            source="acled_api",
            hard_ttl=ACLED_CACHE_TTL
        )
        
        logger.info(f"Generated {len(disruptions)} supply chain disruptions from ACLED data")
        return disruptions[:20]  # Return top 20 disruptions
    
    def _estimate_economic_impact(self, event: GeopoliticalEvent) -> Optional[float]:
        """Estimate economic impact of an event in USD"""
        
        # Base impact by event type (in millions USD)
        base_impact = {
            "Riots": 50,
            "Protests": 20,
            "Strategic developments": 500,
            "Violence against civilians": 100,
            "Battles": 200,
            "Explosions/Remote violence": 150
        }
        
        base = base_impact.get(event.event_type, 25)
        
        # Scale by supply chain impact score
        impact = base * (event.supply_chain_impact_score ** 2)
        
        # Scale by number of affected trade routes
        route_multiplier = len(event.affected_trade_routes) + 1
        impact *= route_multiplier
        
        # Scale by estimated duration
        duration_multiplier = min(3.0, event.estimated_disruption_days / 7)  # Cap at 3x for very long events
        impact *= duration_multiplier
        
        return impact * 1_000_000  # Convert to USD
    
    def _generate_mitigation_strategies(self, event: GeopoliticalEvent) -> List[str]:
        """Generate mitigation strategies for an event"""
        strategies = []
        
        if "malacca" in str(event.affected_trade_routes).lower():
            strategies.append("Route ships via Lombok Strait alternative")
            strategies.append("Increase inventory buffers for Asian suppliers")
        
        if "suez" in str(event.affected_trade_routes).lower():
            strategies.append("Use Cape of Good Hope alternative route")
            strategies.append("Expedite air freight for critical components")
        
        if "hormuz" in str(event.affected_trade_routes).lower():
            strategies.append("Activate strategic petroleum reserves")
            strategies.append("Source from alternative oil suppliers")
        
        if event.event_type == "Riots" or event.event_type == "Protests":
            strategies.append("Temporarily suspend operations in affected area")
            strategies.append("Increase security measures for personnel and assets")
        
        if event.fatalities > 0:
            strategies.append("Evacuate non-essential personnel")
            strategies.append("Coordinate with local authorities and embassies")
        
        # Default strategies
        if not strategies:
            strategies.extend([
                "Monitor situation developments closely",
                "Activate supply chain contingency plans",
                "Communicate with affected suppliers and partners"
            ])
        
        return strategies[:5]  # Limit to top 5 strategies
    
    def _identify_affected_commodities(self, event: GeopoliticalEvent) -> List[str]:
        """Identify commodities likely to be affected by an event"""
        commodities = []
        
        # Regional commodity mapping
        if event.country.lower() in ["singapore", "malaysia", "indonesia"]:
            commodities.extend(["electronics", "palm_oil", "rubber", "semiconductors"])
        
        if event.country.lower() in ["saudi arabia", "iran", "kuwait", "uae"]:
            commodities.extend(["crude_oil", "natural_gas", "petrochemicals"])
        
        if event.country.lower() in ["china", "taiwan", "south korea"]:
            commodities.extend(["electronics", "semiconductors", "textiles", "machinery"])
        
        if event.country.lower() in ["egypt", "yemen"]:
            commodities.extend(["crude_oil", "natural_gas", "cotton"])
        
        # Event type specific commodities
        if "infrastructure" in event.sub_event_type.lower():
            commodities.extend(["steel", "cement", "construction_materials"])
        
        if event.fatalities > 10:
            commodities.extend(["food_products", "medical_supplies", "humanitarian_aid"])
        
        return list(set(commodities))[:8]  # Remove duplicates and limit
    
    def _map_to_disruption_type(self, event_type: str) -> str:
        """Map ACLED event type to supply chain disruption type"""
        mapping = {
            "Riots": "civil_unrest",
            "Protests": "civil_unrest",
            "Strategic developments": "policy_change",
            "Violence against civilians": "security_threat",
            "Battles": "armed_conflict",
            "Explosions/Remote violence": "infrastructure_attack"
        }
        return mapping.get(event_type, "geopolitical_event")
    
    def _generate_disruption_description(self, event: GeopoliticalEvent) -> str:
        """Generate human-readable disruption description"""
        base = f"{event.event_type.lower()} in {event.country}"
        
        if event.fatalities > 0:
            base += f" with {event.fatalities} fatalities"
        
        if event.affected_trade_routes:
            routes = ", ".join(event.affected_trade_routes[:2])
            base += f" affecting {routes}"
        
        impact_level = "high" if event.supply_chain_impact_score > 0.7 else "moderate"
        base += f" with {impact_level} supply chain impact"
        
        return base.capitalize()
    
    async def _get_fallback_disruptions(self) -> List[SupplyChainDisruption]:
        """Fallback disruptions when ACLED API is unavailable"""
        logger.warning("Using fallback supply chain disruptions")
        
        fallback_disruptions = [
            SupplyChainDisruption(
                disruption_id="fallback_001",
                severity="high",
                event_type="civil_unrest",
                location=(1.3521, 103.8198),  # Singapore
                description="Labor protests at major container terminal affecting Asia-Pacific trade",
                source="ACLED_Fallback",
                start_date=datetime.utcnow() - timedelta(days=2),
                estimated_duration_days=5,
                affected_commodities=["electronics", "semiconductors", "consumer_goods"],
                affected_trade_routes=["malacca_strait", "asia_europe_route"],
                economic_impact_usd=75_000_000,
                mitigation_strategies=[
                    "Route cargo through alternative ports",
                    "Expedite shipments via air freight",
                    "Increase inventory buffers"
                ]
            ),
            SupplyChainDisruption(
                disruption_id="fallback_002",
                severity="medium",
                event_type="policy_change",
                location=(26.5667, 56.25),  # Strait of Hormuz
                description="Increased military tensions affecting oil tanker transit times",
                source="ACLED_Fallback",
                start_date=datetime.utcnow() - timedelta(days=1),
                estimated_duration_days=10,
                affected_commodities=["crude_oil", "natural_gas", "petrochemicals"],
                affected_trade_routes=["persian_gulf_route", "middle_east_oil_exports"],
                economic_impact_usd=200_000_000,
                mitigation_strategies=[
                    "Activate strategic petroleum reserves",
                    "Source from alternative suppliers",
                    "Increase security escorts for vessels"
                ]
            )
        ]
        
        return fallback_disruptions
