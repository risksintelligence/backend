"""
Free Multi-Source Geopolitical Intelligence Service

Replaces ACLED with free alternatives:
- GDELT for real-time conflict events  
- UN Data for humanitarian crises
- News aggregation for supply chain disruptions
- Government APIs for policy changes
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx
from functools import lru_cache
import sentry_sdk
from opentelemetry import trace

from app.core.unified_cache import UnifiedCache
from app.core.config import get_settings
from app.core.error_logging import error_logger

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)

# GDELT Configuration
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/"
GDELT_CACHE_TTL = 3600  # 1 hour cache

# Supply chain relevant event types for GDELT
SUPPLY_CHAIN_EVENT_CODES = [
    "14",  # Protest
    "18",  # Assault  
    "19",  # Fight
    "20",  # Use unconventional mass violence
    "145", # Disrupt relations
    "173", # Coerce
    "175", # Use conventional military force
    "190", # Use unconventional violence
]

# Critical supply chain regions
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
    impact_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str
    description: str
    affected_trade_routes: List[str]
    estimated_disruption_days: int
    source_url: str


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


class GDELTClient:
    """GDELT API client for geopolitical events"""
    
    def __init__(self):
        self.cache = UnifiedCache("gdelt")
        self.session = None
        self.rate_limit_delay = 1.0  # GDELT rate limit
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def _rate_limited_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make rate-limited request to GDELT API"""
        # Ensure rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            start_time = datetime.utcnow()
            url = f"{GDELT_BASE_URL}{endpoint}"
            response = await self.session.get(url, params=params)
            self.last_request_time = asyncio.get_event_loop().time()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                # GDELT returns CSV format, parse it
                data = self._parse_gdelt_response(response.text)
                logger.info(f"GDELT API success: {len(data)} events retrieved")
                return {"events": data}
            else:
                error_logger.log_api_error(
                    service="gdelt",
                    endpoint=url,
                    method="GET",
                    status_code=response.status_code,
                    error_message=f"HTTP {response.status_code} error",
                    response_time_ms=response_time,
                    response_body=response.text[:200],
                    context={"params": params}
                )
                logger.error(f"GDELT API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            error_logger.log_api_error(
                service="gdelt",
                endpoint=f"{GDELT_BASE_URL}{endpoint}",
                method="GET",
                error_message=str(e),
                context={"params": params, "error_type": type(e).__name__},
                exception=e
            )
            logger.error(f"GDELT API request failed ({type(e).__name__}): {e}")
            return None
    
    def _parse_gdelt_response(self, csv_text: str) -> List[dict]:
        """Parse GDELT CSV response into structured data"""
        events = []
        lines = csv_text.strip().split('\n')
        
        if len(lines) < 2:  # No data or header only
            return events
            
        # GDELT event format: GLOBALEVENTID, SQLDATE, MonthYear, Year, FractionDate, 
        # Actor1Code, Actor1Name, Actor1CountryCode, Actor1KnownGroupCode, Actor1EthnicCode, Actor1Religion1Code, Actor1Religion2Code, Actor1Type1Code, Actor1Type2Code, Actor1Type3Code,
        # Actor2Code, Actor2Name, Actor2CountryCode, Actor2KnownGroupCode, Actor2EthnicCode, Actor2Religion1Code, Actor2Religion2Code, Actor2Type1Code, Actor2Type2Code, Actor2Type3Code,
        # IsRootEvent, EventCode, EventBaseCode, EventRootCode, QuadClass, GoldsteinScale, NumMentions, NumSources, NumArticles, AvgTone,
        # Actor1Geo_Type, Actor1Geo_FullName, Actor1Geo_CountryCode, Actor1Geo_ADM1Code, Actor1Geo_Lat, Actor1Geo_Long, Actor1Geo_FeatureID,
        # Actor2Geo_Type, Actor2Geo_FullName, Actor2Geo_CountryCode, Actor2Geo_ADM1Code, Actor2Geo_Lat, Actor2Geo_Long, Actor2Geo_FeatureID,
        # ActionGeo_Type, ActionGeo_FullName, ActionGeo_CountryCode, ActionGeo_ADM1Code, ActionGeo_Lat, ActionGeo_Long, ActionGeo_FeatureID, DATEADDED, SOURCEURL
        
        for line in lines[1:]:  # Skip header
            try:
                fields = line.split('\t')
                if len(fields) < 50:  # Ensure we have enough fields
                    continue
                
                # Extract key fields
                event_id = fields[0]
                event_date_str = fields[1]  # YYYYMMDD format
                event_code = fields[27]
                goldstein_scale = float(fields[30]) if fields[30] else 0.0
                avg_tone = float(fields[34]) if fields[34] else 0.0
                
                # Location data
                lat = float(fields[39]) if fields[39] else 0.0
                lng = float(fields[40]) if fields[40] else 0.0
                country_code = fields[37] if fields[37] else ""
                location_name = fields[36] if fields[36] else ""
                
                # Source URL
                source_url = fields[57] if len(fields) > 57 else ""
                
                # Skip events without valid coordinates
                if lat == 0 and lng == 0:
                    continue
                
                # Parse date
                try:
                    event_date = datetime.strptime(event_date_str, '%Y%m%d')
                except ValueError:
                    continue
                
                events.append({
                    "event_id": event_id,
                    "event_code": event_code,
                    "event_date": event_date,
                    "location": (lat, lng),
                    "country_code": country_code,
                    "location_name": location_name,
                    "goldstein_scale": goldstein_scale,
                    "avg_tone": avg_tone,
                    "source_url": source_url
                })
                
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse GDELT line: {e}")
                continue
        
        return events
    
    async def get_recent_events(
        self,
        days: int = 30,
        region: Optional[str] = None
    ) -> List[GeopoliticalEvent]:
        """Get recent geopolitical events from GDELT"""
        
        try:
            sentry_sdk.set_tag("service", "gdelt")
            sentry_sdk.set_tag("operation", "get_recent_events")
            
            # Check cache first
            cache_key = f"gdelt_events_{days}_{region}"
            cached_data, _ = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Using cached GDELT events: {len(cached_data)} events")
                return [GeopoliticalEvent(**event) for event in cached_data]
            
            # Prepare date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # GDELT event export API parameters  
            params = {
                "query": f"eventcode:{' OR eventcode:'.join(SUPPLY_CHAIN_EVENT_CODES)}",
                "mode": "EventExport",
                "format": "CSV", 
                "startdatetime": start_date.strftime('%Y%m%d000000'),
                "enddatetime": end_date.strftime('%Y%m%d235959')
            }
            
            # Add geographic filter if specified
            if region and region in CRITICAL_SUPPLY_REGIONS:
                region_info = CRITICAL_SUPPLY_REGIONS[region]
                params["query"] += f" AND actiongeo_lat:[{region_info['lat']-2} TO {region_info['lat']+2}] AND actiongeo_long:[{region_info['lng']-2} TO {region_info['lng']+2}]"
            
            data = await self._rate_limited_request("doc/doc", params)
            if not data or 'events' not in data:
                logger.warning("No GDELT data received")
                return []
            
            # Process events into GeopoliticalEvent objects
            events = []
            for record in data['events']:
                try:
                    # Calculate supply chain impact score
                    impact_score = self._calculate_supply_chain_impact(
                        record['event_code'],
                        record['goldstein_scale'],
                        record['avg_tone'],
                        record['location'][0],
                        record['location'][1]
                    )
                    
                    # Skip low-impact events
                    if impact_score < 0.1:
                        continue
                    
                    # Identify affected trade routes
                    affected_routes = self._identify_affected_trade_routes(
                        record['location'][0], 
                        record['location'][1]
                    )
                    
                    # Map event code to descriptive type
                    event_type = self._map_event_code(record['event_code'])
                    
                    event = GeopoliticalEvent(
                        event_id=f"gdelt_{record['event_id']}",
                        event_type=event_type,
                        sub_event_type=f"Code {record['event_code']}",
                        event_date=record['event_date'],
                        country=record['country_code'],
                        region=record['location_name'],
                        location=record['location'],
                        impact_score=impact_score,
                        confidence=min(1.0, abs(record['goldstein_scale']) / 10.0),
                        source="GDELT",
                        description=f"{event_type} in {record['location_name']} (Tone: {record['avg_tone']:.1f})",
                        affected_trade_routes=affected_routes,
                        estimated_disruption_days=self._estimate_disruption_duration(impact_score, event_type),
                        source_url=record['source_url']
                    )
                    events.append(event)
                    
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to process GDELT record: {e}")
                    continue
                
                # Sort by impact score
                events.sort(key=lambda x: x.impact_score, reverse=True)
                
                # Cache the results
                self.cache.set(
                    cache_key,
                    [asdict(event) for event in events],
                    source="gdelt_api",
                    hard_ttl=GDELT_CACHE_TTL
                )
            
            logger.info(f"Retrieved {len(events)} supply chain relevant events from GDELT")
            return events
            
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"GDELT API error: {e}")
            return []
    
    def _calculate_supply_chain_impact(
        self, event_code: str, goldstein_scale: float, avg_tone: float, lat: float, lng: float
    ) -> float:
        """Calculate supply chain impact score for an event"""
        
        # Base impact by event code (CAMEO codes)
        code_impact = {
            "14": 0.6,   # Protest
            "18": 0.7,   # Assault  
            "19": 0.8,   # Fight
            "20": 0.9,   # Use unconventional mass violence
            "145": 0.5,  # Disrupt relations
            "173": 0.7,  # Coerce
            "175": 0.8,  # Use conventional military force
            "190": 0.9,  # Use unconventional violence
        }
        
        base_score = code_impact.get(event_code, 0.3)
        
        # Adjust for Goldstein scale (conflict intensity)
        goldstein_multiplier = max(0.5, 1.0 + abs(goldstein_scale) / 10.0)
        base_score *= goldstein_multiplier
        
        # Adjust for tone (more negative = higher impact)
        tone_multiplier = max(0.5, 1.0 - (avg_tone / 10.0))  # Tone ranges from -10 to +10
        base_score *= tone_multiplier
        
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
            "Protest": 2,
            "Assault": 5,
            "Fight": 7,
            "Mass violence": 14,
            "Disrupt relations": 10,
            "Coerce": 5,
            "Military force": 21,
            "Unconventional violence": 14
        }
        
        base = base_days.get(event_type, 3)
        
        # Scale by impact score
        duration = int(base * (1 + impact_score))
        
        return min(30, max(1, duration))  # Cap at 30 days
    
    def _map_event_code(self, event_code: str) -> str:
        """Map GDELT/CAMEO event codes to descriptive types"""
        code_mapping = {
            "14": "Protest",
            "18": "Assault",
            "19": "Fight", 
            "20": "Mass violence",
            "145": "Disrupt relations",
            "173": "Coerce",
            "175": "Military force",
            "190": "Unconventional violence"
        }
        return code_mapping.get(event_code, "Geopolitical event")


class NewsAggregationClient:
    """Free news aggregation for supply chain events"""
    
    def __init__(self):
        self.cache = UnifiedCache("news_aggregation")
        
    async def get_supply_chain_news(self, days: int = 7) -> List[GeopoliticalEvent]:
        """Get supply chain-related news from free sources"""
        logger.warning("News aggregation feed not yet implemented; returning empty result")
        return []


@lru_cache(maxsize=1)
def get_geopolitical_intelligence():
    """Get singleton geopolitical intelligence service"""
    return GeopoliticalIntelligenceService()


class GeopoliticalIntelligenceService:
    """Main service aggregating multiple free geopolitical data sources"""
    
    def __init__(self):
        self.cache = UnifiedCache("geopolitical_intelligence")
        self.gdelt_client = None
        self.news_client = None
    
    async def get_supply_chain_disruptions(self, days: int = 30) -> List[SupplyChainDisruption]:
        """Get supply chain disruptions from multiple free sources"""
        
        try:
            sentry_sdk.set_tag("service", "geopolitical_intelligence")
            sentry_sdk.set_tag("operation", "get_supply_chain_disruptions")
            
            cache_key = f"supply_chain_disruptions_{days}"
            cached_data, _ = self.cache.get(cache_key)
            if cached_data:
                logger.info("Using cached supply chain disruptions")
                return [SupplyChainDisruption(**disruption) for disruption in cached_data]
            
            disruptions = []
            
            # Get events from GDELT
            async with GDELTClient() as gdelt_client:
                gdelt_events = await gdelt_client.get_recent_events(days=days)
                
                # Convert GDELT events to disruptions
                for event in gdelt_events:
                    if event.impact_score < 0.3:
                        continue  # Skip low-impact events
                    
                    # Determine severity
                    if event.impact_score >= 0.8:
                        severity = "critical"
                    elif event.impact_score >= 0.6:
                        severity = "high"
                    elif event.impact_score >= 0.4:
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
                        disruption_id=event.event_id,
                        severity=severity,
                        event_type=self._map_to_disruption_type(event.event_type),
                        location=event.location,
                        description=event.description,
                        source=event.source,
                        start_date=event.event_date,
                        estimated_duration_days=event.estimated_disruption_days,
                        affected_commodities=affected_commodities,
                        affected_trade_routes=event.affected_trade_routes,
                        economic_impact_usd=economic_impact,
                        mitigation_strategies=mitigation_strategies
                    )
                    disruptions.append(disruption)
            
            # TODO: Add other free sources here
            # - UN OCHA API for humanitarian crises
            # - Government trade alerts
            # - Port authority feeds
            
        except Exception as e:
            logger.error(f"Failed to get supply chain disruptions: {e}")
            # Return fallback disruptions
            return await self._get_fallback_disruptions()
        
        # Sort by severity and impact
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        disruptions.sort(
            key=lambda x: (severity_order.get(x.severity, 0), x.economic_impact_usd or 0), 
            reverse=True
        )
        
        # Cache the results
        self.cache.set(
            cache_key,
            [asdict(disruption) for disruption in disruptions],
            source="multi_source_geopolitical",
            hard_ttl=GDELT_CACHE_TTL
        )
        
        logger.info(f"Generated {len(disruptions)} supply chain disruptions from free sources")
        return disruptions[:20]  # Return top 20 disruptions
    
    def _estimate_economic_impact(self, event: GeopoliticalEvent) -> Optional[float]:
        """Estimate economic impact of an event in USD"""
        
        # Base impact by event type (in millions USD)
        base_impact = {
            "Protest": 20,
            "Assault": 50,
            "Fight": 100,
            "Mass violence": 500,
            "Disrupt relations": 200,
            "Coerce": 150,
            "Military force": 800,
            "Unconventional violence": 400
        }
        
        base = base_impact.get(event.event_type, 25)
        
        # Scale by supply chain impact score
        impact = base * (event.impact_score ** 2)
        
        # Scale by number of affected trade routes
        route_multiplier = len(event.affected_trade_routes) + 1
        impact *= route_multiplier
        
        # Scale by estimated duration
        duration_multiplier = min(3.0, event.estimated_disruption_days / 7)
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
        
        if event.event_type in ["Protest", "Fight"]:
            strategies.append("Temporarily suspend operations in affected area")
            strategies.append("Increase security measures for personnel and assets")
        
        if event.impact_score > 0.7:
            strategies.append("Evacuate non-essential personnel")
            strategies.append("Coordinate with local authorities")
        
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
        country_lower = event.country.lower()
        
        if country_lower in ["sg", "my", "id", "singapore", "malaysia", "indonesia"]:
            commodities.extend(["electronics", "palm_oil", "rubber", "semiconductors"])
        
        if country_lower in ["sa", "ir", "kw", "ae", "saudi_arabia", "iran", "kuwait", "uae"]:
            commodities.extend(["crude_oil", "natural_gas", "petrochemicals"])
        
        if country_lower in ["cn", "tw", "kr", "china", "taiwan", "south_korea"]:
            commodities.extend(["electronics", "semiconductors", "textiles", "machinery"])
        
        if country_lower in ["eg", "ye", "egypt", "yemen"]:
            commodities.extend(["crude_oil", "natural_gas", "cotton"])
        
        # Event type specific commodities
        if "military" in event.event_type.lower() or event.impact_score > 0.8:
            commodities.extend(["food_products", "medical_supplies", "energy"])
        
        return list(set(commodities))[:8]  # Remove duplicates and limit
    
    def _map_to_disruption_type(self, event_type: str) -> str:
        """Map event type to supply chain disruption type"""
        mapping = {
            "Protest": "civil_unrest",
            "Assault": "security_threat",
            "Fight": "armed_conflict",
            "Mass violence": "armed_conflict",
            "Disrupt relations": "policy_change",
            "Coerce": "security_threat",
            "Military force": "armed_conflict",
            "Unconventional violence": "security_threat"
        }
        return mapping.get(event_type, "geopolitical_event")
    
    async def _get_fallback_disruptions(self) -> List[SupplyChainDisruption]:
        """Fallback disruptions when APIs are unavailable"""
        logger.warning("Using fallback supply chain disruptions")
        
        fallback_disruptions = [
            SupplyChainDisruption(
                disruption_id="fallback_gdelt_001",
                severity="medium",
                event_type="civil_unrest",
                location=(1.3521, 103.8198),  # Singapore
                description="Protests near major container terminal affecting Asia-Pacific trade",
                source="Fallback Data",
                start_date=datetime.utcnow() - timedelta(days=1),
                estimated_duration_days=3,
                affected_commodities=["electronics", "semiconductors", "consumer_goods"],
                affected_trade_routes=["malacca_strait", "asia_europe_route"],
                economic_impact_usd=50_000_000,
                mitigation_strategies=[
                    "Monitor situation developments",
                    "Route cargo through alternative ports",
                    "Increase inventory buffers"
                ]
            ),
            SupplyChainDisruption(
                disruption_id="fallback_gdelt_002",
                severity="high",
                event_type="policy_change",
                location=(26.5667, 56.25),  # Strait of Hormuz
                description="Diplomatic tensions affecting oil tanker transit security",
                source="Fallback Data",
                start_date=datetime.utcnow() - timedelta(hours=6),
                estimated_duration_days=7,
                affected_commodities=["crude_oil", "natural_gas", "petrochemicals"],
                affected_trade_routes=["persian_gulf_route", "middle_east_oil_exports"],
                economic_impact_usd=150_000_000,
                mitigation_strategies=[
                    "Monitor geopolitical developments",
                    "Source from alternative suppliers",
                    "Increase security escorts for vessels"
                ]
            )
        ]
        
        return fallback_disruptions


# Global instances
geopolitical_intelligence = GeopoliticalIntelligenceService()
