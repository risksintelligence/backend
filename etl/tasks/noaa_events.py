"""
NOAA weather events and alerts fetcher for disruption risk analysis.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

import aiohttp
import pandas as pd

from src.core.config import settings
from src.cache.cache_manager import CacheManager
from src.data.sources.noaa import NoaaConnector

logger = logging.getLogger(__name__)


class NoaaEventsFetcher:
    """
    Fetches weather events and alerts from NOAA for supply chain risk analysis.
    """
    
    def __init__(self):
        """Initialize NOAA events fetcher."""
        self.cache_manager = CacheManager()
        self.noaa_connector = NoaaConnector(self.cache_manager)
        
        # Severe weather events that impact supply chains
        self.severe_event_types = {
            "tornado",
            "hurricane",
            "blizzard", 
            "ice_storm",
            "flood",
            "severe_thunderstorm",
            "winter_storm",
            "fire_weather",
            "dust_storm",
            "high_wind"
        }
        
        # Risk severity mapping
        self.severity_scores = {
            "extreme": 5,
            "severe": 4,
            "moderate": 3,
            "minor": 2,
            "unknown": 1
        }
    
    async def fetch_latest_events(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch latest weather events and alerts from NOAA.
        
        Returns:
            List of weather event records or None if failed
        """
        logger.info("Starting NOAA weather events fetch")
        
        try:
            all_events = []
            
            # Fetch active weather alerts
            alerts = await self._fetch_weather_alerts()
            if alerts:
                all_events.extend(alerts)
            
            # Fetch recent severe weather events
            recent_events = await self._fetch_recent_events()
            if recent_events:
                all_events.extend(recent_events)
            
            # Fetch climate data for trend analysis
            climate_data = await self._fetch_climate_data()
            if climate_data:
                all_events.extend(climate_data)
            
            if all_events:
                # Cache the aggregated data
                cache_key = f"noaa:latest_fetch:{datetime.now().strftime('%Y%m%d_%H')}"
                await self._cache_data(cache_key, all_events)
                
                logger.info(f"NOAA events fetch completed: {len(all_events)} records")
                return all_events
            else:
                logger.warning("No NOAA events retrieved")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching NOAA events: {str(e)}")
            
            # Try to get cached data as fallback
            fallback_data = await self._get_fallback_data()
            if fallback_data:
                logger.info("Using fallback NOAA data")
                return fallback_data
            
            return None
    
    async def _fetch_weather_alerts(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch active weather alerts.
        
        Returns:
            List of alert records or None if failed
        """
        try:
            alerts_data = await self.noaa_connector.get_active_alerts()
            
            if not alerts_data or "features" not in alerts_data:
                return None
            
            transformed_alerts = []
            for alert in alerts_data["features"]:
                properties = alert.get("properties", {})
                geometry = alert.get("geometry", {})
                
                # Calculate risk score based on severity and urgency
                risk_score = self._calculate_alert_risk_score(properties)
                
                transformed_alert = {
                    "source": "noaa",
                    "data_type": "weather_alert",
                    "alert_id": properties.get("id"),
                    "event_type": properties.get("event", "").lower().replace(" ", "_"),
                    "severity": properties.get("severity", "unknown").lower(),
                    "certainty": properties.get("certainty", "unknown").lower(),
                    "urgency": properties.get("urgency", "unknown").lower(),
                    "headline": properties.get("headline"),
                    "description": properties.get("description"),
                    "instruction": properties.get("instruction"),
                    "area_description": properties.get("areaDesc"),
                    "effective_time": properties.get("effective"),
                    "expires_time": properties.get("expires"),
                    "onset_time": properties.get("onset"),
                    "ends_time": properties.get("ends"),
                    "risk_score": risk_score,
                    "geometry": geometry,
                    "last_updated": datetime.now().isoformat(),
                    "is_severe": any(event in properties.get("event", "").lower() 
                                   for event in self.severe_event_types)
                }
                transformed_alerts.append(transformed_alert)
            
            # Filter for supply chain relevant alerts
            relevant_alerts = [
                alert for alert in transformed_alerts 
                if alert["is_severe"] or alert["risk_score"] >= 3
            ]
            
            return relevant_alerts
            
        except Exception as e:
            logger.error(f"Error fetching NOAA weather alerts: {str(e)}")
            return None
    
    async def _fetch_recent_events(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch recent severe weather events from NOAA Storm Events Database.
        
        Returns:
            List of event records or None if failed
        """
        try:
            # Get events from the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            events_data = await self.noaa_connector.get_storm_events(
                start_date=start_date,
                end_date=end_date
            )
            
            if not events_data:
                return None
            
            transformed_events = []
            for event in events_data:
                # Calculate economic impact score
                impact_score = self._calculate_economic_impact(event)
                
                transformed_event = {
                    "source": "noaa",
                    "data_type": "storm_event",
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type", "").lower().replace(" ", "_"),
                    "state": event.get("state"),
                    "state_fips": event.get("state_fips"),
                    "county": event.get("cz_name"),
                    "county_fips": event.get("cz_fips"),
                    "begin_date": event.get("begin_date"),
                    "end_date": event.get("end_date"),
                    "injuries_direct": self._parse_int(event.get("injuries_direct")),
                    "injuries_indirect": self._parse_int(event.get("injuries_indirect")),
                    "deaths_direct": self._parse_int(event.get("deaths_direct")),
                    "deaths_indirect": self._parse_int(event.get("deaths_indirect")),
                    "damage_property": self._parse_float(event.get("damage_property")),
                    "damage_crops": self._parse_float(event.get("damage_crops")),
                    "magnitude": self._parse_float(event.get("magnitude")),
                    "magnitude_type": event.get("magnitude_type"),
                    "impact_score": impact_score,
                    "last_updated": datetime.now().isoformat()
                }
                transformed_events.append(transformed_event)
            
            # Filter for economically significant events
            significant_events = [
                event for event in transformed_events
                if event["impact_score"] >= 2 or event["damage_property"] and event["damage_property"] > 1000000
            ]
            
            return significant_events
            
        except Exception as e:
            logger.error(f"Error fetching NOAA storm events: {str(e)}")
            return None
    
    async def _fetch_climate_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch climate data for trend analysis.
        
        Returns:
            List of climate records or None if failed
        """
        try:
            # Get recent climate data for major economic centers
            climate_stations = [
                "GHCND:USW00094728",  # New York City
                "GHCND:USW00093721",  # Los Angeles
                "GHCND:USW00094846",  # Chicago
                "GHCND:USW00012960",  # Houston
                "GHCND:USW00013874"   # Atlanta
            ]
            
            climate_data = []
            for station_id in climate_stations:
                try:
                    station_data = await self.noaa_connector.get_climate_data(
                        station_id=station_id,
                        start_date=datetime.now() - timedelta(days=7),
                        end_date=datetime.now()
                    )
                    
                    if station_data:
                        for record in station_data:
                            transformed_record = {
                                "source": "noaa",
                                "data_type": "climate_data",
                                "station_id": station_id,
                                "date": record.get("date"),
                                "temperature_max": self._parse_float(record.get("TMAX")),
                                "temperature_min": self._parse_float(record.get("TMIN")),
                                "precipitation": self._parse_float(record.get("PRCP")),
                                "snowfall": self._parse_float(record.get("SNOW")),
                                "wind_speed": self._parse_float(record.get("AWND")),
                                "last_updated": datetime.now().isoformat()
                            }
                            climate_data.append(transformed_record)
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching climate data for {station_id}: {str(e)}")
                    continue
            
            return climate_data if climate_data else None
            
        except Exception as e:
            logger.error(f"Error fetching NOAA climate data: {str(e)}")
            return None
    
    def _calculate_alert_risk_score(self, properties: Dict[str, Any]) -> int:
        """
        Calculate risk score for weather alert.
        
        Args:
            properties: Alert properties
            
        Returns:
            Risk score (1-5)
        """
        try:
            severity = properties.get("severity", "unknown").lower()
            urgency = properties.get("urgency", "unknown").lower()
            certainty = properties.get("certainty", "unknown").lower()
            
            # Base score from severity
            score = self.severity_scores.get(severity, 1)
            
            # Adjust for urgency
            if urgency == "immediate":
                score += 1
            elif urgency == "expected":
                score += 0.5
            
            # Adjust for certainty
            if certainty == "observed":
                score += 1
            elif certainty == "likely":
                score += 0.5
            
            return min(int(score), 5)
            
        except Exception:
            return 1
    
    def _calculate_economic_impact(self, event: Dict[str, Any]) -> int:
        """
        Calculate economic impact score for storm event.
        
        Args:
            event: Storm event data
            
        Returns:
            Impact score (1-5)
        """
        try:
            property_damage = self._parse_float(event.get("damage_property", 0))
            crop_damage = self._parse_float(event.get("damage_crops", 0))
            total_damage = (property_damage or 0) + (crop_damage or 0)
            
            deaths = (self._parse_int(event.get("deaths_direct", 0)) or 0) + \
                    (self._parse_int(event.get("deaths_indirect", 0)) or 0)
            
            injuries = (self._parse_int(event.get("injuries_direct", 0)) or 0) + \
                      (self._parse_int(event.get("injuries_indirect", 0)) or 0)
            
            # Calculate score based on damage and casualties
            score = 1
            
            if total_damage > 100000000:  # $100M+
                score = 5
            elif total_damage > 10000000:  # $10M+
                score = 4
            elif total_damage > 1000000:   # $1M+
                score = 3
            elif total_damage > 100000:    # $100K+
                score = 2
            
            # Adjust for casualties
            if deaths > 0:
                score = min(score + 2, 5)
            elif injuries > 10:
                score = min(score + 1, 5)
            
            return score
            
        except Exception:
            return 1
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse integer value."""
        try:
            return int(float(value)) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse float value."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    async def _cache_data(self, cache_key: str, data: List[Dict[str, Any]]) -> bool:
        """
        Cache fetched data.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            
        Returns:
            True if cached successfully
        """
        try:
            # Cache for 2 hours (weather data changes frequently)
            return self.cache_manager.set(
                cache_key, 
                data, 
                ttl=2 * 3600,
                persist_to_postgres=True
            )
        except Exception as e:
            logger.error(f"Error caching NOAA data: {str(e)}")
            return False
    
    async def _get_fallback_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get fallback data from cache or fallback handler.
        
        Returns:
            Fallback data or None
        """
        try:
            # Try to get recent cached data
            now = datetime.now()
            for hours_ago in range(1, 25):  # Try last 24 hours
                cache_time = (now - timedelta(hours=hours_ago)).strftime('%Y%m%d_%H')
                cache_key = f"noaa:latest_fetch:{cache_time}"
                
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using NOAA fallback data from {cache_time}")
                    return cached_data
            
            # Try fallback handler
            fallback_data = self.cache_manager.fallback_handler.get_fallback_data("noaa")
            if fallback_data:
                return fallback_data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting NOAA fallback data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on NOAA data source.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check if we can reach NOAA API
            health_result = await self.noaa_connector.health_check()
            
            # Check cache availability
            cache_available = self.cache_manager.exists("noaa:latest_fetch")
            
            # Check fallback data
            fallback_available = bool(await self._get_fallback_data())
            
            overall_healthy = (
                health_result.get("api_available", False) or 
                cache_available or 
                fallback_available
            )
            
            return {
                "overall_healthy": overall_healthy,
                "api_available": health_result.get("api_available", False),
                "cache_available": cache_available,
                "fallback_available": fallback_available,
                "last_successful_fetch": health_result.get("last_successful_fetch"),
                "error_count": health_result.get("error_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"NOAA health check failed: {str(e)}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test function for NOAA events fetcher."""
    fetcher = NoaaEventsFetcher()
    
    print("Testing NOAA events fetcher...")
    
    # Test health check
    health = await fetcher.health_check()
    print(f"Health check: {health}")
    
    # Test data fetch
    data = await fetcher.fetch_latest_events()
    if data:
        print(f"Fetched {len(data)} NOAA records")
        print(f"Sample record: {data[0] if data else 'None'}")
    else:
        print("No data retrieved")


if __name__ == "__main__":
    asyncio.run(main())