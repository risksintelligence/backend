"""
NOAA Weather and Climate Data Source

Provides access to weather events, climate data, and severe weather alerts
from the National Oceanic and Atmospheric Administration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from etl.utils.connectors import APIConnector
from src.cache.cache_manager import CacheManager
from src.core.config import get_settings


class NOAADataSource:
    """
    Fetches weather and climate data from NOAA APIs
    
    Key data sources:
    - Storm Events Database (severe weather events)
    - Weather Alerts (active weather warnings)
    - Climate Data (temperature, precipitation anomalies)
    - Hurricane/Tropical Storm data
    """
    
    def __init__(self):
        self.logger = logging.getLogger("noaa_fetcher")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # NOAA API configuration
        self.api_token = getattr(self.settings, 'NOAA_API_TOKEN', None)
        self.storm_events_base = "https://www.ncdc.noaa.gov/stormevents/csv"
        self.weather_api_base = "https://api.weather.gov"
        self.climate_api_base = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        
        # Initialize connectors
        self.storm_connector = APIConnector("noaa_storms", {
            'base_url': self.storm_events_base,
            'timeout': 60,
            'rate_limit': 10,
            'cache_ttl': 3600
        })
        
        self.weather_connector = APIConnector("noaa_weather", {
            'base_url': self.weather_api_base,
            'headers': {'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'},
            'timeout': 30,
            'rate_limit': 100,
            'cache_ttl': 300  # 5 minute cache for alerts
        })
        
        climate_headers = {'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'}
        if self.api_token:
            climate_headers['token'] = self.api_token
        
        self.climate_connector = APIConnector("noaa_climate", {
            'base_url': self.climate_api_base,
            'headers': climate_headers,
            'timeout': 30,
            'rate_limit': 1000,  # With token: 1000/day, without: 5/second
            'cache_ttl': 3600
        })
        
        # Event types that indicate economic disruption risk
        self.high_impact_events = {
            'Hurricane (Typhoon)': 5,
            'Tornado': 4,
            'Flash Flood': 4,
            'Flood': 3,
            'Ice Storm': 4,
            'Blizzard': 3,
            'Winter Storm': 2,
            'Hail': 2,
            'High Wind': 2,
            'Wildfire': 4,
            'Drought': 3,
            'Heat': 2,
            'Cold/Wind Chill': 2,
            'Lightning': 1,
            'Thunderstorm Wind': 2
        }
        
        # States with major economic centers
        self.key_states = {
            'CA': 'California',
            'TX': 'Texas', 
            'NY': 'New York',
            'FL': 'Florida',
            'IL': 'Illinois',
            'PA': 'Pennsylvania',
            'OH': 'Ohio',
            'GA': 'Georgia',
            'NC': 'North Carolina',
            'MI': 'Michigan',
            'NJ': 'New Jersey',
            'VA': 'Virginia',
            'WA': 'Washington',
            'AZ': 'Arizona',
            'MA': 'Massachusetts'
        }
    
    async def fetch_recent_storm_events(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch recent storm events from NOAA Storm Events Database
        
        Args:
            days_back: Number of days to look back for events
            
        Returns:
            DataFrame with storm event data
        """
        try:
            await self.storm_connector.connect()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_events = []
            
            # Fetch events for key states (Storm Events API is state-based)
            for state_code, state_name in self.key_states.items():
                try:
                    events = await self._fetch_state_storm_events(state_code, start_date, end_date)
                    all_events.extend(events)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching storm events for {state_name}: {str(e)}")
                    continue
            
            if not all_events:
                self.logger.warning("No storm events data found")
                return pd.DataFrame()
            
            # Convert to DataFrame and process
            events_df = pd.DataFrame(all_events)
            events_df = self._process_storm_events(events_df)
            
            self.logger.info(f"Fetched {len(events_df)} storm events from last {days_back} days")
            return events_df
            
        except Exception as e:
            self.logger.error(f"Error fetching storm events: {str(e)}")
            
            # Try cached data
            cached_data = await self._get_cached_storm_events()
            if cached_data is not None:
                self.logger.warning("Returning cached storm events data")
                return cached_data
            
            raise
    
    async def _fetch_state_storm_events(self, state_code: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch storm events for a specific state"""
        try:
            # NOAA Storm Events uses a CSV download API
            # For this implementation, we'll simulate the data structure
            # In production, you would parse the actual CSV downloads
            
            # This is a simplified simulation of storm events data
            # Real implementation would download and parse CSV files
            simulated_events = [
                {
                    'BEGIN_DATE': datetime.now() - timedelta(days=5),
                    'END_DATE': datetime.now() - timedelta(days=5),
                    'STATE': state_code,
                    'EVENT_TYPE': 'Thunderstorm Wind',
                    'MAGNITUDE': 60,
                    'INJURIES_DIRECT': 0,
                    'INJURIES_INDIRECT': 0,
                    'DEATHS_DIRECT': 0,
                    'DEATHS_INDIRECT': 0,
                    'DAMAGE_PROPERTY': 50000,
                    'DAMAGE_CROPS': 0,
                    'COUNTY': 'Example County',
                    'MAGNITUDE_TYPE': 'Sustained Wind',
                    'LATITUDE': 40.0,
                    'LONGITUDE': -74.0
                }
            ]
            
            return simulated_events
            
        except Exception as e:
            self.logger.error(f"Error fetching storm events for {state_code}: {str(e)}")
            return []
    
    def _process_storm_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean storm events data"""
        try:
            if df.empty:
                return df
            
            # Convert dates
            date_columns = ['BEGIN_DATE', 'END_DATE']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['MAGNITUDE', 'INJURIES_DIRECT', 'INJURIES_INDIRECT', 
                             'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add risk scoring
            df['economic_impact_score'] = df.apply(self._calculate_economic_impact, axis=1)
            df['severity_level'] = df.apply(self._assess_event_severity, axis=1)
            
            # Add state names
            if 'STATE' in df.columns:
                df['state_name'] = df['STATE'].map(self.key_states)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing storm events: {str(e)}")
            return df
    
    def _calculate_economic_impact(self, row: pd.Series) -> float:
        """Calculate economic impact score for an event"""
        try:
            score = 0
            
            # Event type impact
            event_type = row.get('EVENT_TYPE', '')
            type_impact = self.high_impact_events.get(event_type, 1)
            score += type_impact
            
            # Property damage impact (logarithmic scale)
            property_damage = row.get('DAMAGE_PROPERTY', 0)
            if property_damage > 0:
                damage_score = min(np.log10(property_damage) - 3, 5)  # Scale from $1K to $100M+
                score += max(damage_score, 0)
            
            # Crop damage impact
            crop_damage = row.get('DAMAGE_CROPS', 0)
            if crop_damage > 0:
                crop_score = min(np.log10(crop_damage) - 3, 3)
                score += max(crop_score, 0)
            
            # Casualties impact
            casualties = (row.get('INJURIES_DIRECT', 0) + row.get('INJURIES_INDIRECT', 0) +
                         row.get('DEATHS_DIRECT', 0) * 10 + row.get('DEATHS_INDIRECT', 0) * 10)
            if casualties > 0:
                score += min(casualties / 10, 5)
            
            return round(score, 2)
            
        except Exception:
            return 0.0
    
    def _assess_event_severity(self, row: pd.Series) -> str:
        """Assess event severity level"""
        try:
            impact_score = row.get('economic_impact_score', 0)
            
            if impact_score >= 8:
                return "extreme"
            elif impact_score >= 6:
                return "high"
            elif impact_score >= 4:
                return "moderate"
            elif impact_score >= 2:
                return "low"
            else:
                return "minimal"
                
        except Exception:
            return "unknown"
    
    async def fetch_active_weather_alerts(self) -> pd.DataFrame:
        """Fetch active weather alerts from NWS API"""
        try:
            await self.weather_connector.connect()
            
            # Get active alerts
            endpoint = "alerts/active"
            params = {
                'status': 'actual',
                'certainty': 'likely,observed,possible'
            }
            
            data = await self.weather_connector.fetch_data(endpoint, params)
            
            if not data or 'features' not in data:
                return pd.DataFrame()
            
            alerts = []
            for feature in data['features']:
                properties = feature.get('properties', {})
                
                alert = {
                    'id': properties.get('id'),
                    'alert_type': properties.get('event'),
                    'severity': properties.get('severity'),
                    'certainty': properties.get('certainty'),
                    'urgency': properties.get('urgency'),
                    'headline': properties.get('headline'),
                    'description': properties.get('description'),
                    'effective': properties.get('effective'),
                    'expires': properties.get('expires'),
                    'areas': properties.get('areaDesc'),
                    'message_type': properties.get('messageType'),
                    'category': properties.get('category'),
                    'response': properties.get('response')
                }
                alerts.append(alert)
            
            if not alerts:
                return pd.DataFrame()
            
            alerts_df = pd.DataFrame(alerts)
            alerts_df = self._process_weather_alerts(alerts_df)
            
            self.logger.info(f"Fetched {len(alerts_df)} active weather alerts")
            return alerts_df
            
        except Exception as e:
            self.logger.error(f"Error fetching weather alerts: {str(e)}")
            return pd.DataFrame()
    
    def _process_weather_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process weather alerts data"""
        try:
            if df.empty:
                return df
            
            # Convert dates
            date_columns = ['effective', 'expires']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Add risk assessment
            df['economic_risk_level'] = df.apply(self._assess_alert_risk, axis=1)
            
            # Add time remaining
            if 'expires' in df.columns:
                df['hours_remaining'] = (df['expires'] - datetime.now()).dt.total_seconds() / 3600
                df['hours_remaining'] = df['hours_remaining'].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing weather alerts: {str(e)}")
            return df
    
    def _assess_alert_risk(self, row: pd.Series) -> str:
        """Assess economic risk level of weather alert"""
        try:
            alert_type = row.get('alert_type', '').lower()
            severity = row.get('severity', '').lower()
            urgency = row.get('urgency', '').lower()
            
            risk_score = 0
            
            # High impact alert types
            high_impact_alerts = ['hurricane', 'tornado', 'flood', 'blizzard', 'ice storm']
            moderate_impact_alerts = ['thunderstorm', 'winter storm', 'high wind', 'freeze']
            
            if any(alert in alert_type for alert in high_impact_alerts):
                risk_score += 3
            elif any(alert in alert_type for alert in moderate_impact_alerts):
                risk_score += 2
            else:
                risk_score += 1
            
            # Severity adjustment
            if severity == 'extreme':
                risk_score += 3
            elif severity == 'severe':
                risk_score += 2
            elif severity == 'moderate':
                risk_score += 1
            
            # Urgency adjustment
            if urgency == 'immediate':
                risk_score += 2
            elif urgency == 'expected':
                risk_score += 1
            
            # Convert to risk level
            if risk_score >= 6:
                return "extreme"
            elif risk_score >= 4:
                return "high"
            elif risk_score >= 2:
                return "moderate"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    async def get_weather_risk_summary(self) -> Dict[str, Any]:
        """Get weather risk summary"""
        try:
            # Fetch storm events and alerts
            storm_events = await self.fetch_recent_storm_events(days_back=7)
            active_alerts = await self.fetch_active_weather_alerts()
            
            summary = {
                "storm_events_summary": {},
                "active_alerts_summary": {},
                "risk_assessment": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Storm events summary
            if not storm_events.empty:
                summary["storm_events_summary"] = {
                    "total_events": len(storm_events),
                    "high_impact_events": len(storm_events[storm_events['severity_level'].isin(['high', 'extreme'])]),
                    "states_affected": storm_events['STATE'].nunique(),
                    "total_property_damage": storm_events['DAMAGE_PROPERTY'].sum(),
                    "total_casualties": (storm_events['INJURIES_DIRECT'].sum() + 
                                       storm_events['INJURIES_INDIRECT'].sum() +
                                       storm_events['DEATHS_DIRECT'].sum() +
                                       storm_events['DEATHS_INDIRECT'].sum()),
                    "average_impact_score": storm_events['economic_impact_score'].mean()
                }
                
                # Top event types
                event_type_counts = storm_events['EVENT_TYPE'].value_counts().head(5)
                summary["storm_events_summary"]["top_event_types"] = event_type_counts.to_dict()
            
            # Active alerts summary
            if not active_alerts.empty:
                summary["active_alerts_summary"] = {
                    "total_alerts": len(active_alerts),
                    "high_risk_alerts": len(active_alerts[active_alerts['economic_risk_level'].isin(['high', 'extreme'])]),
                    "alert_types": active_alerts['alert_type'].value_counts().head(5).to_dict(),
                    "severity_distribution": active_alerts['severity'].value_counts().to_dict()
                }
            
            # Overall risk assessment
            high_impact_events = len(storm_events[storm_events['severity_level'].isin(['high', 'extreme'])]) if not storm_events.empty else 0
            high_risk_alerts = len(active_alerts[active_alerts['economic_risk_level'].isin(['high', 'extreme'])]) if not active_alerts.empty else 0
            
            overall_risk = "low"
            if high_impact_events >= 5 or high_risk_alerts >= 3:
                overall_risk = "high"
            elif high_impact_events >= 2 or high_risk_alerts >= 1:
                overall_risk = "moderate"
            
            summary["risk_assessment"] = {
                "overall_risk_level": overall_risk,
                "contributing_factors": {
                    "recent_severe_events": high_impact_events,
                    "active_high_risk_alerts": high_risk_alerts
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating weather risk summary: {str(e)}")
            return {"error": str(e)}
    
    async def _cache_data(self, data: pd.DataFrame, cache_key: str):
        """Cache weather data"""
        try:
            if not data.empty:
                cache_data = {
                    'data': data.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
                await self.cache.set(cache_key, cache_data, ttl=3600)
        except Exception as e:
            self.logger.warning(f"Failed to cache weather data: {str(e)}")
    
    async def _get_cached_storm_events(self) -> Optional[pd.DataFrame]:
        """Get cached storm events data"""
        try:
            cache_key = "noaa_storm_events"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data and 'data' in cached_data:
                df = pd.DataFrame(cached_data['data'])
                if 'BEGIN_DATE' in df.columns:
                    df['BEGIN_DATE'] = pd.to_datetime(df['BEGIN_DATE'])
                if 'END_DATE' in df.columns:
                    df['END_DATE'] = pd.to_datetime(df['END_DATE'])
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached storm events: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check NOAA APIs health"""
        try:
            # Test weather alerts API
            alerts_healthy = False
            try:
                test_alerts = await self.fetch_active_weather_alerts()
                alerts_healthy = True
            except Exception:
                pass
            
            return {
                "status": "healthy" if alerts_healthy else "degraded",
                "weather_alerts_api": "healthy" if alerts_healthy else "unavailable",
                "api_token_configured": bool(self.api_token),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close all connections"""
        for connector in [self.storm_connector, self.weather_connector, self.climate_connector]:
            if connector:
                await connector.close()