"""
National Oceanic and Atmospheric Administration (NOAA) API Integration
"""
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from src.core.config import get_settings

logger = logging.getLogger(__name__)

# NOAA Weather API endpoint (free, no API key required)
NOAA_WEATHER_URL = "https://api.weather.gov"

settings = get_settings()


class NOAAClient:
    """Async client for NOAA weather and climate data with rate limiting and error handling."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "RiskX-Platform/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    
    async def _make_weather_request(self, endpoint: str) -> Optional[Dict]:
        """Make request to NOAA Weather Service API (no auth required)."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        await self._rate_limit()
        
        url = f"{NOAA_WEATHER_URL}/{endpoint}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"NOAA Weather API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"NOAA Weather API timeout for {endpoint}")
            return None
        except Exception as e:
            logger.error(f"NOAA Weather API error for {endpoint}: {e}")
            return None
    
    async def get_severe_weather_alerts(self) -> Optional[Dict]:
        """Get current severe weather alerts for the US."""
        
        data = await self._make_weather_request("alerts/active")
        
        if data and "features" in data:
            alerts = data["features"]
            
            # Categorize alerts by severity and type
            alert_summary = {
                "total_alerts": len(alerts),
                "by_severity": {},
                "by_event": {},
                "by_urgency": {},
                "high_impact_alerts": []
            }
            
            for alert in alerts:
                properties = alert.get("properties", {})
                
                # Categorize by severity
                severity = properties.get("severity", "Unknown")
                alert_summary["by_severity"][severity] = alert_summary["by_severity"].get(severity, 0) + 1
                
                # Categorize by event type
                event = properties.get("event", "Unknown")
                alert_summary["by_event"][event] = alert_summary["by_event"].get(event, 0) + 1
                
                # Categorize by urgency
                urgency = properties.get("urgency", "Unknown")
                alert_summary["by_urgency"][urgency] = alert_summary["by_urgency"].get(urgency, 0) + 1
                
                # Identify high-impact alerts
                if severity in ["Extreme", "Severe"] or urgency in ["Immediate", "Expected"]:
                    alert_summary["high_impact_alerts"].append({
                        "event": event,
                        "severity": severity,
                        "urgency": urgency,
                        "headline": properties.get("headline", ""),
                        "area_desc": properties.get("areaDesc", ""),
                        "effective": properties.get("effective", ""),
                        "expires": properties.get("expires", "")
                    })
            
            # Calculate weather risk score
            extreme_alerts = alert_summary["by_severity"].get("Extreme", 0)
            severe_alerts = alert_summary["by_severity"].get("Severe", 0)
            immediate_alerts = alert_summary["by_urgency"].get("Immediate", 0)
            
            risk_score = min(100, (extreme_alerts * 25) + (severe_alerts * 15) + (immediate_alerts * 10))
            
            alert_summary.update({
                "weather_risk_score": risk_score,
                "risk_level": "Critical" if risk_score >= 75 else "High" if risk_score >= 50 else "Medium",
                "source": "noaa_weather_alerts",
                "last_updated": datetime.utcnow().isoformat()
            })
            
            return alert_summary
        
        return None
    
    async def get_climate_extremes(self) -> Optional[Dict]:
        """Get climate extremes and trends - using free Weather API only."""
        
        # No CDO API key required - using free Weather API only
        logger.info("NOAA CDO API requires registration - using free Weather API instead")
        return None
    
    async def get_transportation_impacts(self) -> Optional[Dict]:
        """Assess weather impacts on transportation systems."""
        
        # Get current severe weather
        weather_data = await self.get_severe_weather_alerts()
        
        if weather_data:
            alerts = weather_data.get("high_impact_alerts", [])
            
            # Analyze transportation-relevant weather events
            transport_impacts = {
                "aviation": {"risk_score": 0, "affected_events": []},
                "maritime": {"risk_score": 0, "affected_events": []},
                "highway": {"risk_score": 0, "affected_events": []},
                "rail": {"risk_score": 0, "affected_events": []}
            }
            
            # Transportation-relevant weather events
            transport_events = {
                "aviation": ["Thunderstorm", "Tornado", "High Wind", "Blizzard", "Ice Storm", "Fog"],
                "maritime": ["Hurricane", "Gale", "Storm", "High Wind", "Flood"],
                "highway": ["Snow", "Ice", "Flood", "Thunderstorm", "Tornado", "High Wind"],
                "rail": ["Flood", "High Wind", "Snow", "Ice Storm", "Thunderstorm"]
            }
            
            for alert in alerts:
                event = alert.get("event", "")
                severity = alert.get("severity", "")
                
                for transport_mode, relevant_events in transport_events.items():
                    if any(relevant_event.lower() in event.lower() for relevant_event in relevant_events):
                        impact_score = 25 if severity == "Extreme" else 15 if severity == "Severe" else 10
                        transport_impacts[transport_mode]["risk_score"] += impact_score
                        transport_impacts[transport_mode]["affected_events"].append({
                            "event": event,
                            "severity": severity,
                            "area": alert.get("area_desc", "")
                        })
            
            # Calculate overall transportation risk
            total_risk = sum(mode["risk_score"] for mode in transport_impacts.values()) / 4
            
            return {
                "transportation_modes": transport_impacts,
                "overall_transport_risk": min(100, total_risk),
                "risk_level": "Critical" if total_risk >= 75 else "High" if total_risk >= 50 else "Medium",
                "active_weather_events": len(alerts),
                "source": "noaa_transport_risk",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None


async def get_environmental_risks() -> Dict[str, Any]:
    """Get comprehensive environmental risk assessment."""
    
    async with NOAAClient() as client:
        # Fetch multiple environmental indicators concurrently
        results = await asyncio.gather(
            client.get_severe_weather_alerts(),
            client.get_climate_extremes(),
            client.get_transportation_impacts(),
            return_exceptions=True
        )
        
        indicators = {}
        indicator_names = ["severe_weather", "climate_extremes", "transportation_impacts"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[indicator_names[i]] = result
        
        # Calculate overall environmental risk score
        overall_risk = 0
        if indicators:
            weather_risk = indicators.get("severe_weather", {}).get("weather_risk_score", 0)
            climate_risk = indicators.get("climate_extremes", {}).get("climate_risk_score", 0)
            transport_risk = indicators.get("transportation_impacts", {}).get("overall_transport_risk", 0)
            overall_risk = (weather_risk + climate_risk + transport_risk) / 3
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "overall_environmental_risk": overall_risk,
            "risk_level": "Critical" if overall_risk >= 75 else "High" if overall_risk >= 50 else "Medium",
            "source": "noaa",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if NOAA APIs are accessible."""
    try:
        async with NOAAClient() as client:
            # Try to get weather alerts (no auth required)
            result = await client.get_severe_weather_alerts()
            return result is not None
    except Exception:
        return False