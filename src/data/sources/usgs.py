"""
U.S. Geological Survey (USGS) API Integration
"""
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)

# USGS API endpoints (all free, no authentication required)
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_REALTIME_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


class USGSClient:
    """Async client for USGS geological data with rate limiting and error handling."""
    
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
    
    async def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Make request to USGS APIs with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        await self._rate_limit()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"USGS API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"USGS API timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"USGS API error for {url}: {e}")
            return None
    
    async def get_recent_earthquakes(self, days: int = 7) -> Optional[Dict]:
        """Get recent earthquake data for risk assessment."""
        
        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": "2.5",  # Focus on significant earthquakes
            "limit": 1000
        }
        
        data = await self._make_request(USGS_EARTHQUAKE_URL, params)
        
        if data and "features" in data:
            earthquakes = data["features"]
            
            # Analyze earthquake patterns
            magnitude_distribution = {"2-3": 0, "3-4": 0, "4-5": 0, "5-6": 0, "6+": 0}
            depth_distribution = {"shallow": 0, "intermediate": 0, "deep": 0}
            regional_activity = {}
            significant_quakes = []
            
            total_energy = 0  # Cumulative seismic energy
            
            for eq in earthquakes:
                props = eq.get("properties", {})
                geometry = eq.get("geometry", {})
                
                magnitude = props.get("mag", 0)
                depth = geometry.get("coordinates", [0, 0, 0])[2] if len(geometry.get("coordinates", [])) > 2 else 0
                place = props.get("place", "Unknown")
                
                # Magnitude distribution
                if magnitude >= 6:
                    magnitude_distribution["6+"] += 1
                elif magnitude >= 5:
                    magnitude_distribution["5-6"] += 1
                elif magnitude >= 4:
                    magnitude_distribution["4-5"] += 1
                elif magnitude >= 3:
                    magnitude_distribution["3-4"] += 1
                else:
                    magnitude_distribution["2-3"] += 1
                
                # Depth distribution (km)
                if depth <= 70:
                    depth_distribution["shallow"] += 1
                elif depth <= 300:
                    depth_distribution["intermediate"] += 1
                else:
                    depth_distribution["deep"] += 1
                
                # Regional activity
                if place:
                    # Extract region from place string
                    region = place.split(",")[-1].strip() if "," in place else place
                    regional_activity[region] = regional_activity.get(region, 0) + 1
                
                # Significant earthquakes (magnitude 4.5+)
                if magnitude >= 4.5:
                    significant_quakes.append({
                        "magnitude": magnitude,
                        "place": place,
                        "time": datetime.fromtimestamp(props.get("time", 0) / 1000).isoformat(),
                        "depth": depth,
                        "coordinates": geometry.get("coordinates", [])
                    })
                
                # Calculate seismic energy (simplified formula)
                if magnitude > 0:
                    energy = 10 ** (1.5 * magnitude + 4.8)  # Joules
                    total_energy += energy
            
            # Calculate seismic risk score
            total_quakes = len(earthquakes)
            significant_count = len(significant_quakes)
            major_quakes = magnitude_distribution["6+"] + magnitude_distribution["5-6"]
            shallow_quakes = depth_distribution["shallow"]
            
            # Risk scoring
            risk_score = min(100, (
                (significant_count * 10) +
                (major_quakes * 25) +
                (shallow_quakes * 2) +
                (total_quakes * 0.5)
            ))
            
            return {
                "period_days": days,
                "total_earthquakes": total_quakes,
                "significant_earthquakes": significant_count,
                "magnitude_distribution": magnitude_distribution,
                "depth_distribution": depth_distribution,
                "regional_activity": dict(sorted(regional_activity.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]),
                "significant_events": significant_quakes[:10],  # Top 10 significant
                "total_seismic_energy": total_energy,
                "seismic_risk_score": risk_score,
                "risk_level": "Critical" if risk_score >= 75 else "High" if risk_score >= 50 else "Medium",
                "source": "usgs_earthquakes",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def get_infrastructure_vulnerability(self) -> Optional[Dict]:
        """Assess infrastructure vulnerability to geological hazards."""
        
        # Get recent earthquake data
        earthquake_data = await self.get_recent_earthquakes(30)  # Last 30 days
        
        if earthquake_data:
            # Analyze infrastructure risk based on earthquake patterns
            significant_events = earthquake_data.get("significant_events", [])
            regional_activity = earthquake_data.get("regional_activity", {})
            
            # Critical infrastructure vulnerability assessment
            infrastructure_risks = {
                "power_grid": {"risk_score": 0, "vulnerability_factors": []},
                "transportation": {"risk_score": 0, "vulnerability_factors": []},
                "water_systems": {"risk_score": 0, "vulnerability_factors": []},
                "communications": {"risk_score": 0, "vulnerability_factors": []},
                "emergency_services": {"risk_score": 0, "vulnerability_factors": []}
            }
            
            # High-risk regions (based on earthquake activity)
            high_risk_regions = ["California", "Alaska", "Nevada", "Hawaii", "Washington", 
                               "Oregon", "Idaho", "Montana", "Utah", "Wyoming"]
            
            for region, activity_count in regional_activity.items():
                region_risk = min(100, activity_count * 5)
                
                # Check if region contains high-risk keywords
                is_high_risk = any(hr_region.lower() in region.lower() for hr_region in high_risk_regions)
                
                if is_high_risk and region_risk > 20:
                    # Add risk to all infrastructure types
                    for infra_type in infrastructure_risks:
                        infrastructure_risks[infra_type]["risk_score"] += region_risk * 0.3
                        infrastructure_risks[infra_type]["vulnerability_factors"].append({
                            "factor": f"Seismic activity in {region}",
                            "risk_level": region_risk,
                            "earthquake_count": activity_count
                        })
            
            # Add magnitude-based risks
            mag_dist = earthquake_data.get("magnitude_distribution", {})
            major_quakes = mag_dist.get("6+", 0) + mag_dist.get("5-6", 0)
            
            if major_quakes > 0:
                major_risk = min(50, major_quakes * 20)
                for infra_type in infrastructure_risks:
                    infrastructure_risks[infra_type]["risk_score"] += major_risk
                    infrastructure_risks[infra_type]["vulnerability_factors"].append({
                        "factor": "Major earthquake activity",
                        "risk_level": major_risk,
                        "major_earthquake_count": major_quakes
                    })
            
            # Calculate overall infrastructure vulnerability
            avg_risk = sum(infra["risk_score"] for infra in infrastructure_risks.values()) / len(infrastructure_risks)
            
            return {
                "infrastructure_systems": infrastructure_risks,
                "overall_vulnerability": min(100, avg_risk),
                "risk_level": "Critical" if avg_risk >= 75 else "High" if avg_risk >= 50 else "Medium",
                "assessment_period": "30_days",
                "source": "usgs_infrastructure_risk",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def get_natural_hazard_assessment(self) -> Optional[Dict]:
        """Comprehensive natural hazard risk assessment."""
        
        # Get earthquake data for different time periods
        recent_quakes = await self.get_recent_earthquakes(7)   # Last week
        monthly_quakes = await self.get_recent_earthquakes(30) # Last month
        
        if recent_quakes and monthly_quakes:
            # Trend analysis
            weekly_significant = recent_quakes.get("significant_earthquakes", 0)
            monthly_significant = monthly_quakes.get("significant_earthquakes", 0)
            
            # Calculate trends
            weekly_rate = weekly_significant / 7
            monthly_rate = monthly_significant / 30
            
            trend = "increasing" if weekly_rate > monthly_rate else "decreasing" if weekly_rate < monthly_rate else "stable"
            
            # Hazard level assessment
            hazard_levels = {
                "seismic": {
                    "current_activity": recent_quakes.get("seismic_risk_score", 0),
                    "trend": trend,
                    "significant_events": weekly_significant
                },
                "volcanic": {
                    "current_activity": 0,  # Would need additional volcanic data
                    "trend": "stable",
                    "significant_events": 0
                },
                "landslide": {
                    "current_activity": min(50, weekly_significant * 5),  # Correlated with seismic activity
                    "trend": trend,
                    "significant_events": weekly_significant
                }
            }
            
            # Calculate composite natural hazard score
            composite_score = sum(hazard["current_activity"] for hazard in hazard_levels.values()) / len(hazard_levels)
            
            # Regional risk mapping
            regional_risks = monthly_quakes.get("regional_activity", {})
            top_risk_regions = dict(list(regional_risks.items())[:5])
            
            return {
                "hazard_types": hazard_levels,
                "composite_hazard_score": composite_score,
                "risk_level": "Critical" if composite_score >= 75 else "High" if composite_score >= 50 else "Medium",
                "activity_trend": trend,
                "top_risk_regions": top_risk_regions,
                "assessment_period": "7_day_analysis",
                "source": "usgs_natural_hazards",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def get_realtime_feed(self) -> Optional[Dict]:
        """Get real-time earthquake feed summary."""
        
        # Get multiple real-time feeds
        feeds = [
            "significant_hour.geojson",
            "significant_day.geojson",
            "significant_week.geojson"
        ]
        
        feed_data = {}
        
        for feed in feeds:
            url = f"{USGS_REALTIME_URL}/{feed}"
            data = await self._make_request(url)
            
            if data and "features" in data:
                period = feed.split("_")[1].split(".")[0]  # Extract time period
                earthquakes = data["features"]
                
                # Count by magnitude
                mag_counts = {"3-4": 0, "4-5": 0, "5-6": 0, "6+": 0}
                for eq in earthquakes:
                    mag = eq.get("properties", {}).get("mag", 0)
                    if mag >= 6:
                        mag_counts["6+"] += 1
                    elif mag >= 5:
                        mag_counts["5-6"] += 1
                    elif mag >= 4:
                        mag_counts["4-5"] += 1
                    elif mag >= 3:
                        mag_counts["3-4"] += 1
                
                feed_data[period] = {
                    "total_significant": len(earthquakes),
                    "magnitude_breakdown": mag_counts,
                    "latest_event": earthquakes[0] if earthquakes else None
                }
        
        if feed_data:
            # Calculate real-time risk
            hour_significant = feed_data.get("hour", {}).get("total_significant", 0)
            day_significant = feed_data.get("day", {}).get("total_significant", 0)
            week_significant = feed_data.get("week", {}).get("total_significant", 0)
            
            realtime_risk = min(100, (hour_significant * 20) + (day_significant * 5) + (week_significant * 1))
            
            return {
                "realtime_feeds": feed_data,
                "realtime_risk_score": realtime_risk,
                "risk_level": "Critical" if realtime_risk >= 75 else "High" if realtime_risk >= 50 else "Medium",
                "source": "usgs_realtime",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None


async def get_geological_hazards() -> Dict[str, Any]:
    """Get comprehensive geological hazard assessment."""
    
    async with USGSClient() as client:
        # Fetch multiple geological indicators concurrently
        results = await asyncio.gather(
            client.get_recent_earthquakes(7),
            client.get_infrastructure_vulnerability(),
            client.get_natural_hazard_assessment(),
            client.get_realtime_feed(),
            return_exceptions=True
        )
        
        indicators = {}
        indicator_names = ["recent_earthquakes", "infrastructure_vulnerability", 
                          "natural_hazards", "realtime_activity"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[indicator_names[i]] = result
        
        # Calculate overall geological risk score
        overall_risk = 0
        if indicators:
            seismic_risk = indicators.get("recent_earthquakes", {}).get("seismic_risk_score", 0)
            infra_risk = indicators.get("infrastructure_vulnerability", {}).get("overall_vulnerability", 0)
            hazard_risk = indicators.get("natural_hazards", {}).get("composite_hazard_score", 0)
            realtime_risk = indicators.get("realtime_activity", {}).get("realtime_risk_score", 0)
            
            overall_risk = (seismic_risk + infra_risk + hazard_risk + realtime_risk) / 4
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "overall_geological_risk": overall_risk,
            "risk_level": "Critical" if overall_risk >= 75 else "High" if overall_risk >= 50 else "Medium",
            "source": "usgs",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if USGS APIs are accessible."""
    try:
        async with USGSClient() as client:
            # Try to get recent earthquake data
            result = await client.get_recent_earthquakes(1)
            return result is not None
    except Exception:
        return False