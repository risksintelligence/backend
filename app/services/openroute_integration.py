"""
OpenStreetMap + OpenRouteService Integration for Supply Chain Mapping
Replaces S&P Global geographic risk assessment with free mapping services
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from app.core.cache import cache_with_fallback, CacheConfig
from app.core.config import settings
import json
import math

logger = logging.getLogger(__name__)

# OpenRouteService API (free tier: 2000 requests/day)
ORS_API_BASE = "https://api.openrouteservice.org"
ORS_API_KEY = settings.OPENROUTE_API_KEY if hasattr(settings, 'OPENROUTE_API_KEY') else None

# Nominatim for geocoding (OpenStreetMap)
NOMINATIM_API = "https://nominatim.openstreetmap.org"

class OpenRouteIntegration:
    """OpenRouteService and OpenStreetMap integration for supply chain mapping"""
    
    def __init__(self):
        headers = {
            "User-Agent": "RiskX Observatory/1.0 (contact@riskx.com)"
        }
        
        if ORS_API_KEY:
            headers["Authorization"] = ORS_API_KEY
        
        self.session = httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.rate_limit_delay = 0.1  # Rate limiting for free tier
    
    async def geocode_location(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to coordinates using Nominatim"""
        try:
            params = {
                "q": address,
                "format": "json",
                "limit": 1,
                "addressdetails": 1
            }
            
            response = await self.session.get(f"{NOMINATIM_API}/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                return (lat, lon)
            
            return None
            
        except Exception as e:
            logger.error(f"Geocoding failed for {address}: {e}")
            return None
    
    async def calculate_route(self, start_coords: Tuple[float, float], 
                            end_coords: Tuple[float, float],
                            profile: str = "driving-hgv") -> Optional[Dict[str, Any]]:
        """Calculate route between two points using OpenRouteService"""
        try:
            if not ORS_API_KEY:
                logger.warning("OpenRouteService API key not configured, using distance estimation")
                return self.estimate_route_fallback(start_coords, end_coords)
            
            url = f"{ORS_API_BASE}/v2/directions/{profile}"
            
            body = {
                "coordinates": [[start_coords[1], start_coords[0]], [end_coords[1], end_coords[0]]],
                "format": "json",
                "instructions": False,
                "geometry": True,
                "elevation": False
            }
            
            response = await self.session.post(url, json=body)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_route_data(data)
            elif response.status_code == 403:
                logger.warning("OpenRouteService API key quota exceeded, using fallback")
                return self.estimate_route_fallback(start_coords, end_coords)
            else:
                logger.warning(f"OpenRouteService returned status {response.status_code}")
                return self.estimate_route_fallback(start_coords, end_coords)
                
        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            return self.estimate_route_fallback(start_coords, end_coords)
    
    def process_route_data(self, route_data: Dict) -> Dict[str, Any]:
        """Process OpenRouteService route response"""
        try:
            routes = route_data.get("routes", [])
            if not routes:
                return None
            
            route = routes[0]
            summary = route.get("summary", {})
            
            return {
                "distance_km": round(summary.get("distance", 0) / 1000, 2),
                "duration_hours": round(summary.get("duration", 0) / 3600, 2),
                "geometry": route.get("geometry", ""),
                "data_source": "OpenRouteService"
            }
            
        except Exception as e:
            logger.error(f"Failed to process route data: {e}")
            return None
    
    def estimate_route_fallback(self, start: Tuple[float, float], end: Tuple[float, float]) -> Dict[str, Any]:
        """Fallback route estimation using haversine distance"""
        try:
            distance_km = self.haversine_distance(start, end)
            # Rough estimation: 80 km/h average speed for freight
            duration_hours = distance_km / 80
            
            return {
                "distance_km": round(distance_km, 2),
                "duration_hours": round(duration_hours, 2),
                "geometry": "",
                "data_source": "Haversine estimation"
            }
            
        except Exception as e:
            logger.error(f"Fallback route estimation failed: {e}")
            return {
                "distance_km": 0,
                "duration_hours": 0,
                "geometry": "",
                "data_source": "Fallback"
            }
    
    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two points"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    async def analyze_supply_chain_routes(self) -> Dict[str, Any]:
        """Analyze major supply chain routes and logistics risk"""
        try:
            # Major supply chain corridors
            key_routes = [
                {
                    "name": "Trans-Pacific (Shanghai to Los Angeles)",
                    "start": "Shanghai, China",
                    "end": "Los Angeles, USA",
                    "type": "maritime",
                    "importance": "high"
                },
                {
                    "name": "Trans-Atlantic (Hamburg to New York)",
                    "start": "Hamburg, Germany", 
                    "end": "New York, USA",
                    "type": "maritime",
                    "importance": "high"
                },
                {
                    "name": "European Corridor (Rotterdam to Berlin)",
                    "start": "Rotterdam, Netherlands",
                    "end": "Berlin, Germany",
                    "type": "road",
                    "importance": "medium"
                },
                {
                    "name": "NAFTA Corridor (Detroit to Mexico City)",
                    "start": "Detroit, USA",
                    "end": "Mexico City, Mexico",
                    "type": "road",
                    "importance": "medium"
                },
                {
                    "name": "Asia-Europe (Hong Kong to Frankfurt)",
                    "start": "Hong Kong",
                    "end": "Frankfurt, Germany",
                    "type": "air",
                    "importance": "high"
                }
            ]
            
            route_analysis = {}
            total_risk_score = 0
            analyzed_routes = 0
            
            for route in key_routes:
                try:
                    # Geocode start and end points
                    start_coords = await self.geocode_location(route["start"])
                    end_coords = await self.geocode_location(route["end"])
                    
                    if start_coords and end_coords:
                        # Calculate route if terrestrial
                        if route["type"] == "road":
                            route_data = await self.calculate_route(start_coords, end_coords)
                        else:
                            # For maritime/air, use great circle distance
                            route_data = self.estimate_route_fallback(start_coords, end_coords)
                        
                        if route_data:
                            # Add risk assessment
                            risk_assessment = self.assess_route_risk(route, route_data, start_coords, end_coords)
                            
                            route_analysis[route["name"]] = {
                                **route_data,
                                **risk_assessment,
                                "route_info": route,
                                "coordinates": {
                                    "start": start_coords,
                                    "end": end_coords
                                }
                            }
                            
                            total_risk_score += risk_assessment.get("risk_score", 50)
                            analyzed_routes += 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze route {route['name']}: {e}")
                    continue
            
            # Calculate overall supply chain risk
            avg_risk_score = total_risk_score / analyzed_routes if analyzed_routes > 0 else 50
            
            # Generate risk insights
            risk_insights = self.generate_risk_insights(route_analysis)
            
            return {
                "routes": route_analysis,
                "summary": {
                    "total_routes_analyzed": analyzed_routes,
                    "average_risk_score": round(avg_risk_score, 1),
                    "overall_risk_level": self.get_risk_level(avg_risk_score),
                    "total_distance_analyzed_km": sum(r.get("distance_km", 0) for r in route_analysis.values())
                },
                "risk_insights": risk_insights,
                "data_source": "OpenStreetMap + OpenRouteService",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Supply chain route analysis failed: {e}")
            return self.get_fallback_route_analysis()
    
    def assess_route_risk(self, route_info: Dict, route_data: Dict, 
                         start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Dict[str, Any]:
        """Assess risk factors for a supply chain route"""
        try:
            risk_score = 30  # Base risk score
            risk_factors = []
            
            # Distance risk (longer = higher risk)
            distance = route_data.get("distance_km", 0)
            if distance > 10000:  # Very long routes
                risk_score += 20
                risk_factors.append("Very long distance route")
            elif distance > 5000:
                risk_score += 10
                risk_factors.append("Long distance route")
            
            # Route type risk
            route_type = route_info.get("type", "road")
            if route_type == "maritime":
                risk_score += 15  # Weather, piracy, port congestion
                risk_factors.append("Maritime transport risks")
            elif route_type == "air":
                risk_score += 5   # Weather, capacity constraints
                risk_factors.append("Air transport capacity limits")
            
            # Geographic risk factors (simplified)
            start_lat, start_lon = start_coords
            end_lat, end_lon = end_coords
            
            # Crossing multiple regions increases risk
            if abs(start_lat - end_lat) > 30:  # Crossing climate zones
                risk_score += 10
                risk_factors.append("Cross-climatic transport")
            
            # Transoceanic routes
            if abs(start_lon - end_lon) > 90:  # Rough transoceanic threshold
                risk_score += 15
                risk_factors.append("Transoceanic route")
            
            # Strategic importance (from route info)
            importance = route_info.get("importance", "medium")
            if importance == "high":
                risk_score += 10  # High importance = higher scrutiny/risk
                risk_factors.append("Strategically important corridor")
            
            # Duration risk
            duration = route_data.get("duration_hours", 0)
            if duration > 240:  # >10 days
                risk_score += 10
                risk_factors.append("Extended transit time")
            
            final_risk_score = min(100, max(0, risk_score))
            
            return {
                "risk_score": final_risk_score,
                "risk_level": self.get_risk_level(final_risk_score),
                "risk_factors": risk_factors,
                "assessment_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Route risk assessment failed: {e}")
            return {
                "risk_score": 50,
                "risk_level": "medium",
                "risk_factors": ["Assessment data limited"],
                "assessment_date": datetime.utcnow().isoformat()
            }
    
    def generate_risk_insights(self, route_analysis: Dict) -> List[Dict[str, str]]:
        """Generate actionable risk insights from route analysis"""
        insights = []
        
        try:
            if not route_analysis:
                return insights
            
            # Identify highest risk routes
            high_risk_routes = [
                name for name, data in route_analysis.items()
                if data.get("risk_score", 0) > 70
            ]
            
            if high_risk_routes:
                insights.append({
                    "type": "high_risk_alert",
                    "title": "High-Risk Routes Identified",
                    "description": f"Routes with elevated risk: {', '.join(high_risk_routes[:3])}"
                })
            
            # Distance optimization opportunities
            long_routes = [
                (name, data.get("distance_km", 0))
                for name, data in route_analysis.items()
                if data.get("distance_km", 0) > 8000
            ]
            
            if long_routes:
                longest = max(long_routes, key=lambda x: x[1])
                insights.append({
                    "type": "optimization",
                    "title": "Route Optimization Opportunity",
                    "description": f"Consider alternative routing for {longest[0]} ({longest[1]:,.0f} km)"
                })
            
            # Modal diversification
            route_types = {}
            for data in route_analysis.values():
                route_type = data.get("route_info", {}).get("type", "unknown")
                route_types[route_type] = route_types.get(route_type, 0) + 1
            
            if len(route_types) < 3:
                insights.append({
                    "type": "diversification",
                    "title": "Transport Mode Diversification",
                    "description": "Consider diversifying across air, maritime, and land transport modes"
                })
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            logger.error(f"Failed to generate risk insights: {e}")
            return []
    
    def get_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level"""
        if score >= 75:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
    
    def get_fallback_route_analysis(self) -> Dict[str, Any]:
        """Return fallback data when mapping services fail"""
        return {
            "routes": {
                "Trans-Pacific": {
                    "distance_km": 11000,
                    "duration_hours": 336,  # 14 days
                    "risk_score": 65,
                    "risk_level": "high",
                    "risk_factors": ["Long maritime route", "Weather dependency"],
                    "data_source": "Fallback estimation"
                },
                "European Corridor": {
                    "distance_km": 650,
                    "duration_hours": 8,
                    "risk_score": 35,
                    "risk_level": "medium",
                    "risk_factors": ["Border crossings"],
                    "data_source": "Fallback estimation"
                }
            },
            "summary": {
                "total_routes_analyzed": 2,
                "average_risk_score": 50.0,
                "overall_risk_level": "medium",
                "total_distance_analyzed_km": 11650
            },
            "risk_insights": [
                {
                    "type": "fallback_notice",
                    "title": "Limited Analysis Available",
                    "description": "Using fallback data - connect mapping services for detailed analysis"
                }
            ],
            "data_source": "OpenStreetMap + OpenRouteService (fallback)",
            "last_updated": datetime.utcnow().isoformat()
        }

# Global instance
open_route = OpenRouteIntegration()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="openroute_supply_chain",
        ttl_seconds=7200,  # 2 hour cache (due to API limits)
        fallback_ttl_seconds=86400  # 24 hour fallback
    )
)
async def get_supply_chain_mapping() -> Dict[str, Any]:
    """Get supply chain mapping and logistics risk analysis"""
    try:
        return await open_route.analyze_supply_chain_routes()
    except Exception as e:
        logger.error(f"OpenRoute integration failed: {e}")
        return open_route.get_fallback_route_analysis()

async def cleanup_openroute():
    """Cleanup OpenRoute session"""
    if hasattr(open_route, 'session'):
        await open_route.session.aclose()