"""
World Bank WITS API Integration for Trade Flows and Country Risk
Replaces S&P Global supply chain stress monitoring with free World Bank data
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.core.cache import cache_with_fallback, CacheConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

# World Bank APIs
WITS_API_BASE = "https://wits.worldbank.org/api/v1"
WB_DATA_API = "https://api.worldbank.org/v2"
COUNTRY_API = f"{WB_DATA_API}/country"
INDICATORS_API = f"{WB_DATA_API}/country/all/indicator"

class WorldBankWITSIntegration:
    """World Bank WITS API integration for trade and country risk analysis"""
    
    def __init__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def get_trade_flows(self, countries: List[str], year: int = 2021) -> Dict[str, Any]:
        """Get trade flows using World Bank Data API"""
        try:
            # Get merchandise trade as % of GDP for multiple countries
            country_codes = ";".join(countries)
            
            # Get trade data (imports + exports as % of GDP)
            trade_url = f"{WB_DATA_API}/country/{country_codes}/indicator/TG.VAL.TOTL.GD.ZS"
            params = {
                "format": "json",
                "date": str(year)
            }
            
            response = await self.session.get(trade_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    trade_data = {}
                    for item in data[1]:
                        if item['value'] is not None:
                            country_name = item['country']['value']
                            country_code = item['countryiso3code']
                            trade_percentage = item['value']
                            
                            trade_data[country_code] = {
                                "country_name": country_name,
                                "trade_percentage_gdp": trade_percentage,
                                "year": year,
                                "last_updated": item.get('date', str(year))
                            }
                    
                    logger.info(f"Retrieved trade data for {len(trade_data)} countries")
                    return trade_data
                else:
                    logger.warning("No trade data found in World Bank response")
                    return {}
            else:
                logger.warning(f"World Bank API returned status {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to fetch trade flows: {e}")
            return {}
    
    async def get_country_indicators(self, country_code: str = "USA") -> Optional[Dict[str, Any]]:
        """Get country economic indicators from World Bank Data API"""
        try:
            # Key economic indicators for country risk assessment
            indicators = {
                "NY.GDP.MKTP.CD": "GDP (current US$)",
                "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
                "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
                "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
                "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
                "DT.DOD.DECT.CD": "Total debt service (current US$)",
                "IC.BUS.EASE.XQ": "Ease of doing business score"
            }
            
            results = {}
            for indicator_code, indicator_name in indicators.items():
                try:
                    url = f"{INDICATORS_API}/{indicator_code}"
                    params = {
                        "country": country_code,
                        "format": "json",
                        "date": "2020:2023",  # Last 4 years
                        "per_page": "4"
                    }
                    
                    response = await self.session.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 1 and data[1]:  # WB API returns metadata in first element
                            results[indicator_code] = {
                                "name": indicator_name,
                                "values": data[1]
                            }
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch indicator {indicator_code}: {e}")
                    continue
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"Failed to fetch country indicators: {e}")
            return None
    
    async def get_global_trade_overview(self) -> Dict[str, Any]:
        """Get global trade overview and stress indicators using real World Bank data"""
        try:
            # Major trading economies
            major_economies = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "ITA"]
            
            # Get trade flows for all countries in one call
            trade_data = await self.get_trade_flows(major_economies, year=2021)
            
            country_risks = {}
            
            # Get country risk indicators for each country
            for country in major_economies:
                indicators = await self.get_country_indicators(country)
                if indicators:
                    country_risks[country] = self.calculate_country_risk_score(indicators)
                else:
                    # If no indicators available, use trade data to estimate basic risk
                    trade_info = trade_data.get(country)
                    if trade_info:
                        trade_pct = trade_info.get("trade_percentage_gdp", 50)
                        # Higher trade dependency = higher risk in volatile times
                        base_risk = 50 + (trade_pct - 50) * 0.3  # Adjust based on trade openness
                        country_risks[country] = {
                            "risk_score": max(0, min(100, base_risk)),
                            "risk_level": self.get_risk_level(base_risk),
                            "risk_factors": {"trade_openness": trade_pct},
                            "last_updated": datetime.utcnow().isoformat()
                        }
                
                # Small delay to be respectful to the API
                await asyncio.sleep(0.1)
            
            # Calculate global stress indicators
            stress_score = self.calculate_global_stress_score(country_risks)
            
            return {
                "global_stress_score": stress_score,
                "country_risks": country_risks,
                "trade_data": trade_data,
                "data_source": "World Bank Data API",
                "last_updated": datetime.utcnow().isoformat(),
                "countries_analyzed": len(country_risks)
            }
            
        except Exception as e:
            logger.error(f"Failed to get global trade overview: {e}")
            # Return empty data instead of fallback - let the caller handle it
            return {
                "global_stress_score": 0,
                "country_risks": {},
                "trade_data": {},
                "data_source": "World Bank Data API (error)",
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat(),
                "countries_analyzed": 0
            }
    
    def calculate_country_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate country risk score from economic indicators"""
        try:
            risk_score = 50  # Base score
            risk_factors = {}
            
            # GDP Growth (±15 points)
            gdp_growth = self.get_latest_value(indicators.get("NY.GDP.MKTP.KD.ZG", {}).get("values", []))
            if gdp_growth is not None:
                risk_factors["gdp_growth"] = gdp_growth
                if gdp_growth > 3:
                    risk_score += 15
                elif gdp_growth > 1:
                    risk_score += 5
                elif gdp_growth < -2:
                    risk_score -= 15
                elif gdp_growth < 0:
                    risk_score -= 8
            
            # Inflation (±10 points)
            inflation = self.get_latest_value(indicators.get("FP.CPI.TOTL.ZG", {}).get("values", []))
            if inflation is not None:
                risk_factors["inflation"] = inflation
                if inflation < 2:
                    risk_score += 5
                elif inflation < 4:
                    risk_score += 2
                elif inflation > 10:
                    risk_score -= 10
                elif inflation > 6:
                    risk_score -= 5
            
            # Unemployment (±8 points)
            unemployment = self.get_latest_value(indicators.get("SL.UEM.TOTL.ZS", {}).get("values", []))
            if unemployment is not None:
                risk_factors["unemployment"] = unemployment
                if unemployment < 4:
                    risk_score += 8
                elif unemployment < 6:
                    risk_score += 3
                elif unemployment > 12:
                    risk_score -= 8
                elif unemployment > 8:
                    risk_score -= 4
            
            # Trade Openness (±7 points)
            trade_pct_gdp = self.get_latest_value(indicators.get("NE.TRD.GNFS.ZS", {}).get("values", []))
            if trade_pct_gdp is not None:
                risk_factors["trade_openness"] = trade_pct_gdp
                if trade_pct_gdp > 80:
                    risk_score += 7
                elif trade_pct_gdp > 40:
                    risk_score += 3
                elif trade_pct_gdp < 20:
                    risk_score -= 7
            
            # Business Environment (±10 points)
            ease_business = self.get_latest_value(indicators.get("IC.BUS.EASE.XQ", {}).get("values", []))
            if ease_business is not None:
                risk_factors["ease_of_business"] = ease_business
                if ease_business > 75:
                    risk_score += 10
                elif ease_business > 60:
                    risk_score += 5
                elif ease_business < 40:
                    risk_score -= 10
                elif ease_business < 50:
                    risk_score -= 5
            
            final_score = max(0, min(100, risk_score))
            risk_level = self.get_risk_level(final_score)
            
            return {
                "risk_score": final_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate country risk: {e}")
            return {
                "risk_score": 50,
                "risk_level": "medium",
                "risk_factors": {},
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def get_latest_value(self, values: List[Dict]) -> Optional[float]:
        """Extract the most recent value from World Bank indicator data"""
        if not values:
            return None
        
        # World Bank data is sorted by date, get the first non-null value
        for item in values:
            if item.get("value") is not None:
                return float(item["value"])
        
        return None
    
    def calculate_global_stress_score(self, country_risks: Dict[str, Dict]) -> float:
        """Calculate global stress score from country risk data"""
        if not country_risks:
            return 50.0
        
        # Weight by economic importance (simplified)
        weights = {
            "USA": 0.25,
            "CHN": 0.20,
            "DEU": 0.15,
            "JPN": 0.15,
            "GBR": 0.10,
            "FRA": 0.08,
            "IND": 0.07
        }
        
        weighted_score = 0
        total_weight = 0
        
        for country, risk_data in country_risks.items():
            weight = weights.get(country, 0.05)
            score = risk_data.get("risk_score", 50)
            weighted_score += score * weight
            total_weight += weight
        
        return round(weighted_score / total_weight if total_weight > 0 else 50.0, 1)
    
    def get_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level"""
        if score >= 75:
            return "low"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "high"
        else:
            return "critical"
    
    async def build_supply_chain_network(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build supply chain network nodes and edges from real World Bank trade data"""
        try:
            trade_overview = await self.get_global_trade_overview()
            
            if trade_overview.get("countries_analyzed", 0) == 0:
                raise Exception("No trade data available from World Bank API")
            
            # Create nodes from real data
            nodes = []
            country_risks = trade_overview.get("country_risks", {})
            trade_data = trade_overview.get("trade_data", {})
            
            country_info = {
                "USA": {"name": "United States", "region": "North America"},
                "CHN": {"name": "China", "region": "Asia"},
                "DEU": {"name": "Germany", "region": "Europe"},
                "JPN": {"name": "Japan", "region": "Asia"},
                "GBR": {"name": "United Kingdom", "region": "Europe"},
                "FRA": {"name": "France", "region": "Europe"},
                "IND": {"name": "India", "region": "Asia"},
                "ITA": {"name": "Italy", "region": "Europe"}
            }
            
            for code, info in country_info.items():
                risk_data = country_risks.get(code, {})
                trade_info = trade_data.get(code, {})
                
                node = {
                    "id": code,
                    "name": trade_info.get("country_name", info["name"]),
                    "region": info["region"],
                    "trade_percentage_gdp": trade_info.get("trade_percentage_gdp", 0),
                    "risk_score": risk_data.get("risk_score", 50),
                    "risk_level": risk_data.get("risk_level", "medium"),
                    "type": "country",
                    "data_year": trade_info.get("year", 2021),
                    "has_real_data": code in trade_data
                }
                nodes.append(node)
            
            # Create edges based on real trade relationships and risk factors
            edges = []
            
            # Calculate trade relationships based on trade openness and risk scores
            for i, source_node in enumerate(nodes):
                for j, target_node in enumerate(nodes):
                    if i != j:  # Don't connect country to itself
                        source_code = source_node["id"]
                        target_code = target_node["id"]
                        
                        # Calculate trade relationship strength based on real data
                        source_trade = source_node.get("trade_percentage_gdp", 0)
                        target_trade = target_node.get("trade_percentage_gdp", 0)
                        
                        # Only create edges for significant trade relationships
                        if source_trade > 20 and target_trade > 20:
                            # Estimate trade value based on GDP trade percentage
                            # This is a simplification - in reality you'd need bilateral trade data
                            trade_strength = min(source_trade, target_trade) * 10000000000  # Scale to billions
                            
                            # Calculate combined risk
                            avg_risk = (source_node["risk_score"] + target_node["risk_score"]) / 2
                            risk_level = "low" if avg_risk > 70 else "medium" if avg_risk > 50 else "high"
                            
                            edge = {
                                "source": source_code,
                                "target": target_code,
                                "type": "trade",
                                "value": trade_strength,
                                "weight": min(trade_strength / 1e9, 50),  # Normalize for visualization
                                "risk_level": risk_level,
                                "risk_score": avg_risk,
                                "source_trade_pct": source_trade,
                                "target_trade_pct": target_trade
                            }
                            edges.append(edge)
            
            # Update node metrics based on created edges
            for node in nodes:
                node_id = node["id"]
                outgoing_edges = [e for e in edges if e["source"] == node_id]
                incoming_edges = [e for e in edges if e["target"] == node_id]
                
                total_outgoing = sum(e["value"] for e in outgoing_edges)
                total_incoming = sum(e["value"] for e in incoming_edges)
                total_trade = total_outgoing + total_incoming
                
                node["trade_volume"] = total_trade
                node["export_dependence"] = total_outgoing / total_trade if total_trade > 0 else 0
                node["import_dependence"] = total_incoming / total_trade if total_trade > 0 else 0
                node["trade_partners"] = len(outgoing_edges) + len(incoming_edges)
            
            logger.info(f"Built real data supply chain network: {len(nodes)} nodes, {len(edges)} edges")
            logger.info(f"Countries with real data: {sum(1 for n in nodes if n['has_real_data'])}")
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Failed to build supply chain network from real data: {e}")
            raise Exception(f"Could not build network from World Bank data: {e}")
    
    def get_fallback_trade_data(self) -> Dict[str, Any]:
        """Return fallback data when World Bank API fails"""
        return {
            "global_stress_score": 62.5,
            "country_risks": {
                "USA": {"risk_score": 75, "risk_level": "low"},
                "CHN": {"risk_score": 55, "risk_level": "medium"},
                "DEU": {"risk_score": 70, "risk_level": "low"},
                "JPN": {"risk_score": 65, "risk_level": "medium"},
                "GBR": {"risk_score": 60, "risk_level": "medium"}
            },
            "trade_data": {},
            "data_source": "World Bank WITS (fallback)",
            "last_updated": datetime.utcnow().isoformat()
        }

# Global instance
wb_wits = WorldBankWITSIntegration()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="wb_wits_trade",
        ttl_seconds=3600,  # 1 hour cache
        fallback_ttl_seconds=43200  # 12 hour fallback
    )
)
async def get_trade_intelligence() -> Dict[str, Any]:
    """Get trade intelligence and supply chain stress monitoring"""
    return await wb_wits.get_global_trade_overview()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="wb_country_risk",
        ttl_seconds=7200,  # 2 hour cache
        fallback_ttl_seconds=86400  # 24 hour fallback
    )
)
async def get_country_risk_assessment(country_code: str = "USA") -> Dict[str, Any]:
    """Get detailed country risk assessment"""
    indicators = await wb_wits.get_country_indicators(country_code)
    if indicators:
        risk_assessment = wb_wits.calculate_country_risk_score(indicators)
        risk_assessment["country_code"] = country_code
        risk_assessment["data_source"] = "World Bank"
        return risk_assessment
    else:
        raise Exception(f"Could not get country indicators for {country_code}")

def get_wits_integration():
    """Get the global WITS integration instance"""
    return wb_wits

async def cleanup_worldbank():
    """Cleanup World Bank session"""
    if hasattr(wb_wits, 'session'):
        await wb_wits.session.aclose()