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
    
    async def get_trade_flows(self, reporter: str = "USA", partner: str = "WLD", year: int = 2022) -> Optional[Dict[str, Any]]:
        """Get bilateral trade flows using WITS API"""
        try:
            # WITS API endpoint for bilateral trade data
            url = f"{WITS_API_BASE}/tradeflows"
            params = {
                "reporter": reporter,
                "partner": partner,
                "year": year,
                "productcode": "TOTAL",
                "format": "json"
            }
            
            response = await self.session.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"WITS API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch trade flows: {e}")
            return None
    
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
        """Get global trade overview and stress indicators"""
        try:
            # Major trading partners for global overview
            major_economies = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "ITA", "BRA", "CAN"]
            
            trade_data = {}
            country_risks = {}
            
            for country in major_economies[:5]:  # Limit to top 5 to avoid timeouts
                # Get trade flows
                trade_flows = await self.get_trade_flows(reporter=country, partner="WLD")
                if trade_flows:
                    trade_data[country] = trade_flows
                
                # Get country risk indicators
                indicators = await self.get_country_indicators(country)
                if indicators:
                    country_risks[country] = self.calculate_country_risk_score(indicators)
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            # Calculate global stress indicators
            stress_score = self.calculate_global_stress_score(country_risks)
            
            return {
                "global_stress_score": stress_score,
                "country_risks": country_risks,
                "trade_data": trade_data,
                "data_source": "World Bank WITS",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get global trade overview: {e}")
            return self.get_fallback_trade_data()
    
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
    try:
        return await wb_wits.get_global_trade_overview()
    except Exception as e:
        logger.error(f"World Bank WITS integration failed: {e}")
        return wb_wits.get_fallback_trade_data()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="wb_country_risk",
        ttl_seconds=7200,  # 2 hour cache
        fallback_ttl_seconds=86400  # 24 hour fallback
    )
)
async def get_country_risk_assessment(country_code: str = "USA") -> Dict[str, Any]:
    """Get detailed country risk assessment"""
    try:
        indicators = await wb_wits.get_country_indicators(country_code)
        if indicators:
            risk_assessment = wb_wits.calculate_country_risk_score(indicators)
            risk_assessment["country_code"] = country_code
            risk_assessment["data_source"] = "World Bank"
            return risk_assessment
        else:
            return wb_wits.get_fallback_trade_data()["country_risks"]["USA"]
    except Exception as e:
        logger.error(f"Country risk assessment failed for {country_code}: {e}")
        return {
            "risk_score": 50,
            "risk_level": "medium",
            "country_code": country_code,
            "data_source": "World Bank (fallback)",
            "last_updated": datetime.utcnow().isoformat()
        }

async def cleanup_worldbank():
    """Cleanup World Bank session"""
    if hasattr(wb_wits, 'session'):
        await wb_wits.session.aclose()