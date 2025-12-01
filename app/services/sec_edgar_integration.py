"""
SEC EDGAR API Integration for Financial Health Analysis
Replaces S&P Global financial intelligence with free SEC data
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.core.cache import cache_with_fallback, CacheConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

SEC_EDGAR_BASE_URL = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_EDGAR_BASE_URL}/api/xbrl/companyfacts"
SUBMISSIONS_URL = f"{SEC_EDGAR_BASE_URL}/submissions"

# SEC requires User-Agent header for API access
SEC_HEADERS = {
    "User-Agent": "RiskX Observatory (contact@riskx.com)",
    "Accept": "application/json"
}

class SECEDGARIntegration:
    """SEC EDGAR API integration for company financial health analysis"""
    
    def __init__(self):
        self.session = httpx.AsyncClient(
            headers=SEC_HEADERS,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def get_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive financial facts for a company by CIK"""
        try:
            # Normalize CIK to 10 digits with leading zeros
            cik_padded = cik.zfill(10)
            url = f"{COMPANY_TICKERS_URL}/CIK{cik_padded}.json"
            
            response = await self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch company facts for CIK {cik}: {e}")
            return None
    
    async def get_company_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        """Get recent filings and submissions for a company"""
        try:
            cik_padded = cik.zfill(10)
            url = f"{SUBMISSIONS_URL}/CIK{cik_padded}.json"
            
            response = await self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return None
    
    def calculate_financial_health_score(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial health metrics from SEC filing data"""
        try:
            # Extract key financial metrics
            dei = facts.get("facts", {}).get("dei", {})
            us_gaap = facts.get("facts", {}).get("us-gaap", {})
            
            # Get most recent annual data
            metrics = {}
            
            # Revenue trend analysis
            revenues = self._extract_metric_values(us_gaap.get("Revenues", {}))
            if revenues:
                metrics["revenue_trend"] = self._calculate_growth_trend(revenues)
                metrics["latest_revenue"] = revenues[0]["val"] if revenues else 0
            
            # Profitability metrics
            net_income = self._extract_metric_values(us_gaap.get("NetIncomeLoss", {}))
            if net_income:
                metrics["profitability_trend"] = self._calculate_growth_trend(net_income)
                metrics["latest_net_income"] = net_income[0]["val"] if net_income else 0
            
            # Liquidity ratios
            current_assets = self._extract_metric_values(us_gaap.get("AssetsCurrent", {}))
            current_liabilities = self._extract_metric_values(us_gaap.get("LiabilitiesCurrent", {}))
            
            if current_assets and current_liabilities:
                metrics["current_ratio"] = current_assets[0]["val"] / current_liabilities[0]["val"]
            
            # Debt ratios
            total_debt = self._extract_metric_values(us_gaap.get("DebtCurrent", {}))
            total_assets = self._extract_metric_values(us_gaap.get("Assets", {}))
            
            if total_debt and total_assets:
                metrics["debt_to_assets"] = total_debt[0]["val"] / total_assets[0]["val"]
            
            # Calculate overall health score (0-100)
            health_score = self._calculate_health_score(metrics)
            
            return {
                "financial_health_score": health_score,
                "risk_level": self._get_risk_level(health_score),
                "metrics": metrics,
                "data_source": "SEC EDGAR",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate financial health: {e}")
            return self._get_fallback_health_data()
    
    def _extract_metric_values(self, metric_data: Dict) -> List[Dict]:
        """Extract and sort metric values from SEC data structure"""
        values = []
        for unit_key, unit_data in metric_data.items():
            if isinstance(unit_data, dict):
                for val_dict in unit_data.values():
                    if isinstance(val_dict, list):
                        values.extend(val_dict)
        
        # Sort by end date (most recent first)
        return sorted(values, key=lambda x: x.get("end", ""), reverse=True)[:5]
    
    def _calculate_growth_trend(self, values: List[Dict]) -> float:
        """Calculate growth trend from historical values"""
        if len(values) < 2:
            return 0.0
        
        latest = values[0]["val"]
        previous = values[1]["val"]
        
        if previous == 0:
            return 0.0
        
        return ((latest - previous) / previous) * 100
    
    def _calculate_health_score(self, metrics: Dict) -> int:
        """Calculate overall financial health score (0-100)"""
        score = 50  # Base score
        
        # Revenue trend (±20 points)
        revenue_trend = metrics.get("revenue_trend", 0)
        if revenue_trend > 10:
            score += 20
        elif revenue_trend > 0:
            score += 10
        elif revenue_trend < -10:
            score -= 20
        elif revenue_trend < 0:
            score -= 10
        
        # Profitability (±15 points)
        profit_trend = metrics.get("profitability_trend", 0)
        if profit_trend > 10:
            score += 15
        elif profit_trend > 0:
            score += 7
        elif profit_trend < -10:
            score -= 15
        
        # Liquidity (±10 points)
        current_ratio = metrics.get("current_ratio", 1.0)
        if current_ratio > 2.0:
            score += 10
        elif current_ratio > 1.5:
            score += 5
        elif current_ratio < 1.0:
            score -= 10
        
        # Debt levels (±5 points)
        debt_ratio = metrics.get("debt_to_assets", 0.3)
        if debt_ratio < 0.2:
            score += 5
        elif debt_ratio > 0.6:
            score -= 5
        
        return max(0, min(100, score))
    
    def _get_risk_level(self, score: int) -> str:
        """Convert numerical score to risk level"""
        if score >= 80:
            return "low"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "high"
        else:
            return "critical"
    
    def _get_fallback_health_data(self) -> Dict[str, Any]:
        """Return fallback data when SEC analysis fails"""
        return {
            "financial_health_score": 50,
            "risk_level": "medium",
            "metrics": {},
            "data_source": "SEC EDGAR (fallback)",
            "last_updated": datetime.utcnow().isoformat(),
            "note": "Limited data available"
        }

# Global instance
sec_edgar = SECEDGARIntegration()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="sec_edgar",
        ttl_seconds=3600,  # 1 hour cache
        fallback_ttl_seconds=86400  # 24 hour fallback
    )
)
async def get_company_financial_health(cik: str) -> Dict[str, Any]:
    """Get financial health analysis for a company"""
    try:
        # Get comprehensive financial data
        facts = await sec_edgar.get_company_facts(cik)
        if not facts:
            return sec_edgar._get_fallback_health_data()
        
        # Calculate health metrics
        health_data = sec_edgar.calculate_financial_health_score(facts)
        
        # Add company info
        submissions = await sec_edgar.get_company_submissions(cik)
        if submissions:
            health_data["company_name"] = submissions.get("name", "Unknown Company")
            health_data["ticker"] = submissions.get("tickers", [""])[0] if submissions.get("tickers") else ""
        
        return health_data
        
    except Exception as e:
        logger.error(f"SEC EDGAR integration failed for CIK {cik}: {e}")
        return sec_edgar._get_fallback_health_data()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="sec_edgar_market_intel",
        ttl_seconds=1800,  # 30 minute cache
        fallback_ttl_seconds=7200  # 2 hour fallback
    )
)
async def get_market_intelligence() -> Dict[str, Any]:
    """Get overall market intelligence from SEC filing data"""
    try:
        # Sample of major companies for market overview
        major_companies = [
            "0000789019",  # Microsoft
            "0000320193",  # Apple
            "0001652044",  # Google/Alphabet
            "0001018724",  # Amazon
            "0001045810",  # NVIDIA
        ]
        
        company_data = []
        for cik in major_companies:
            health_data = await get_company_financial_health(cik)
            if health_data:
                company_data.append(health_data)
            
            # Rate limiting - SEC allows max 10 requests per second
            await asyncio.sleep(0.1)
        
        # Calculate market-wide metrics
        avg_health_score = sum(c.get("financial_health_score", 50) for c in company_data) / len(company_data)
        
        risk_distribution = {}
        for company in company_data:
            risk_level = company.get("risk_level", "medium")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            "market_health_score": round(avg_health_score, 1),
            "risk_distribution": risk_distribution,
            "companies_analyzed": len(company_data),
            "data_source": "SEC EDGAR",
            "last_updated": datetime.utcnow().isoformat(),
            "companies": company_data
        }
        
    except Exception as e:
        logger.error(f"SEC EDGAR market intelligence failed: {e}")
        return {
            "market_health_score": 65.0,
            "risk_distribution": {"low": 2, "medium": 2, "high": 1},
            "companies_analyzed": 5,
            "data_source": "SEC EDGAR (fallback)",
            "last_updated": datetime.utcnow().isoformat(),
            "companies": []
        }

async def cleanup_sec_edgar():
    """Cleanup SEC EDGAR session"""
    if hasattr(sec_edgar, 'session'):
        await sec_edgar.session.aclose()