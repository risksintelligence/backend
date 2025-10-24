"""
Cybersecurity & Infrastructure Security Agency (CISA) API Integration
"""
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CVE_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class CISAClient:
    """Async client for CISA vulnerability data with rate limiting and error handling."""
    
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
        """Make request to CISA/CVE APIs with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        await self._rate_limit()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"CISA API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"CISA API timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"CISA API error for {url}: {e}")
            return None
    
    async def get_kev_catalog(self) -> Optional[Dict]:
        """Get CISA Known Exploited Vulnerabilities catalog."""
        
        data = await self._make_request(CISA_KEV_URL)
        
        if data and "vulnerabilities" in data:
            vulnerabilities = data["vulnerabilities"]
            
            # Get recent high-priority vulnerabilities (last 30 days)
            recent_date = datetime.now() - timedelta(days=30)
            recent_vulns = []
            
            for vuln in vulnerabilities:
                try:
                    date_added = datetime.strptime(vuln.get("dateAdded", ""), "%Y-%m-%d")
                    if date_added >= recent_date:
                        recent_vulns.append({
                            "cve_id": vuln.get("cveID"),
                            "vendor_project": vuln.get("vendorProject"),
                            "product": vuln.get("product"),
                            "vulnerability_name": vuln.get("vulnerabilityName"),
                            "date_added": vuln.get("dateAdded"),
                            "short_description": vuln.get("shortDescription"),
                            "required_action": vuln.get("requiredAction"),
                            "due_date": vuln.get("dueDate"),
                            "known_ransomware": vuln.get("knownRansomwareCampaignUse", "Unknown")
                        })
                except (ValueError, TypeError):
                    continue
            
            # Calculate risk metrics
            total_vulns = len(vulnerabilities)
            recent_vulns_count = len(recent_vulns)
            ransomware_vulns = sum(1 for v in vulnerabilities 
                                 if v.get("knownRansomwareCampaignUse") == "Known")
            
            return {
                "catalog_version": data.get("catalogVersion"),
                "date_released": data.get("dateReleased"),
                "count": data.get("count", total_vulns),
                "total_vulnerabilities": total_vulns,
                "recent_vulnerabilities": recent_vulns_count,
                "ransomware_associated": ransomware_vulns,
                "recent_high_priority": recent_vulns[:10],  # Top 10 recent
                "risk_score": min(100, (recent_vulns_count * 2) + (ransomware_vulns * 0.1)),
                "source": "cisa_kev",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def get_infrastructure_sectors(self) -> Optional[Dict]:
        """Get critical infrastructure sector risk assessment."""
        
        # Based on CISA's 16 critical infrastructure sectors
        sectors = [
            "Energy", "Water and Wastewater", "Transportation", "Communications",
            "Information Technology", "Healthcare and Public Health", "Food and Agriculture",
            "Banking and Finance", "Chemical", "Critical Manufacturing",
            "Defense Industrial Base", "Emergency Services", "Nuclear Reactors",
            "Dams", "Government Facilities", "Commercial Facilities"
        ]
        
        # Get KEV data to assess sector-specific risks
        kev_data = await self.get_kev_catalog()
        
        if kev_data:
            # Analyze vulnerabilities by sector keywords
            sector_risks = {}
            
            for sector in sectors:
                sector_vulns = 0
                recent_vulns = kev_data.get("recent_high_priority", [])
                
                for vuln in recent_vulns:
                    description = vuln.get("short_description", "").lower()
                    product = vuln.get("product", "").lower()
                    vendor = vuln.get("vendor_project", "").lower()
                    
                    # Basic keyword matching for sector relevance
                    sector_keywords = {
                        "Energy": ["power", "electric", "grid", "scada", "energy"],
                        "Information Technology": ["software", "windows", "linux", "network", "server"],
                        "Communications": ["router", "switch", "telecom", "phone", "communication"],
                        "Transportation": ["transport", "vehicle", "traffic", "aviation", "maritime"],
                        "Healthcare and Public Health": ["medical", "health", "hospital", "patient"],
                        "Banking and Finance": ["bank", "financial", "payment", "atm", "trading"],
                        "Water and Wastewater": ["water", "wastewater", "treatment", "pump"],
                        "Food and Agriculture": ["food", "agriculture", "farm", "crop"],
                        "Chemical": ["chemical", "plant", "industrial", "manufacturing"],
                        "Defense Industrial Base": ["defense", "military", "weapon", "aerospace"]
                    }
                    
                    keywords = sector_keywords.get(sector, [sector.lower().split()[0]])
                    
                    if any(keyword in description or keyword in product or keyword in vendor 
                           for keyword in keywords):
                        sector_vulns += 1
                
                risk_level = "Critical" if sector_vulns >= 3 else "High" if sector_vulns >= 2 else "Medium" if sector_vulns >= 1 else "Low"
                
                sector_risks[sector] = {
                    "vulnerability_count": sector_vulns,
                    "risk_level": risk_level,
                    "risk_score": min(100, sector_vulns * 25)
                }
            
            return {
                "sectors": sector_risks,
                "overall_infrastructure_risk": kev_data.get("risk_score", 0),
                "assessment_date": datetime.utcnow().isoformat(),
                "source": "cisa_infrastructure"
            }
        
        return None
    
    async def get_threat_intelligence(self) -> Optional[Dict]:
        """Get current cybersecurity threat intelligence summary."""
        
        kev_data = await self.get_kev_catalog()
        
        if kev_data:
            recent_vulns = kev_data.get("recent_high_priority", [])
            
            # Analyze threat patterns
            threat_types = {}
            exploit_trends = {}
            vendor_risks = {}
            
            for vuln in recent_vulns:
                # Categorize by threat type
                description = vuln.get("short_description", "").lower()
                if "remote code execution" in description or "rce" in description:
                    threat_types["Remote Code Execution"] = threat_types.get("Remote Code Execution", 0) + 1
                elif "privilege escalation" in description:
                    threat_types["Privilege Escalation"] = threat_types.get("Privilege Escalation", 0) + 1
                elif "sql injection" in description:
                    threat_types["SQL Injection"] = threat_types.get("SQL Injection", 0) + 1
                elif "cross-site scripting" in description or "xss" in description:
                    threat_types["Cross-Site Scripting"] = threat_types.get("Cross-Site Scripting", 0) + 1
                else:
                    threat_types["Other"] = threat_types.get("Other", 0) + 1
                
                # Track vendor risks
                vendor = vuln.get("vendor_project", "Unknown")
                if vendor != "Unknown":
                    vendor_risks[vendor] = vendor_risks.get(vendor, 0) + 1
                
                # Track exploitation timeline
                date_added = vuln.get("date_added")
                if date_added:
                    exploit_trends[date_added] = exploit_trends.get(date_added, 0) + 1
            
            # Calculate threat intelligence score
            active_threats = len(recent_vulns)
            ransomware_threats = kev_data.get("ransomware_associated", 0)
            threat_diversity = len(threat_types)
            
            threat_score = min(100, (active_threats * 3) + (ransomware_threats * 0.5) + (threat_diversity * 2))
            
            return {
                "active_exploited_vulnerabilities": active_threats,
                "ransomware_associated_threats": ransomware_threats,
                "threat_types": threat_types,
                "top_vulnerable_vendors": dict(sorted(vendor_risks.items(), 
                                                    key=lambda x: x[1], reverse=True)[:10]),
                "threat_intelligence_score": threat_score,
                "risk_level": "Critical" if threat_score >= 75 else "High" if threat_score >= 50 else "Medium",
                "assessment_date": datetime.utcnow().isoformat(),
                "source": "cisa_threat_intel"
            }
        
        return None


async def get_cybersecurity_threats() -> Dict[str, Any]:
    """Get comprehensive cybersecurity threat assessment."""
    
    async with CISAClient() as client:
        # Fetch multiple threat indicators concurrently
        results = await asyncio.gather(
            client.get_kev_catalog(),
            client.get_infrastructure_sectors(),
            client.get_threat_intelligence(),
            return_exceptions=True
        )
        
        indicators = {}
        indicator_names = ["kev_catalog", "infrastructure_risks", "threat_intelligence"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[indicator_names[i]] = result
        
        # Calculate overall cybersecurity risk score
        overall_risk = 0
        if indicators:
            kev_risk = indicators.get("kev_catalog", {}).get("risk_score", 0)
            threat_risk = indicators.get("threat_intelligence", {}).get("threat_intelligence_score", 0)
            overall_risk = (kev_risk + threat_risk) / 2
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "overall_cybersecurity_risk": overall_risk,
            "risk_level": "Critical" if overall_risk >= 75 else "High" if overall_risk >= 50 else "Medium",
            "source": "cisa",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if CISA APIs are accessible."""
    try:
        async with CISAClient() as client:
            # Try to get KEV catalog
            result = await client.get_kev_catalog()
            return result is not None
    except Exception:
        return False