"""
CISA cyber security alerts and vulnerabilities fetcher for supply chain risk analysis.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

import aiohttp
import pandas as pd

from src.core.config import settings
from src.cache.cache_manager import CacheManager
from src.data.sources.cisa import CisaConnector

logger = logging.getLogger(__name__)


class CisaCyberFetcher:
    """
    Fetches cyber security alerts and vulnerability data from CISA for supply chain risk analysis.
    """
    
    def __init__(self):
        """Initialize CISA cyber fetcher."""
        self.cache_manager = CacheManager()
        self.cisa_connector = CisaConnector(self.cache_manager)
        
        # Critical vulnerability types for supply chain impact
        self.critical_vuln_types = {
            "remote_code_execution",
            "privilege_escalation", 
            "authentication_bypass",
            "sql_injection",
            "buffer_overflow",
            "zero_day"
        }
        
        # CVSS score to risk level mapping
        self.cvss_risk_levels = {
            (9.0, 10.0): "critical",
            (7.0, 8.9): "high", 
            (4.0, 6.9): "medium",
            (0.1, 3.9): "low"
        }
        
        # Industrial sectors for supply chain analysis
        self.supply_chain_sectors = {
            "energy",
            "transportation", 
            "manufacturing",
            "agriculture",
            "water",
            "healthcare",
            "financial_services",
            "communications",
            "critical_manufacturing"
        }
    
    async def fetch_latest_advisories(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch latest cyber security advisories and alerts from CISA.
        
        Returns:
            List of cyber security records or None if failed
        """
        logger.info("Starting CISA cyber security data fetch")
        
        try:
            all_data = []
            
            # Fetch Known Exploited Vulnerabilities (KEV)
            kev_data = await self._fetch_kev_catalog()
            if kev_data:
                all_data.extend(kev_data)
            
            # Fetch Industrial Control Systems (ICS) advisories
            ics_advisories = await self._fetch_ics_advisories()
            if ics_advisories:
                all_data.extend(ics_advisories)
            
            # Fetch current cybersecurity alerts
            alerts = await self._fetch_cybersecurity_alerts()
            if alerts:
                all_data.extend(alerts)
            
            # Fetch vulnerability bulletins
            bulletins = await self._fetch_vulnerability_bulletins()
            if bulletins:
                all_data.extend(bulletins)
            
            if all_data:
                # Cache the aggregated data
                cache_key = f"cisa:latest_fetch:{datetime.now().strftime('%Y%m%d_%H')}"
                await self._cache_data(cache_key, all_data)
                
                logger.info(f"CISA cyber data fetch completed: {len(all_data)} records")
                return all_data
            else:
                logger.warning("No CISA cyber data retrieved")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching CISA cyber data: {str(e)}")
            
            # Try to get cached data as fallback
            fallback_data = await self._get_fallback_data()
            if fallback_data:
                logger.info("Using fallback CISA data")
                return fallback_data
            
            return None
    
    async def _fetch_kev_catalog(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch Known Exploited Vulnerabilities (KEV) catalog.
        
        Returns:
            List of KEV records or None if failed
        """
        try:
            kev_data = await self.cisa_connector.get_kev_catalog()
            
            if not kev_data or "vulnerabilities" not in kev_data:
                return None
            
            transformed_kev = []
            for vuln in kev_data["vulnerabilities"]:
                # Calculate risk score based on exploitation and criticality
                risk_score = self._calculate_vulnerability_risk_score(vuln)
                
                # Determine supply chain impact
                supply_chain_impact = self._assess_supply_chain_impact(vuln)
                
                transformed_vuln = {
                    "source": "cisa",
                    "data_type": "known_exploited_vulnerability",
                    "cve_id": vuln.get("cveID"),
                    "vendor_project": vuln.get("vendorProject"),
                    "product": vuln.get("product"),
                    "vulnerability_name": vuln.get("vulnerabilityName"),
                    "date_added": vuln.get("dateAdded"),
                    "short_description": vuln.get("shortDescription"),
                    "required_action": vuln.get("requiredAction"),
                    "due_date": vuln.get("dueDate"),
                    "known_ransomware": vuln.get("knownRansomwareCampaignUse", "Unknown"),
                    "notes": vuln.get("notes", ""),
                    "risk_score": risk_score,
                    "supply_chain_impact": supply_chain_impact,
                    "exploitation_status": "known_exploited",
                    "last_updated": datetime.now().isoformat(),
                    "is_critical": risk_score >= 4
                }
                transformed_kev.append(transformed_vuln)
            
            # Filter for supply chain relevant vulnerabilities
            relevant_kev = [
                vuln for vuln in transformed_kev
                if vuln["supply_chain_impact"]["score"] >= 3 or vuln["is_critical"]
            ]
            
            return relevant_kev
            
        except Exception as e:
            logger.error(f"Error fetching CISA KEV catalog: {str(e)}")
            return None
    
    async def _fetch_ics_advisories(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch Industrial Control Systems (ICS) advisories.
        
        Returns:
            List of ICS advisory records or None if failed
        """
        try:
            ics_data = await self.cisa_connector.get_ics_advisories(limit=50)
            
            if not ics_data:
                return None
            
            transformed_ics = []
            for advisory in ics_data:
                # Calculate industrial impact score
                impact_score = self._calculate_industrial_impact(advisory)
                
                transformed_advisory = {
                    "source": "cisa",
                    "data_type": "ics_advisory",
                    "advisory_id": advisory.get("id"),
                    "title": advisory.get("title"),
                    "summary": advisory.get("summary"),
                    "release_date": advisory.get("releaseDate"),
                    "last_revised": advisory.get("lastRevised"),
                    "affected_products": advisory.get("affectedProducts", []),
                    "cvss_score": self._parse_float(advisory.get("cvssScore")),
                    "severity": advisory.get("severity", "unknown").lower(),
                    "sector": advisory.get("sector", []),
                    "threat_vector": advisory.get("threatVector"),
                    "mitigation": advisory.get("mitigation"),
                    "impact_score": impact_score,
                    "supply_chain_relevance": self._assess_ics_supply_chain_relevance(advisory),
                    "last_updated": datetime.now().isoformat()
                }
                transformed_ics.append(transformed_advisory)
            
            # Filter for high-impact industrial advisories
            relevant_ics = [
                advisory for advisory in transformed_ics
                if advisory["impact_score"] >= 3 or advisory["supply_chain_relevance"]
            ]
            
            return relevant_ics
            
        except Exception as e:
            logger.error(f"Error fetching CISA ICS advisories: {str(e)}")
            return None
    
    async def _fetch_cybersecurity_alerts(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch current cybersecurity alerts.
        
        Returns:
            List of alert records or None if failed
        """
        try:
            alerts_data = await self.cisa_connector.get_cybersecurity_alerts(limit=25)
            
            if not alerts_data:
                return None
            
            transformed_alerts = []
            for alert in alerts_data:
                # Calculate alert urgency score
                urgency_score = self._calculate_alert_urgency(alert)
                
                transformed_alert = {
                    "source": "cisa",
                    "data_type": "cybersecurity_alert",
                    "alert_id": alert.get("id"),
                    "title": alert.get("title"),
                    "summary": alert.get("summary"),
                    "release_date": alert.get("releaseDate"),
                    "alert_type": alert.get("alertType", "unknown"),
                    "affected_systems": alert.get("affectedSystems", []),
                    "threat_actors": alert.get("threatActors", []),
                    "indicators": alert.get("indicators", []),
                    "mitigation_steps": alert.get("mitigationSteps", []),
                    "urgency_score": urgency_score,
                    "economic_threat": self._assess_economic_threat(alert),
                    "last_updated": datetime.now().isoformat()
                }
                transformed_alerts.append(transformed_alert)
            
            # Filter for economically relevant alerts
            relevant_alerts = [
                alert for alert in transformed_alerts
                if alert["urgency_score"] >= 3 or alert["economic_threat"]
            ]
            
            return relevant_alerts
            
        except Exception as e:
            logger.error(f"Error fetching CISA cybersecurity alerts: {str(e)}")
            return None
    
    async def _fetch_vulnerability_bulletins(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch vulnerability bulletins and analysis.
        
        Returns:
            List of bulletin records or None if failed
        """
        try:
            bulletins_data = await self.cisa_connector.get_vulnerability_bulletins(limit=20)
            
            if not bulletins_data:
                return None
            
            transformed_bulletins = []
            for bulletin in bulletins_data:
                # Calculate bulletin priority score
                priority_score = self._calculate_bulletin_priority(bulletin)
                
                transformed_bulletin = {
                    "source": "cisa",
                    "data_type": "vulnerability_bulletin",
                    "bulletin_id": bulletin.get("id"),
                    "title": bulletin.get("title"),
                    "description": bulletin.get("description"),
                    "publish_date": bulletin.get("publishDate"),
                    "vulnerabilities": bulletin.get("vulnerabilities", []),
                    "affected_vendors": bulletin.get("affectedVendors", []),
                    "severity_breakdown": bulletin.get("severityBreakdown", {}),
                    "exploitation_likelihood": bulletin.get("exploitationLikelihood", "unknown"),
                    "priority_score": priority_score,
                    "business_impact": self._assess_business_impact(bulletin),
                    "last_updated": datetime.now().isoformat()
                }
                transformed_bulletins.append(transformed_bulletin)
            
            # Filter for high-priority bulletins
            relevant_bulletins = [
                bulletin for bulletin in transformed_bulletins
                if bulletin["priority_score"] >= 3
            ]
            
            return relevant_bulletins
            
        except Exception as e:
            logger.error(f"Error fetching CISA vulnerability bulletins: {str(e)}")
            return None
    
    def _calculate_vulnerability_risk_score(self, vuln: Dict[str, Any]) -> int:
        """
        Calculate risk score for vulnerability (1-5 scale).
        
        Args:
            vuln: Vulnerability data
            
        Returns:
            Risk score
        """
        try:
            score = 3  # Base score for known exploited vulnerability
            
            # Increase score for critical products
            product = vuln.get("product", "").lower()
            vendor = vuln.get("vendorProject", "").lower()
            
            critical_products = ["windows", "linux", "apache", "nginx", "oracle", "cisco", "vmware"]
            if any(prod in product or prod in vendor for prod in critical_products):
                score += 1
            
            # Increase score for ransomware campaigns
            if vuln.get("knownRansomwareCampaignUse", "").lower() == "known":
                score += 1
            
            # Check if recent (added in last 30 days)
            date_added = vuln.get("dateAdded")
            if date_added:
                try:
                    added_date = datetime.strptime(date_added, "%Y-%m-%d")
                    if (datetime.now() - added_date).days <= 30:
                        score += 1
                except ValueError:
                    pass
            
            return min(score, 5)
            
        except Exception:
            return 3
    
    def _assess_supply_chain_impact(self, vuln: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess supply chain impact of vulnerability.
        
        Args:
            vuln: Vulnerability data
            
        Returns:
            Supply chain impact assessment
        """
        try:
            product = vuln.get("product", "").lower()
            vendor = vuln.get("vendorProject", "").lower()
            description = vuln.get("shortDescription", "").lower()
            
            impact_score = 1
            affected_sectors = []
            
            # Check for supply chain critical systems
            supply_chain_keywords = [
                "scada", "industrial", "control", "manufacturing", "energy", 
                "transportation", "logistics", "supply", "chain", "iot", "operational"
            ]
            
            if any(keyword in description or keyword in product for keyword in supply_chain_keywords):
                impact_score += 2
                affected_sectors.append("industrial_control")
            
            # Check for widely used software
            widespread_software = [
                "windows", "linux", "apache", "nginx", "java", "python", 
                "mysql", "postgresql", "docker", "kubernetes"
            ]
            
            if any(software in product or software in vendor for software in widespread_software):
                impact_score += 1
                affected_sectors.append("enterprise_software")
            
            # Check for network infrastructure
            network_keywords = ["router", "switch", "firewall", "vpn", "network", "dns"]
            if any(keyword in product or keyword in description for keyword in network_keywords):
                impact_score += 1
                affected_sectors.append("network_infrastructure")
            
            return {
                "score": min(impact_score, 5),
                "affected_sectors": affected_sectors,
                "assessment": "high" if impact_score >= 4 else "medium" if impact_score >= 2 else "low"
            }
            
        except Exception:
            return {"score": 1, "affected_sectors": [], "assessment": "low"}
    
    def _calculate_industrial_impact(self, advisory: Dict[str, Any]) -> int:
        """Calculate industrial impact score for ICS advisory."""
        try:
            score = 1
            
            # CVSS score impact
            cvss_score = self._parse_float(advisory.get("cvssScore"))
            if cvss_score:
                if cvss_score >= 9.0:
                    score += 3
                elif cvss_score >= 7.0:
                    score += 2
                elif cvss_score >= 4.0:
                    score += 1
            
            # Sector impact
            sectors = advisory.get("sector", [])
            critical_sectors = ["energy", "water", "transportation", "manufacturing"]
            if any(sector.lower() in critical_sectors for sector in sectors):
                score += 1
            
            # Affected products
            products = advisory.get("affectedProducts", [])
            if len(products) > 5:  # Widely used products
                score += 1
            
            return min(score, 5)
            
        except Exception:
            return 1
    
    def _assess_ics_supply_chain_relevance(self, advisory: Dict[str, Any]) -> bool:
        """Assess if ICS advisory is relevant to supply chains."""
        try:
            title = advisory.get("title", "").lower()
            summary = advisory.get("summary", "").lower()
            
            supply_chain_terms = [
                "supply", "chain", "manufacturing", "industrial", "scada",
                "plc", "hmi", "automation", "control", "operational"
            ]
            
            return any(term in title or term in summary for term in supply_chain_terms)
            
        except Exception:
            return False
    
    def _calculate_alert_urgency(self, alert: Dict[str, Any]) -> int:
        """Calculate urgency score for cybersecurity alert."""
        try:
            score = 1
            
            # Alert type priority
            alert_type = alert.get("alertType", "").lower()
            if "emergency" in alert_type or "critical" in alert_type:
                score += 2
            elif "urgent" in alert_type or "high" in alert_type:
                score += 1
            
            # Threat actors
            threat_actors = alert.get("threatActors", [])
            if threat_actors:
                score += 1
            
            # Indicators count
            indicators = alert.get("indicators", [])
            if len(indicators) > 10:
                score += 1
            
            return min(score, 5)
            
        except Exception:
            return 1
    
    def _assess_economic_threat(self, alert: Dict[str, Any]) -> bool:
        """Assess if alert poses economic threat."""
        try:
            title = alert.get("title", "").lower()
            summary = alert.get("summary", "").lower()
            
            economic_terms = [
                "financial", "banking", "payment", "economic", "supply", "chain",
                "critical", "infrastructure", "ransomware", "disruption"
            ]
            
            return any(term in title or term in summary for term in economic_terms)
            
        except Exception:
            return False
    
    def _calculate_bulletin_priority(self, bulletin: Dict[str, Any]) -> int:
        """Calculate priority score for vulnerability bulletin."""
        try:
            score = 1
            
            # Severity breakdown
            severity_breakdown = bulletin.get("severityBreakdown", {})
            critical_count = severity_breakdown.get("critical", 0)
            high_count = severity_breakdown.get("high", 0)
            
            if critical_count > 0:
                score += 2
            if high_count > 5:
                score += 1
            
            # Exploitation likelihood
            exploitation = bulletin.get("exploitationLikelihood", "").lower()
            if "high" in exploitation:
                score += 2
            elif "medium" in exploitation:
                score += 1
            
            return min(score, 5)
            
        except Exception:
            return 1
    
    def _assess_business_impact(self, bulletin: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of vulnerability bulletin."""
        try:
            affected_vendors = bulletin.get("affectedVendors", [])
            vulnerabilities = bulletin.get("vulnerabilities", [])
            
            # Calculate impact based on vendor reach and vulnerability count
            vendor_reach_score = min(len(affected_vendors), 5)
            vuln_volume_score = min(len(vulnerabilities) // 10, 3)
            
            total_impact = vendor_reach_score + vuln_volume_score
            
            return {
                "score": min(total_impact, 5),
                "vendor_count": len(affected_vendors),
                "vulnerability_count": len(vulnerabilities),
                "assessment": "high" if total_impact >= 4 else "medium" if total_impact >= 2 else "low"
            }
            
        except Exception:
            return {"score": 1, "vendor_count": 0, "vulnerability_count": 0, "assessment": "low"}
    
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
            # Cache for 4 hours (cyber threat data changes frequently)
            return self.cache_manager.set(
                cache_key, 
                data, 
                ttl=4 * 3600,
                persist_to_postgres=True
            )
        except Exception as e:
            logger.error(f"Error caching CISA data: {str(e)}")
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
                cache_key = f"cisa:latest_fetch:{cache_time}"
                
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using CISA fallback data from {cache_time}")
                    return cached_data
            
            # Try fallback handler
            fallback_data = self.cache_manager.fallback_handler.get_fallback_data("cisa")
            if fallback_data:
                return fallback_data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting CISA fallback data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on CISA data source.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check if we can reach CISA API
            health_result = await self.cisa_connector.health_check()
            
            # Check cache availability
            cache_available = self.cache_manager.exists("cisa:latest_fetch")
            
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
            logger.error(f"CISA health check failed: {str(e)}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test function for CISA cyber fetcher."""
    fetcher = CisaCyberFetcher()
    
    print("Testing CISA cyber fetcher...")
    
    # Test health check
    health = await fetcher.health_check()
    print(f"Health check: {health}")
    
    # Test data fetch
    data = await fetcher.fetch_latest_advisories()
    if data:
        print(f"Fetched {len(data)} CISA records")
        print(f"Sample record: {data[0] if data else 'None'}")
    else:
        print("No data retrieved")


if __name__ == "__main__":
    asyncio.run(main())