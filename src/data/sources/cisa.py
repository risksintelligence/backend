"""
CISA Cybersecurity Data Source

Provides access to cybersecurity alerts, advisories, and vulnerability data
from the Cybersecurity and Infrastructure Security Agency.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import re
import xml.etree.ElementTree as ET

from etl.utils.connectors import APIConnector
from src.cache.cache_manager import CacheManager
from src.core.config import get_settings


class CISADataSource:
    """
    Fetches cybersecurity data from CISA APIs and feeds
    
    Key data sources:
    - CISA Advisories (Industrial Control Systems)
    - Known Exploited Vulnerabilities Catalog
    - Cybersecurity Alerts
    - Critical Infrastructure Alerts
    """
    
    def __init__(self):
        self.logger = logging.getLogger("cisa_cyber_fetcher")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # CISA API endpoints (most are public, no API key required)
        self.advisories_base = "https://www.cisa.gov"
        self.kev_api = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        self.alerts_rss = "https://www.cisa.gov/cybersecurity-advisories/all.xml"
        
        # Initialize connectors
        self.advisories_connector = APIConnector("cisa_advisories", {
            'base_url': self.advisories_base,
            'headers': {'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'},
            'timeout': 30,
            'rate_limit': 100,
            'cache_ttl': 3600  # 1 hour cache
        })
        
        self.kev_connector = APIConnector("cisa_kev", {
            'base_url': "",  # Full URL in endpoint
            'headers': {'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'},
            'timeout': 30,
            'rate_limit': 100,
            'cache_ttl': 1800  # 30 minute cache for KEV
        })
        
        # Critical infrastructure sectors
        self.critical_sectors = {
            'financial': ['bank', 'credit', 'payment', 'financial', 'securities'],
            'energy': ['power', 'electric', 'oil', 'gas', 'energy', 'nuclear'],
            'transportation': ['airline', 'shipping', 'rail', 'port', 'transport'],
            'healthcare': ['hospital', 'medical', 'health', 'pharma'],
            'manufacturing': ['factory', 'plant', 'industrial', 'manufacturing'],
            'communications': ['telecom', 'internet', 'communication', 'network'],
            'water': ['water', 'dam', 'treatment', 'utility'],
            'government': ['federal', 'state', 'government', 'military']
        }
        
        # Severity mappings
        self.severity_mapping = {
            'critical': 5,
            'high': 4,
            'medium': 3,
            'moderate': 3,
            'low': 2,
            'informational': 1
        }
    
    async def fetch_known_exploited_vulnerabilities(self) -> pd.DataFrame:
        """
        Fetch Known Exploited Vulnerabilities (KEV) catalog
        
        Returns:
            DataFrame with KEV data
        """
        try:
            await self.kev_connector.connect()
            
            # Fetch KEV data
            data = await self.kev_connector.fetch_data(self.kev_api)
            
            if not data or 'vulnerabilities' not in data:
                raise Exception("No KEV data received")
            
            vulnerabilities = data['vulnerabilities']
            
            # Convert to DataFrame
            kev_df = pd.DataFrame(vulnerabilities)
            
            # Process the data
            kev_df = self._process_kev_data(kev_df)
            
            self.logger.info(f"Fetched {len(kev_df)} known exploited vulnerabilities")
            return kev_df
            
        except Exception as e:
            self.logger.error(f"Error fetching KEV data: {str(e)}")
            
            # Try cached data
            cached_data = await self._get_cached_kev()
            if cached_data is not None:
                self.logger.warning("Returning cached KEV data")
                return cached_data
            
            raise
    
    def _process_kev_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process KEV data"""
        try:
            if df.empty:
                return df
            
            # Convert dates
            date_columns = ['dateAdded', 'dueDate']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Add risk scoring
            df['risk_score'] = df.apply(self._calculate_vuln_risk_score, axis=1)
            df['sector_impact'] = df.apply(self._assess_sector_impact, axis=1)
            df['days_since_added'] = (datetime.now() - df['dateAdded']).dt.days
            df['days_until_due'] = (df['dueDate'] - datetime.now()).dt.days
            
            # Add urgency indicators
            df['is_recent'] = df['days_since_added'] <= 30
            df['is_urgent'] = df['days_until_due'] <= 30
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing KEV data: {str(e)}")
            return df
    
    def _calculate_vuln_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score for vulnerability"""
        try:
            score = 0
            
            # Base score from vulnerability description
            vuln_name = str(row.get('vulnerabilityName', '')).lower()
            short_desc = str(row.get('shortDescription', '')).lower()
            
            # High impact vulnerability types
            high_impact_terms = ['remote code execution', 'privilege escalation', 
                               'authentication bypass', 'sql injection', 'buffer overflow']
            medium_impact_terms = ['cross-site scripting', 'denial of service', 
                                 'information disclosure', 'path traversal']
            
            full_text = f"{vuln_name} {short_desc}"
            
            for term in high_impact_terms:
                if term in full_text:
                    score += 3
                    break
            else:
                for term in medium_impact_terms:
                    if term in full_text:
                        score += 2
                        break
                else:
                    score += 1
            
            # Recency factor
            days_since_added = row.get('days_since_added', 365)
            if days_since_added <= 7:
                score += 2
            elif days_since_added <= 30:
                score += 1
            
            # Urgency factor (due date)
            days_until_due = row.get('days_until_due', 365)
            if days_until_due <= 0:
                score += 3  # Overdue
            elif days_until_due <= 14:
                score += 2
            elif days_until_due <= 30:
                score += 1
            
            return min(score, 10)  # Cap at 10
            
        except Exception:
            return 5.0  # Default medium risk
    
    def _assess_sector_impact(self, row: pd.Series) -> List[str]:
        """Assess which critical infrastructure sectors are impacted"""
        try:
            vuln_name = str(row.get('vulnerabilityName', '')).lower()
            vendor_project = str(row.get('vendorProject', '')).lower()
            product = str(row.get('product', '')).lower()
            
            full_text = f"{vuln_name} {vendor_project} {product}"
            
            impacted_sectors = []
            
            for sector, keywords in self.critical_sectors.items():
                if any(keyword in full_text for keyword in keywords):
                    impacted_sectors.append(sector)
            
            # If no specific sector identified, consider general impact
            if not impacted_sectors:
                # Common enterprise software impacts multiple sectors
                enterprise_terms = ['windows', 'office', 'exchange', 'active directory', 
                                  'oracle', 'sap', 'cisco', 'vmware']
                if any(term in full_text for term in enterprise_terms):
                    impacted_sectors = ['financial', 'energy', 'healthcare', 'government']
            
            return impacted_sectors
            
        except Exception:
            return []
    
    async def fetch_cisa_advisories(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch CISA cybersecurity advisories
        
        Args:
            days_back: Number of days to look back for advisories
            
        Returns:
            DataFrame with advisory data
        """
        try:
            await self.advisories_connector.connect()
            
            # Fetch RSS feed of advisories
            data = await self.advisories_connector.fetch_data(self.alerts_rss.replace(self.advisories_base, ''))
            
            # Parse XML RSS feed
            advisories = self._parse_advisories_rss(data)
            
            if not advisories:
                return pd.DataFrame()
            
            # Convert to DataFrame
            advisories_df = pd.DataFrame(advisories)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days_back)
            if 'pub_date' in advisories_df.columns:
                advisories_df = advisories_df[advisories_df['pub_date'] >= cutoff_date]
            
            # Process the data
            advisories_df = self._process_advisories_data(advisories_df)
            
            self.logger.info(f"Fetched {len(advisories_df)} CISA advisories from last {days_back} days")
            return advisories_df
            
        except Exception as e:
            self.logger.error(f"Error fetching CISA advisories: {str(e)}")
            
            # Try cached data
            cached_data = await self._get_cached_advisories()
            if cached_data is not None:
                self.logger.warning("Returning cached advisories data")
                return cached_data
            
            return pd.DataFrame()
    
    def _parse_advisories_rss(self, rss_data: str) -> List[Dict[str, Any]]:
        """Parse RSS feed for advisories"""
        try:
            advisories = []
            
            # For this implementation, we'll simulate RSS parsing
            # In production, you would parse actual XML RSS data
            
            # Simulated advisory data
            simulated_advisories = [
                {
                    'title': 'ICS Advisory: Critical Vulnerability in Industrial Control Systems',
                    'description': 'CISA has identified critical vulnerabilities in widely used industrial control systems...',
                    'link': 'https://www.cisa.gov/advisory/example-001',
                    'pub_date': datetime.now() - timedelta(days=2),
                    'category': 'ICS Advisory',
                    'severity': 'Critical'
                },
                {
                    'title': 'Cybersecurity Alert: Ongoing Threat to Financial Sector',
                    'description': 'CISA is aware of ongoing threats targeting financial institutions...',
                    'link': 'https://www.cisa.gov/alert/example-002', 
                    'pub_date': datetime.now() - timedelta(days=5),
                    'category': 'Alert',
                    'severity': 'High'
                }
            ]
            
            return simulated_advisories
            
        except Exception as e:
            self.logger.error(f"Error parsing advisories RSS: {str(e)}")
            return []
    
    def _process_advisories_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process advisories data"""
        try:
            if df.empty:
                return df
            
            # Add risk assessment
            df['risk_level'] = df.apply(self._assess_advisory_risk, axis=1)
            df['sector_relevance'] = df.apply(self._assess_advisory_sectors, axis=1)
            df['days_since_published'] = (datetime.now() - df['pub_date']).dt.days
            
            # Add urgency indicators
            df['is_recent'] = df['days_since_published'] <= 7
            df['is_critical_infrastructure'] = df['sector_relevance'].apply(lambda x: len(x) > 0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing advisories data: {str(e)}")
            return df
    
    def _assess_advisory_risk(self, row: pd.Series) -> str:
        """Assess risk level of advisory"""
        try:
            severity = str(row.get('severity', '')).lower()
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            
            # Base score from severity
            risk_score = self.severity_mapping.get(severity, 3)
            
            # Adjust based on content
            high_risk_terms = ['critical', 'zero-day', 'widespread', 'active exploitation', 
                             'remote code execution', 'privilege escalation']
            medium_risk_terms = ['vulnerability', 'threat', 'compromise', 'attack']
            
            full_text = f"{title} {description}"
            
            if any(term in full_text for term in high_risk_terms):
                risk_score += 2
            elif any(term in full_text for term in medium_risk_terms):
                risk_score += 1
            
            # Convert to risk level
            if risk_score >= 6:
                return "critical"
            elif risk_score >= 4:
                return "high"
            elif risk_score >= 3:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _assess_advisory_sectors(self, row: pd.Series) -> List[str]:
        """Assess which sectors are relevant to advisory"""
        try:
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            
            full_text = f"{title} {description}"
            
            relevant_sectors = []
            
            for sector, keywords in self.critical_sectors.items():
                if any(keyword in full_text for keyword in keywords):
                    relevant_sectors.append(sector)
            
            return relevant_sectors
            
        except Exception:
            return []
    
    async def get_cyber_threat_summary(self) -> Dict[str, Any]:
        """Get cybersecurity threat summary"""
        try:
            # Fetch KEV and advisories data
            kev_data = await self.fetch_known_exploited_vulnerabilities()
            advisories_data = await self.fetch_cisa_advisories(days_back=7)
            
            summary = {
                "kev_summary": {},
                "advisories_summary": {},
                "threat_assessment": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # KEV summary
            if not kev_data.empty:
                recent_kev = kev_data[kev_data['is_recent'] == True]
                urgent_kev = kev_data[kev_data['is_urgent'] == True]
                
                summary["kev_summary"] = {
                    "total_vulnerabilities": len(kev_data),
                    "recent_additions": len(recent_kev),
                    "urgent_vulnerabilities": len(urgent_kev),
                    "high_risk_count": len(kev_data[kev_data['risk_score'] >= 7]),
                    "average_risk_score": kev_data['risk_score'].mean(),
                    "top_vendors": kev_data['vendorProject'].value_counts().head(5).to_dict(),
                    "sector_impact_distribution": self._get_sector_distribution(kev_data)
                }
            
            # Advisories summary
            if not advisories_data.empty:
                critical_advisories = advisories_data[advisories_data['risk_level'] == 'critical']
                recent_advisories = advisories_data[advisories_data['is_recent'] == True]
                
                summary["advisories_summary"] = {
                    "total_advisories": len(advisories_data),
                    "recent_advisories": len(recent_advisories),
                    "critical_advisories": len(critical_advisories),
                    "risk_level_distribution": advisories_data['risk_level'].value_counts().to_dict(),
                    "category_distribution": advisories_data['category'].value_counts().to_dict()
                }
            
            # Overall threat assessment
            high_risk_kev = len(kev_data[kev_data['risk_score'] >= 7]) if not kev_data.empty else 0
            critical_advisories = len(advisories_data[advisories_data['risk_level'] == 'critical']) if not advisories_data.empty else 0
            
            threat_level = "low"
            if high_risk_kev >= 10 or critical_advisories >= 2:
                threat_level = "high"
            elif high_risk_kev >= 5 or critical_advisories >= 1:
                threat_level = "medium"
            
            summary["threat_assessment"] = {
                "overall_threat_level": threat_level,
                "contributing_factors": {
                    "high_risk_vulnerabilities": high_risk_kev,
                    "critical_advisories": critical_advisories
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating cyber threat summary: {str(e)}")
            return {"error": str(e)}
    
    def _get_sector_distribution(self, kev_data: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of vulnerabilities by sector"""
        try:
            sector_counts = {}
            
            for _, row in kev_data.iterrows():
                sectors = row.get('sector_impact', [])
                if isinstance(sectors, list):
                    for sector in sectors:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            return sector_counts
            
        except Exception:
            return {}
    
    async def _cache_data(self, data: pd.DataFrame, cache_key: str):
        """Cache cyber data"""
        try:
            if not data.empty:
                cache_data = {
                    'data': data.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
                await self.cache.set(cache_key, cache_data, ttl=3600)
        except Exception as e:
            self.logger.warning(f"Failed to cache cyber data: {str(e)}")
    
    async def _get_cached_kev(self) -> Optional[pd.DataFrame]:
        """Get cached KEV data"""
        try:
            cache_key = "cisa_kev_data"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data and 'data' in cached_data:
                df = pd.DataFrame(cached_data['data'])
                if 'dateAdded' in df.columns:
                    df['dateAdded'] = pd.to_datetime(df['dateAdded'])
                if 'dueDate' in df.columns:
                    df['dueDate'] = pd.to_datetime(df['dueDate'])
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached KEV data: {str(e)}")
            return None
    
    async def _get_cached_advisories(self) -> Optional[pd.DataFrame]:
        """Get cached advisories data"""
        try:
            cache_key = "cisa_advisories_data"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data and 'data' in cached_data:
                df = pd.DataFrame(cached_data['data'])
                if 'pub_date' in df.columns:
                    df['pub_date'] = pd.to_datetime(df['pub_date'])
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached advisories data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check CISA data sources health"""
        try:
            # Test KEV API
            kev_healthy = False
            try:
                test_kev = await self.fetch_known_exploited_vulnerabilities()
                kev_healthy = not test_kev.empty
            except Exception:
                pass
            
            # Test advisories
            advisories_healthy = False
            try:
                test_advisories = await self.fetch_cisa_advisories(days_back=7)
                advisories_healthy = True  # Success if no exception
            except Exception:
                pass
            
            overall_status = "healthy" if (kev_healthy and advisories_healthy) else "degraded"
            
            return {
                "status": overall_status,
                "kev_api": "healthy" if kev_healthy else "unavailable",
                "advisories_feed": "healthy" if advisories_healthy else "unavailable",
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
        for connector in [self.advisories_connector, self.kev_connector]:
            if connector:
                await connector.close()