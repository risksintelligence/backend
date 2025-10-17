"""
Policy Tracker Module

Tracks government policy references and decisions influenced by RiskX platform.
Monitors policy documents, regulatory guidance, and legislative activity.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import re

from ..core.config import get_settings
from ..cache.cache_manager import CacheManager


@dataclass
class PolicyReference:
    """Represents a policy reference or decision"""
    reference_id: str
    agency: str
    document_type: str  # 'regulation', 'guidance', 'report', 'testimony', 'press_release'
    title: str
    content: str
    url: str
    publication_date: datetime
    influence_level: str  # 'high', 'medium', 'low'
    reference_type: str  # 'direct_citation', 'methodology_reference', 'data_usage', 'conceptual'
    policy_area: str  # 'financial_stability', 'supply_chain', 'cybersecurity', 'risk_management'
    keywords_matched: List[str]
    context_excerpt: str


@dataclass
class PolicyMetrics:
    """Aggregated policy tracking metrics"""
    total_references: int
    high_influence_references: int
    agency_breakdown: Dict[str, int]
    policy_area_breakdown: Dict[str, int]
    reference_type_breakdown: Dict[str, int]
    recent_policy_developments: List[PolicyReference]
    trending_policy_topics: List[str]
    legislative_mentions: int


class PolicyTracker:
    """
    Tracks government policy references and decisions influenced by RiskX.
    
    Monitors:
    - Federal Reserve communications and reports
    - Treasury Department publications
    - CISA cybersecurity guidance
    - Congressional testimony and reports
    - Regulatory agency publications
    - Government research reports
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Search terms that indicate policy relevance
        self.policy_search_terms = [
            "risk intelligence", "systemic risk assessment", "supply chain resilience",
            "financial stability monitoring", "predictive risk analytics",
            "risk intelligence framework", "economic vulnerability assessment"
        ]
        
        # Government sources to monitor
        self.government_sources = {
            # Federal Reserve
            "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
            "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
            "fed_reports": "https://www.federalreserve.gov/feeds/publications.xml",
            
            # Treasury Department
            "treasury_press": "https://home.treasury.gov/rss/press-releases",
            "treasury_reports": "https://home.treasury.gov/rss/reports",
            
            # Department of Homeland Security / CISA
            "cisa_news": "https://www.cisa.gov/news.xml",
            "cisa_alerts": "https://us-cert.cisa.gov/ncas/alerts.xml",
            
            # Commerce Department
            "commerce_news": "https://www.commerce.gov/news/rss.xml",
            "bea_news": "https://www.bea.gov/rss/news_releases.xml",
            
            # Other agencies
            "gao_reports": "https://www.gao.gov/rss/reports.xml",
            "crs_reports": "https://crsreports.congress.gov/rss/reports.xml",
            "whitehouse": "https://www.whitehouse.gov/feed/"
        }
        
        # Policy areas and their keywords
        self.policy_areas = {
            "financial_stability": [
                "financial stability", "systemic risk", "bank supervision",
                "monetary policy", "financial regulation", "capital adequacy"
            ],
            "supply_chain": [
                "supply chain", "logistics", "trade security", "critical infrastructure",
                "supply chain resilience", "economic security"
            ],
            "cybersecurity": [
                "cybersecurity", "cyber threat", "information security",
                "critical infrastructure protection", "cyber resilience"
            ],
            "risk_management": [
                "risk assessment", "risk management", "vulnerability analysis",
                "threat assessment", "risk mitigation", "emergency preparedness"
            ],
            "economic_policy": [
                "economic policy", "fiscal policy", "trade policy",
                "economic analysis", "economic indicators", "economic security"
            ]
        }
    
    async def track_daily_policy_activity(self) -> PolicyMetrics:
        """
        Track daily policy activity across all monitored government sources
        """
        self.logger.info("Starting daily policy activity tracking")
        
        try:
            all_references = []
            
            # Check each government source
            for source_name, feed_url in self.government_sources.items():
                try:
                    references = await self._analyze_policy_source(source_name, feed_url)
                    all_references.extend(references)
                    self.logger.info(f"Found {len(references)} policy references from {source_name}")
                except Exception as e:
                    self.logger.warning(f"Error analyzing {source_name}: {str(e)}")
                
                # Rate limiting for government APIs
                await asyncio.sleep(1)
            
            # Filter for new policy references
            new_references = await self._filter_new_references(all_references)
            
            # Calculate metrics
            metrics = self._calculate_policy_metrics(new_references)
            
            # Cache results
            if new_references:
                await self._cache_policy_data(new_references, metrics)
            
            self.logger.info(f"Policy tracking completed. Found {len(new_references)} new references")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in daily policy activity tracking: {str(e)}")
            raise
    
    async def _analyze_policy_source(self, source_name: str, feed_url: str) -> List[PolicyReference]:
        """Analyze a specific government source for policy references"""
        references = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(feed_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:15]:  # Limit to recent entries
                            reference = await self._process_policy_entry(entry, source_name)
                            if reference:
                                references.append(reference)
            
            except Exception as e:
                self.logger.error(f"Error fetching {source_name}: {str(e)}")
        
        return references
    
    async def _process_policy_entry(self, entry: Any, source_name: str) -> Optional[PolicyReference]:
        """Process a single policy entry to extract reference data"""
        try:
            title = entry.get('title', '')
            content = entry.get('summary', '') or entry.get('description', '')
            full_text = f"{title} {content}".lower()
            
            # Check for policy-relevant terms
            matched_terms = []
            for term in self.policy_search_terms:
                if term.lower() in full_text:
                    matched_terms.append(term)
            
            # Also check for broader risk management terms
            risk_terms = [
                "risk assessment", "risk management", "risk analysis", "risk monitoring",
                "vulnerability", "resilience", "stability", "security", "threat"
            ]
            
            risk_matches = [term for term in risk_terms if term in full_text]
            
            # Require either direct match or strong risk context
            if not matched_terms and len(risk_matches) < 2:
                return None
            
            # Extract metadata
            url = entry.get('link', '')
            pub_date = self._parse_publication_date(entry.get('published'))
            
            # Determine agency from source
            agency = self._determine_agency(source_name)
            
            # Determine document type
            doc_type = self._determine_document_type(title, content, source_name)
            
            # Assess influence level
            influence_level = self._assess_influence_level(title, content, source_name, matched_terms)
            
            # Categorize reference type
            reference_type = self._categorize_reference_type(title, content, matched_terms)
            
            # Determine policy area
            policy_area = self._determine_policy_area(title, content)
            
            # Extract context excerpt
            context_excerpt = self._extract_context_excerpt(content, matched_terms + risk_matches)
            
            reference = PolicyReference(
                reference_id=f"{source_name}_{hash(url)}_{pub_date.strftime('%Y%m%d')}",
                agency=agency,
                document_type=doc_type,
                title=title,
                content=content[:1000],  # Truncate content
                url=url,
                publication_date=pub_date,
                influence_level=influence_level,
                reference_type=reference_type,
                policy_area=policy_area,
                keywords_matched=matched_terms + risk_matches[:3],  # Limit matches
                context_excerpt=context_excerpt
            )
            
            return reference
            
        except Exception as e:
            self.logger.warning(f"Error processing policy entry: {str(e)}")
            return None
    
    def _parse_publication_date(self, date_str: Optional[str]) -> datetime:
        """Parse publication date from various formats"""
        if not date_str:
            return datetime.now()
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            return datetime.now()
    
    def _determine_agency(self, source_name: str) -> str:
        """Determine government agency from source name"""
        agency_mapping = {
            "fed": "Federal Reserve",
            "treasury": "Treasury Department",
            "cisa": "CISA",
            "commerce": "Commerce Department",
            "bea": "Bureau of Economic Analysis",
            "gao": "Government Accountability Office",
            "crs": "Congressional Research Service",
            "whitehouse": "White House"
        }
        
        for key, agency in agency_mapping.items():
            if key in source_name.lower():
                return agency
        
        return "Government Agency"
    
    def _determine_document_type(self, title: str, content: str, source_name: str) -> str:
        """Determine the type of policy document"""
        text = f"{title} {content}".lower()
        
        if any(word in text for word in ['regulation', 'rule', 'regulatory']):
            return 'regulation'
        elif any(word in text for word in ['guidance', 'guidelines', 'advisory']):
            return 'guidance'  
        elif any(word in text for word in ['report', 'analysis', 'study']):
            return 'report'
        elif any(word in text for word in ['testimony', 'statement', 'remarks']):
            return 'testimony'
        elif any(word in text for word in ['press release', 'announcement']):
            return 'press_release'
        elif any(word in text for word in ['policy', 'framework', 'strategy']):
            return 'policy_document'
        else:
            return 'other'
    
    def _assess_influence_level(self, title: str, content: str, source_name: str, matched_terms: List[str]) -> str:
        """Assess the level of influence this reference represents"""
        text = f"{title} {content}".lower()
        
        # High influence indicators
        high_indicators = [
            'federal reserve chair', 'secretary', 'commissioner', 'director',
            'policy', 'regulation', 'framework', 'guidance', 'testimony'
        ]
        
        # Check for high-level source
        high_level_sources = ['fed_speeches', 'treasury_press', 'whitehouse']
        
        # Direct mentions carry high weight
        if matched_terms and any(term.lower() in text for term in self.policy_search_terms):
            if any(indicator in text for indicator in high_indicators) or source_name in high_level_sources:
                return 'high'
            else:
                return 'medium'
        
        # Indirect references with risk context
        if len(matched_terms) == 0:  # Only risk terms matched
            return 'low'
        
        return 'medium'
    
    def _categorize_reference_type(self, title: str, content: str, matched_terms: List[str]) -> str:
        """Categorize how RiskX concepts are referenced"""
        text = f"{title} {content}".lower()
        
        # Direct citation indicators
        if any(term.lower() in text for term in self.policy_search_terms):
            if any(word in text for word in ['data', 'analysis', 'research', 'study']):
                return 'data_usage'
            elif any(word in text for word in ['methodology', 'framework', 'approach']):
                return 'methodology_reference'
            else:
                return 'direct_citation'
        else:
            return 'conceptual'
    
    def _determine_policy_area(self, title: str, content: str) -> str:
        """Determine the primary policy area"""
        text = f"{title} {content}".lower()
        
        area_scores = {}
        for area, keywords in self.policy_areas.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                area_scores[area] = score
        
        if area_scores:
            return max(area_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def _extract_context_excerpt(self, content: str, keywords: List[str]) -> str:
        """Extract relevant context excerpt around keywords"""
        if not keywords or not content:
            return content[:200] + "..." if len(content) > 200 else content
        
        content_lower = content.lower()
        
        # Find the first keyword occurrence
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in content_lower:
                # Find position and extract surrounding context
                pos = content_lower.find(keyword_lower)
                start = max(0, pos - 100)
                end = min(len(content), pos + 200)
                
                excerpt = content[start:end]
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(content):
                    excerpt = excerpt + "..."
                
                return excerpt
        
        # Fallback to beginning of content
        return content[:200] + "..." if len(content) > 200 else content
    
    async def _filter_new_references(self, references: List[PolicyReference]) -> List[PolicyReference]:
        """Filter out policy references that have already been tracked"""
        if not references:
            return []
        
        try:
            # Get existing reference IDs from cache
            existing_ids = await self.cache.get("tracked_policy_reference_ids") or set()
            
            # Filter for new references
            new_references = []
            new_ids = set()
            
            for reference in references:
                if reference.reference_id not in existing_ids:
                    new_references.append(reference)
                    new_ids.add(reference.reference_id)
            
            # Update tracked IDs
            if new_ids:
                updated_ids = existing_ids.union(new_ids)
                await self.cache.set("tracked_policy_reference_ids", list(updated_ids), ttl=86400 * 30)
            
            return new_references
            
        except Exception as e:
            self.logger.error(f"Error filtering new policy references: {str(e)}")
            return references  # Return all if filtering fails
    
    def _calculate_policy_metrics(self, references: List[PolicyReference]) -> PolicyMetrics:
        """Calculate aggregated policy tracking metrics"""
        if not references:
            return PolicyMetrics(
                total_references=0,
                high_influence_references=0,
                agency_breakdown={},
                policy_area_breakdown={},
                reference_type_breakdown={},
                recent_policy_developments=[],
                trending_policy_topics=[],
                legislative_mentions=0
            )
        
        # Count high influence references
        high_influence_count = len([r for r in references if r.influence_level == 'high'])
        
        # Agency breakdown
        agency_counts = {}
        for reference in references:
            agency_counts[reference.agency] = agency_counts.get(reference.agency, 0) + 1
        
        # Policy area breakdown
        area_counts = {}
        for reference in references:
            area_counts[reference.policy_area] = area_counts.get(reference.policy_area, 0) + 1
        
        # Reference type breakdown
        type_counts = {}
        for reference in references:
            type_counts[reference.reference_type] = type_counts.get(reference.reference_type, 0) + 1
        
        # Count legislative mentions
        legislative_count = len([r for r in references if 
                               'congress' in r.agency.lower() or 'legislative' in r.document_type.lower()])
        
        # Extract trending topics from keywords
        all_keywords = []
        for reference in references:
            all_keywords.extend(reference.keywords_matched)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        trending_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        trending_topics = [topic[0] for topic in trending_topics]
        
        # Recent policy developments (top 5 by influence and recency)
        recent_developments = sorted(
            references,
            key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}[x.influence_level],
                x.publication_date
            ),
            reverse=True
        )[:5]
        
        return PolicyMetrics(
            total_references=len(references),
            high_influence_references=high_influence_count,
            agency_breakdown=agency_counts,
            policy_area_breakdown=area_counts,
            reference_type_breakdown=type_counts,
            recent_policy_developments=recent_developments,
            trending_policy_topics=trending_topics,
            legislative_mentions=legislative_count
        )
    
    async def _cache_policy_data(self, references: List[PolicyReference], metrics: PolicyMetrics):
        """Cache policy tracking data"""
        try:
            # Cache references
            reference_data = []
            for reference in references:
                data = {
                    "reference_id": reference.reference_id,
                    "agency": reference.agency,
                    "document_type": reference.document_type,
                    "title": reference.title,
                    "content": reference.content,
                    "url": reference.url,
                    "publication_date": reference.publication_date.isoformat(),
                    "influence_level": reference.influence_level,
                    "reference_type": reference.reference_type,
                    "policy_area": reference.policy_area,
                    "keywords_matched": reference.keywords_matched,
                    "context_excerpt": reference.context_excerpt
                }
                reference_data.append(data)
            
            # Cache today's references
            date_key = f"policy_references_{datetime.now().strftime('%Y%m%d')}"
            await self.cache.set(date_key, reference_data, ttl=86400 * 30)
            
            # Cache metrics
            metrics_data = {
                "total_references": metrics.total_references,
                "high_influence_references": metrics.high_influence_references,
                "agency_breakdown": metrics.agency_breakdown,
                "policy_area_breakdown": metrics.policy_area_breakdown,
                "reference_type_breakdown": metrics.reference_type_breakdown,
                "trending_policy_topics": metrics.trending_policy_topics,
                "legislative_mentions": metrics.legislative_mentions,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.cache.set("policy_latest_metrics", metrics_data, ttl=86400)
            
        except Exception as e:
            self.logger.error(f"Error caching policy data: {str(e)}")
    
    async def get_policy_influence_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get policy influence summary for specified period"""
        try:
            all_references = []
            
            # Aggregate references for the period
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                date_key = f"policy_references_{date.strftime('%Y%m%d')}"
                
                cached_references = await self.cache.get(date_key)
                if cached_references:
                    all_references.extend(cached_references)
            
            if not all_references:
                return {"error": "No policy data available for specified period"}
            
            # Analyze influence patterns
            high_influence = [r for r in all_references if r.get("influence_level") == "high"]
            direct_citations = [r for r in all_references if r.get("reference_type") == "direct_citation"]
            
            # Agency analysis
            agency_influence = {}
            for reference in all_references:
                agency = reference.get("agency", "Unknown")
                if agency not in agency_influence:
                    agency_influence[agency] = {"total": 0, "high_influence": 0}
                
                agency_influence[agency]["total"] += 1
                if reference.get("influence_level") == "high":
                    agency_influence[agency]["high_influence"] += 1
            
            # Recent high-impact developments
            high_impact_recent = [
                r for r in all_references 
                if r.get("influence_level") == "high" and 
                (datetime.now() - datetime.fromisoformat(r.get("publication_date", datetime.now().isoformat()))).days <= 7
            ]
            
            summary = {
                "period_days": days,
                "total_policy_references": len(all_references),
                "high_influence_references": len(high_influence),
                "direct_citations": len(direct_citations),
                "influence_rate": len(high_influence) / len(all_references) * 100 if all_references else 0,
                "agency_influence": dict(sorted(agency_influence.items(), key=lambda x: x[1]["high_influence"], reverse=True)[:5]),
                "policy_areas": self._analyze_policy_areas(all_references),
                "recent_high_impact": [
                    {
                        "title": r.get("title", ""),
                        "agency": r.get("agency", ""),
                        "url": r.get("url", ""),
                        "policy_area": r.get("policy_area", ""),
                        "publication_date": r.get("publication_date", "")
                    }
                    for r in high_impact_recent[:5]
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating policy influence summary: {str(e)}")
            return {"error": "Could not generate policy influence summary"}
    
    def _analyze_policy_areas(self, references: List[Dict]) -> Dict[str, Any]:
        """Analyze policy area distribution and trends"""
        area_counts = {}
        area_influence = {}
        
        for reference in references:
            area = reference.get("policy_area", "general")
            influence = reference.get("influence_level", "low")
            
            # Count total references per area
            area_counts[area] = area_counts.get(area, 0) + 1
            
            # Track influence levels per area
            if area not in area_influence:
                area_influence[area] = {"high": 0, "medium": 0, "low": 0}
            area_influence[area][influence] += 1
        
        # Calculate influence scores
        area_scores = {}
        for area in area_counts:
            high_count = area_influence[area]["high"]
            medium_count = area_influence[area]["medium"]
            total_count = area_counts[area]
            
            # Weighted influence score
            influence_score = (high_count * 3 + medium_count * 2) / total_count
            area_scores[area] = {
                "total_references": total_count,
                "influence_breakdown": area_influence[area],
                "influence_score": round(influence_score, 2)
            }
        
        return dict(sorted(area_scores.items(), key=lambda x: x[1]["influence_score"], reverse=True))