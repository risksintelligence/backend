"""
Impact Tracker Module

Tracks policy decisions and media impact influenced by RiskX platform insights.
Monitors when platform data or analysis is referenced in decision-making processes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..core.config import get_settings
from ..cache.cache_manager import CacheManager


@dataclass
class ImpactEvent:
    """Represents a tracked impact event"""
    event_id: str
    event_type: str  # 'policy_decision', 'media_coverage', 'academic_citation'
    source: str
    title: str
    content: str
    url: str
    timestamp: datetime
    relevance_score: float
    impact_category: str  # 'high', 'medium', 'low'
    keywords_matched: List[str]
    

@dataclass
class ImpactMetrics:
    """Aggregated impact metrics"""
    total_events: int
    policy_decisions: int
    media_mentions: int
    academic_citations: int
    high_impact_events: int
    average_relevance: float
    trending_keywords: List[str]
    time_period: str


class ImpactTracker:
    """
    Tracks and measures the real-world impact of RiskX platform.
    
    Monitors:
    - Policy decisions citing platform data
    - Media coverage mentioning platform insights
    - Academic research referencing platform
    - Government agency usage indicators
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Keywords that indicate RiskX impact
        self.risk_keywords = [
            "risk intelligence", "supply chain risk", "financial stability",
            "systemic risk", "economic vulnerability", "disruption prediction",
            "risk assessment", "financial resilience", "supply chain disruption"
        ]
        
        # Sources to monitor for impact
        self.policy_sources = [
            "https://www.federalreserve.gov/feeds/press_all.xml",
            "https://home.treasury.gov/rss/press-releases",
            "https://www.cisa.gov/news.xml",
            "https://www.bea.gov/rss/news_releases.xml"
        ]
        
        self.media_sources = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.wsj.com/xml/rss/3_7014.xml",
            "https://rss.cnn.com/rss/money_news_economy.rss"
        ]
    
    async def track_daily_impact(self) -> ImpactMetrics:
        """
        Run daily impact tracking across all monitored sources
        """
        self.logger.info("Starting daily impact tracking")
        
        try:
            # Get cached baseline metrics
            baseline_metrics = await self._get_baseline_metrics()
            
            # Track policy impact
            policy_events = await self._track_policy_impact()
            
            # Track media coverage
            media_events = await self._track_media_coverage()
            
            # Track academic citations (weekly task, check if needed)
            citation_events = await self._check_citation_updates()
            
            # Aggregate all events
            all_events = policy_events + media_events + citation_events
            
            # Calculate metrics
            metrics = await self._calculate_impact_metrics(all_events, baseline_metrics)
            
            # Cache results
            await self._cache_impact_data(all_events, metrics)
            
            self.logger.info(f"Impact tracking completed. Found {len(all_events)} new events")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in daily impact tracking: {str(e)}")
            raise
    
    async def _track_policy_impact(self) -> List[ImpactEvent]:
        """Track policy decisions and government communications"""
        events = []
        
        async with aiohttp.ClientSession() as session:
            for source_url in self.policy_sources:
                try:
                    events.extend(await self._parse_rss_feed(session, source_url, "policy_decision"))
                except Exception as e:
                    self.logger.warning(f"Failed to parse policy source {source_url}: {str(e)}")
        
        return events
    
    async def _track_media_coverage(self) -> List[ImpactEvent]:
        """Track media mentions and coverage"""
        events = []
        
        async with aiohttp.ClientSession() as session:
            for source_url in self.media_sources:
                try:
                    events.extend(await self._parse_rss_feed(session, source_url, "media_coverage"))
                except Exception as e:
                    self.logger.warning(f"Failed to parse media source {source_url}: {str(e)}")
        
        return events
    
    async def _check_citation_updates(self) -> List[ImpactEvent]:
        """Check for new academic citations (delegated to CitationMonitor)"""
        try:
            from .citation_monitor import CitationMonitor
            citation_monitor = CitationMonitor()
            citations = await citation_monitor.check_new_citations()
            
            # Convert citations to impact events
            events = []
            for citation in citations:
                event = ImpactEvent(
                    event_id=f"citation_{citation.get('id', 'unknown')}",
                    event_type="academic_citation",
                    source=citation.get('journal', 'Unknown'),
                    title=citation.get('title', ''),
                    content=citation.get('abstract', ''),
                    url=citation.get('url', ''),
                    timestamp=datetime.now(),
                    relevance_score=citation.get('relevance_score', 0.5),
                    impact_category=self._categorize_impact(citation.get('relevance_score', 0.5)),
                    keywords_matched=citation.get('keywords_matched', [])
                )
                events.append(event)
            
            return events
            
        except ImportError:
            self.logger.warning("CitationMonitor not available, skipping citation check")
            return []
        except Exception as e:
            self.logger.error(f"Error checking citations: {str(e)}")
            return []
    
    async def _parse_rss_feed(self, session: aiohttp.ClientSession, url: str, event_type: str) -> List[ImpactEvent]:
        """Parse RSS feed and extract relevant events"""
        events = []
        
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:10]:  # Limit to recent entries
                        # Check if entry is relevant to risk intelligence
                        relevance_score, keywords_matched = self._calculate_relevance(
                            entry.get('title', '') + ' ' + entry.get('summary', '')
                        )
                        
                        if relevance_score > 0.3:  # Minimum relevance threshold
                            event = ImpactEvent(
                                event_id=f"{event_type}_{entry.get('id', entry.get('link', 'unknown'))}",
                                event_type=event_type,
                                source=feed.feed.get('title', url),
                                title=entry.get('title', ''),
                                content=entry.get('summary', ''),
                                url=entry.get('link', ''),
                                timestamp=self._parse_timestamp(entry.get('published')),
                                relevance_score=relevance_score,
                                impact_category=self._categorize_impact(relevance_score),
                                keywords_matched=keywords_matched
                            )
                            events.append(event)
        
        except Exception as e:
            self.logger.error(f"Error parsing RSS feed {url}: {str(e)}")
        
        return events
    
    def _calculate_relevance(self, text: str) -> tuple[float, List[str]]:
        """Calculate relevance score and matched keywords"""
        text_lower = text.lower()
        matched_keywords = []
        score = 0.0
        
        for keyword in self.risk_keywords:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)
                # Weight score based on keyword importance
                if "risk intelligence" in keyword or "systemic risk" in keyword:
                    score += 0.3
                elif "financial" in keyword or "supply chain" in keyword:
                    score += 0.2
                else:
                    score += 0.1
        
        # Normalize score to 0-1 range
        score = min(score, 1.0)
        
        return score, matched_keywords
    
    def _categorize_impact(self, relevance_score: float) -> str:
        """Categorize impact level based on relevance score"""
        if relevance_score >= 0.7:
            return "high"
        elif relevance_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp from RSS feed"""
        if not timestamp_str:
            return datetime.now()
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(timestamp_str)
        except:
            return datetime.now()
    
    async def _get_baseline_metrics(self) -> Optional[ImpactMetrics]:
        """Get cached baseline metrics for comparison"""
        try:
            cached_metrics = await self.cache.get("impact_baseline_metrics")
            if cached_metrics:
                return ImpactMetrics(**cached_metrics)
        except Exception as e:
            self.logger.warning(f"Could not retrieve baseline metrics: {str(e)}")
        
        return None
    
    async def _calculate_impact_metrics(self, events: List[ImpactEvent], baseline: Optional[ImpactMetrics]) -> ImpactMetrics:
        """Calculate aggregated impact metrics"""
        if not events:
            return ImpactMetrics(
                total_events=0,
                policy_decisions=0,
                media_mentions=0,
                academic_citations=0,
                high_impact_events=0,
                average_relevance=0.0,
                trending_keywords=[],
                time_period="24h"
            )
        
        # Count events by type
        policy_count = len([e for e in events if e.event_type == "policy_decision"])
        media_count = len([e for e in events if e.event_type == "media_coverage"])
        citation_count = len([e for e in events if e.event_type == "academic_citation"])
        high_impact_count = len([e for e in events if e.impact_category == "high"])
        
        # Calculate average relevance
        avg_relevance = sum(e.relevance_score for e in events) / len(events)
        
        # Extract trending keywords
        all_keywords = []
        for event in events:
            all_keywords.extend(event.keywords_matched)
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Get top trending keywords
        trending_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        trending_keywords = [kw[0] for kw in trending_keywords]
        
        return ImpactMetrics(
            total_events=len(events),
            policy_decisions=policy_count,
            media_mentions=media_count,
            academic_citations=citation_count,
            high_impact_events=high_impact_count,
            average_relevance=avg_relevance,
            trending_keywords=trending_keywords,
            time_period="24h"
        )
    
    async def _cache_impact_data(self, events: List[ImpactEvent], metrics: ImpactMetrics):
        """Cache impact data for future reference"""
        try:
            # Cache events
            events_data = [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "source": e.source,
                    "title": e.title,
                    "content": e.content[:500],  # Truncate content
                    "url": e.url,
                    "timestamp": e.timestamp.isoformat(),
                    "relevance_score": e.relevance_score,
                    "impact_category": e.impact_category,
                    "keywords_matched": e.keywords_matched
                }
                for e in events
            ]
            
            await self.cache.set(
                f"impact_events_{datetime.now().strftime('%Y%m%d')}", 
                events_data, 
                ttl=86400 * 7  # Keep for 7 days
            )
            
            # Cache metrics
            metrics_data = {
                "total_events": metrics.total_events,
                "policy_decisions": metrics.policy_decisions,
                "media_mentions": metrics.media_mentions,
                "academic_citations": metrics.academic_citations,
                "high_impact_events": metrics.high_impact_events,
                "average_relevance": metrics.average_relevance,
                "trending_keywords": metrics.trending_keywords,
                "time_period": metrics.time_period,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.cache.set("impact_latest_metrics", metrics_data, ttl=86400)
            
        except Exception as e:
            self.logger.error(f"Error caching impact data: {str(e)}")
    
    async def get_impact_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get impact summary for specified number of days"""
        try:
            # Aggregate cached data for the period
            all_events = []
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                date_key = f"impact_events_{date.strftime('%Y%m%d')}"
                
                cached_events = await self.cache.get(date_key)
                if cached_events:
                    all_events.extend(cached_events)
            
            # Calculate summary statistics
            summary = {
                "period_days": days,
                "total_events": len(all_events),
                "policy_impact": len([e for e in all_events if e.get("event_type") == "policy_decision"]),
                "media_coverage": len([e for e in all_events if e.get("event_type") == "media_coverage"]),
                "academic_citations": len([e for e in all_events if e.get("event_type") == "academic_citation"]),
                "high_impact_events": len([e for e in all_events if e.get("impact_category") == "high"]),
                "top_sources": self._get_top_sources(all_events),
                "recent_highlights": self._get_recent_highlights(all_events)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating impact summary: {str(e)}")
            return {"error": "Could not generate impact summary"}
    
    def _get_top_sources(self, events: List[Dict]) -> List[Dict[str, Any]]:
        """Get top sources by event count"""
        source_counts = {}
        for event in events:
            source = event.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"source": source, "count": count} for source, count in top_sources]
    
    def _get_recent_highlights(self, events: List[Dict], limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent high-impact events"""
        # Filter for high relevance events
        high_relevance = [e for e in events if e.get("relevance_score", 0) > 0.6]
        
        # Sort by timestamp (most recent first)
        high_relevance.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Return top highlights
        highlights = []
        for event in high_relevance[:limit]:
            highlights.append({
                "title": event.get("title", ""),
                "source": event.get("source", ""),
                "url": event.get("url", ""),
                "relevance_score": event.get("relevance_score", 0),
                "timestamp": event.get("timestamp", "")
            })
        
        return highlights