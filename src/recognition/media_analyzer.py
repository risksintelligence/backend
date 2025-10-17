"""
Media Analyzer Module

Analyzes news mentions and media coverage of the RiskX platform.
Tracks sentiment, reach, and influence across various media sources.
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
class MediaMention:
    """Represents a media mention"""
    mention_id: str
    source: str
    title: str
    content: str
    url: str
    author: Optional[str]
    publication_date: datetime
    sentiment_score: float  # -1 to 1
    reach_estimate: int  # Estimated audience reach
    media_type: str  # 'news', 'blog', 'social', 'press_release'
    prominence_score: float  # How prominently RiskX is featured
    context_category: str  # 'announcement', 'analysis', 'criticism', 'endorsement'
    keywords_matched: List[str]


@dataclass
class MediaMetrics:
    """Aggregated media metrics"""
    total_mentions: int
    positive_sentiment: int
    negative_sentiment: int
    neutral_sentiment: int
    average_sentiment: float
    total_estimated_reach: int
    top_sources: List[Dict[str, Any]]
    trending_topics: List[str]
    prominence_distribution: Dict[str, int]
    media_type_breakdown: Dict[str, int]


class MediaAnalyzer:
    """
    Analyzes media coverage and news mentions of RiskX platform.
    
    Tracks coverage across:
    - Financial news outlets
    - Technology publications
    - Government press releases
    - Academic news
    - Risk management publications
    - General business media
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Search terms for media monitoring
        self.search_terms = [
            "RiskX", "Risk Intelligence Observatory", 
            "AI risk intelligence", "systemic risk prediction",
            "supply chain risk intelligence", "financial stability AI"
        ]
        
        # Media sources to monitor
        self.media_sources = {
            # Financial News
            "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "wsj": "https://www.wsj.com/xml/rss/3_7014.xml",
            "ft": "https://www.ft.com/rss/home/us",
            
            # Technology
            "techcrunch": "https://techcrunch.com/feed/",
            "venturebeat": "https://venturebeat.com/feed/",
            "wired": "https://www.wired.com/feed/rss",
            
            # Government & Policy
            "fed_news": "https://www.federalreserve.gov/feeds/press_all.xml",
            "treasury": "https://home.treasury.gov/rss/press-releases",
            "cisa": "https://www.cisa.gov/news.xml",
            
            # Business & Finance
            "fortune": "https://fortune.com/feed/",
            "forbes": "https://www.forbes.com/innovation/feed2/",
            "economist": "https://www.economist.com/rss/finance_and_economics_rss.xml"
        }
        
        # Estimated reach for different sources (rough estimates)
        self.source_reach_estimates = {
            "reuters": 50000000,
            "bloomberg": 30000000,
            "wsj": 25000000,
            "ft": 15000000,
            "techcrunch": 20000000,
            "wired": 10000000,
            "fortune": 8000000,
            "forbes": 12000000,
            "economist": 20000000,
            "default": 5000000
        }
    
    async def analyze_daily_coverage(self) -> MediaMetrics:
        """
        Analyze daily media coverage across all monitored sources
        """
        self.logger.info("Starting daily media coverage analysis")
        
        try:
            all_mentions = []
            
            # Check each media source
            for source_name, feed_url in self.media_sources.items():
                try:
                    mentions = await self._analyze_media_source(source_name, feed_url)
                    all_mentions.extend(mentions)
                    self.logger.info(f"Found {len(mentions)} mentions from {source_name}")
                except Exception as e:
                    self.logger.warning(f"Error analyzing {source_name}: {str(e)}")
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Filter for new mentions
            new_mentions = await self._filter_new_mentions(all_mentions)
            
            # Calculate metrics
            metrics = self._calculate_media_metrics(new_mentions)
            
            # Cache results
            if new_mentions:
                await self._cache_media_data(new_mentions, metrics)
            
            self.logger.info(f"Media analysis completed. Found {len(new_mentions)} new mentions")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in daily media coverage analysis: {str(e)}")
            raise
    
    async def _analyze_media_source(self, source_name: str, feed_url: str) -> List[MediaMention]:
        """Analyze a specific media source for mentions"""
        mentions = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(feed_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:20]:  # Limit to recent entries
                            mention = await self._process_media_entry(entry, source_name)
                            if mention:
                                mentions.append(mention)
            
            except Exception as e:
                self.logger.error(f"Error fetching {source_name}: {str(e)}")
        
        return mentions
    
    async def _process_media_entry(self, entry: Any, source_name: str) -> Optional[MediaMention]:
        """Process a single media entry to extract mention data"""
        try:
            title = entry.get('title', '')
            content = entry.get('summary', '') or entry.get('description', '')
            full_text = f"{title} {content}".lower()
            
            # Check if any search terms are mentioned
            matched_terms = []
            for term in self.search_terms:
                if term.lower() in full_text:
                    matched_terms.append(term)
            
            if not matched_terms:
                return None  # No relevant mentions found
            
            # Extract metadata
            url = entry.get('link', '')
            author = self._extract_author(entry)
            pub_date = self._parse_publication_date(entry.get('published'))
            
            # Analyze sentiment
            sentiment_score = self._analyze_sentiment(title, content)
            
            # Calculate prominence score
            prominence_score = self._calculate_prominence(title, content, matched_terms)
            
            # Estimate reach
            reach_estimate = self._estimate_reach(source_name)
            
            # Categorize context
            context_category = self._categorize_context(title, content)
            
            # Determine media type
            media_type = self._determine_media_type(source_name, entry)
            
            mention = MediaMention(
                mention_id=f"{source_name}_{hash(url)}_{pub_date.strftime('%Y%m%d')}",
                source=source_name,
                title=title,
                content=content[:1000],  # Truncate content
                url=url,
                author=author,
                publication_date=pub_date,
                sentiment_score=sentiment_score,
                reach_estimate=reach_estimate,
                media_type=media_type,
                prominence_score=prominence_score,
                context_category=context_category,
                keywords_matched=matched_terms
            )
            
            return mention
            
        except Exception as e:
            self.logger.warning(f"Error processing media entry: {str(e)}")
            return None
    
    def _extract_author(self, entry: Any) -> Optional[str]:
        """Extract author from feed entry"""
        author = entry.get('author')
        if author:
            return author
        
        # Try alternative fields
        author_detail = entry.get('author_detail', {})
        if author_detail.get('name'):
            return author_detail['name']
        
        return None
    
    def _parse_publication_date(self, date_str: Optional[str]) -> datetime:
        """Parse publication date from various formats"""
        if not date_str:
            return datetime.now()
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            # Fallback to current time
            return datetime.now()
    
    def _analyze_sentiment(self, title: str, content: str) -> float:
        """
        Analyze sentiment of the mention (simplified implementation)
        Returns score from -1 (negative) to 1 (positive)
        """
        text = f"{title} {content}".lower()
        
        # Positive indicators
        positive_words = [
            'breakthrough', 'innovative', 'successful', 'effective', 'promising',
            'advancement', 'improvement', 'solution', 'reliable', 'accurate',
            'transparency', 'trustworthy', 'valuable', 'beneficial', 'progress'
        ]
        
        # Negative indicators
        negative_words = [
            'failure', 'problem', 'concern', 'risk', 'criticism', 'flawed',
            'inadequate', 'unreliable', 'biased', 'dangerous', 'controversy',
            'limitation', 'shortcoming', 'error', 'mistake'
        ]
        
        # Neutral/technical indicators
        neutral_words = [
            'analysis', 'study', 'research', 'data', 'methodology', 'framework',
            'system', 'platform', 'algorithm', 'model', 'implementation'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        neutral_count = sum(1 for word in neutral_words if word in text)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        # Calculate weighted sentiment
        sentiment = (positive_count - negative_count) / total_sentiment_words
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_prominence(self, title: str, content: str, matched_terms: List[str]) -> float:
        """
        Calculate how prominently RiskX is featured in the article
        Returns score from 0 to 1
        """
        score = 0.0
        
        # Check if mentioned in title
        title_lower = title.lower()
        for term in matched_terms:
            if term.lower() in title_lower:
                score += 0.5  # High prominence if in title
        
        # Check position in content
        content_lower = content.lower()
        content_words = content_lower.split()
        
        for term in matched_terms:
            term_lower = term.lower()
            if term_lower in content_lower:
                # Find position of first mention
                words_before_mention = len(content_lower[:content_lower.find(term_lower)].split())
                total_words = len(content_words)
                
                if total_words > 0:
                    # Earlier mentions get higher scores
                    position_score = 1 - (words_before_mention / total_words)
                    score += position_score * 0.3
        
        # Check frequency of mentions
        total_mentions = sum(content_lower.count(term.lower()) for term in matched_terms)
        if total_mentions > 1:
            score += min(0.2, total_mentions * 0.05)
        
        return min(1.0, score)
    
    def _estimate_reach(self, source_name: str) -> int:
        """Estimate audience reach based on source"""
        for source_key, reach in self.source_reach_estimates.items():
            if source_key in source_name.lower():
                return reach
        return self.source_reach_estimates["default"]
    
    def _categorize_context(self, title: str, content: str) -> str:
        """Categorize the context of the mention"""
        text = f"{title} {content}".lower()
        
        # Context patterns
        if any(word in text for word in ['announces', 'launches', 'introduces', 'releases']):
            return 'announcement'
        elif any(word in text for word in ['analyzes', 'examines', 'studies', 'evaluates']):
            return 'analysis'
        elif any(word in text for word in ['criticizes', 'questions', 'challenges', 'concerns']):
            return 'criticism'
        elif any(word in text for word in ['endorses', 'recommends', 'praises', 'supports']):
            return 'endorsement'
        elif any(word in text for word in ['uses', 'adopts', 'implements', 'integrates']):
            return 'adoption'
        else:
            return 'general'
    
    def _determine_media_type(self, source_name: str, entry: Any) -> str:
        """Determine the type of media"""
        source_lower = source_name.lower()
        
        if any(word in source_lower for word in ['fed', 'treasury', 'cisa', 'gov']):
            return 'government'
        elif any(word in source_lower for word in ['reuters', 'bloomberg', 'wsj', 'ft']):
            return 'financial_news'
        elif any(word in source_lower for word in ['tech', 'wired', 'venture']):
            return 'technology'
        elif any(word in source_lower for word in ['fortune', 'forbes', 'economist']):
            return 'business'
        else:
            return 'general_news'
    
    async def _filter_new_mentions(self, mentions: List[MediaMention]) -> List[MediaMention]:
        """Filter out mentions that have already been tracked"""
        if not mentions:
            return []
        
        try:
            # Get existing mention IDs from cache
            existing_ids = await self.cache.get("tracked_mention_ids") or set()
            
            # Filter for new mentions
            new_mentions = []
            new_ids = set()
            
            for mention in mentions:
                if mention.mention_id not in existing_ids:
                    new_mentions.append(mention)
                    new_ids.add(mention.mention_id)
            
            # Update tracked IDs
            if new_ids:
                updated_ids = existing_ids.union(new_ids)
                await self.cache.set("tracked_mention_ids", list(updated_ids), ttl=86400 * 30)
            
            return new_mentions
            
        except Exception as e:
            self.logger.error(f"Error filtering new mentions: {str(e)}")
            return mentions  # Return all if filtering fails
    
    def _calculate_media_metrics(self, mentions: List[MediaMention]) -> MediaMetrics:
        """Calculate aggregated media metrics"""
        if not mentions:
            return MediaMetrics(
                total_mentions=0,
                positive_sentiment=0,
                negative_sentiment=0,
                neutral_sentiment=0,
                average_sentiment=0.0,
                total_estimated_reach=0,
                top_sources=[],
                trending_topics=[],
                prominence_distribution={},
                media_type_breakdown={}
            )
        
        # Sentiment breakdown
        positive_count = len([m for m in mentions if m.sentiment_score > 0.1])
        negative_count = len([m for m in mentions if m.sentiment_score < -0.1])
        neutral_count = len(mentions) - positive_count - negative_count
        
        # Average sentiment
        avg_sentiment = sum(m.sentiment_score for m in mentions) / len(mentions)
        
        # Total reach
        total_reach = sum(m.reach_estimate for m in mentions)
        
        # Top sources
        source_counts = {}
        for mention in mentions:
            source_counts[mention.source] = source_counts.get(mention.source, 0) + 1
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_sources = [{"source": source, "count": count} for source, count in top_sources]
        
        # Trending topics (keywords)
        all_keywords = []
        for mention in mentions:
            all_keywords.extend(mention.keywords_matched)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        trending_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        trending_topics = [topic[0] for topic in trending_topics]
        
        # Prominence distribution
        prominence_dist = {"high": 0, "medium": 0, "low": 0}
        for mention in mentions:
            if mention.prominence_score >= 0.7:
                prominence_dist["high"] += 1
            elif mention.prominence_score >= 0.3:
                prominence_dist["medium"] += 1
            else:
                prominence_dist["low"] += 1
        
        # Media type breakdown
        type_counts = {}
        for mention in mentions:
            type_counts[mention.media_type] = type_counts.get(mention.media_type, 0) + 1
        
        return MediaMetrics(
            total_mentions=len(mentions),
            positive_sentiment=positive_count,
            negative_sentiment=negative_count,
            neutral_sentiment=neutral_count,
            average_sentiment=avg_sentiment,
            total_estimated_reach=total_reach,
            top_sources=top_sources,
            trending_topics=trending_topics,
            prominence_distribution=prominence_dist,
            media_type_breakdown=type_counts
        )
    
    async def _cache_media_data(self, mentions: List[MediaMention], metrics: MediaMetrics):
        """Cache media analysis data"""
        try:
            # Cache mentions
            mention_data = []
            for mention in mentions:
                data = {
                    "mention_id": mention.mention_id,
                    "source": mention.source,
                    "title": mention.title,
                    "content": mention.content,
                    "url": mention.url,
                    "author": mention.author,
                    "publication_date": mention.publication_date.isoformat(),
                    "sentiment_score": mention.sentiment_score,
                    "reach_estimate": mention.reach_estimate,
                    "media_type": mention.media_type,
                    "prominence_score": mention.prominence_score,
                    "context_category": mention.context_category,
                    "keywords_matched": mention.keywords_matched
                }
                mention_data.append(data)
            
            # Cache today's mentions
            date_key = f"media_mentions_{datetime.now().strftime('%Y%m%d')}"
            await self.cache.set(date_key, mention_data, ttl=86400 * 30)
            
            # Cache metrics
            metrics_data = {
                "total_mentions": metrics.total_mentions,
                "positive_sentiment": metrics.positive_sentiment,
                "negative_sentiment": metrics.negative_sentiment,
                "neutral_sentiment": metrics.neutral_sentiment,
                "average_sentiment": metrics.average_sentiment,
                "total_estimated_reach": metrics.total_estimated_reach,
                "top_sources": metrics.top_sources,
                "trending_topics": metrics.trending_topics,
                "prominence_distribution": metrics.prominence_distribution,
                "media_type_breakdown": metrics.media_type_breakdown,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.cache.set("media_latest_metrics", metrics_data, ttl=86400)
            
        except Exception as e:
            self.logger.error(f"Error caching media data: {str(e)}")
    
    async def get_media_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get media coverage summary for specified number of days"""
        try:
            all_mentions = []
            
            # Aggregate mentions for the period
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                date_key = f"media_mentions_{date.strftime('%Y%m%d')}"
                
                cached_mentions = await self.cache.get(date_key)
                if cached_mentions:
                    all_mentions.extend(cached_mentions)
            
            # Calculate summary statistics
            if not all_mentions:
                return {"error": "No media data available for specified period"}
            
            # Sentiment analysis
            positive_mentions = [m for m in all_mentions if m.get("sentiment_score", 0) > 0.1]
            negative_mentions = [m for m in all_mentions if m.get("sentiment_score", 0) < -0.1]
            
            # Reach analysis
            total_reach = sum(m.get("reach_estimate", 0) for m in all_mentions)
            high_reach_mentions = [m for m in all_mentions if m.get("reach_estimate", 0) > 10000000]
            
            # Recent highlights
            recent_highlights = sorted(
                all_mentions, 
                key=lambda x: (x.get("prominence_score", 0), x.get("reach_estimate", 0)), 
                reverse=True
            )[:5]
            
            summary = {
                "period_days": days,
                "total_mentions": len(all_mentions),
                "sentiment_breakdown": {
                    "positive": len(positive_mentions),
                    "negative": len(negative_mentions),
                    "neutral": len(all_mentions) - len(positive_mentions) - len(negative_mentions)
                },
                "total_estimated_reach": total_reach,
                "high_reach_mentions": len(high_reach_mentions),
                "average_sentiment": sum(m.get("sentiment_score", 0) for m in all_mentions) / len(all_mentions),
                "top_sources": self._get_top_media_sources(all_mentions),
                "recent_highlights": [
                    {
                        "title": h.get("title", ""),
                        "source": h.get("source", ""),
                        "url": h.get("url", ""),
                        "sentiment_score": h.get("sentiment_score", 0),
                        "reach_estimate": h.get("reach_estimate", 0),
                        "publication_date": h.get("publication_date", "")
                    }
                    for h in recent_highlights
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating media summary: {str(e)}")
            return {"error": "Could not generate media summary"}
    
    def _get_top_media_sources(self, mentions: List[Dict]) -> List[Dict[str, Any]]:
        """Get top media sources by mention count and reach"""
        source_stats = {}
        
        for mention in mentions:
            source = mention.get("source", "Unknown")
            if source not in source_stats:
                source_stats[source] = {
                    "count": 0,
                    "total_reach": 0,
                    "avg_sentiment": 0,
                    "sentiment_sum": 0
                }
            
            source_stats[source]["count"] += 1
            source_stats[source]["total_reach"] += mention.get("reach_estimate", 0)
            source_stats[source]["sentiment_sum"] += mention.get("sentiment_score", 0)
        
        # Calculate averages
        for source, stats in source_stats.items():
            stats["avg_sentiment"] = stats["sentiment_sum"] / stats["count"]
            del stats["sentiment_sum"]  # Remove temporary field
        
        # Sort by combined score (count * reach)
        sorted_sources = sorted(
            source_stats.items(),
            key=lambda x: x[1]["count"] * x[1]["total_reach"],
            reverse=True
        )
        
        return [{"source": source, **stats} for source, stats in sorted_sources[:5]]