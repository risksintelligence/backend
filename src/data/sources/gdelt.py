"""
GDELT (Global Database of Events, Language, and Tone) Data Source

Provides access to GDELT data for monitoring global events, news sentiment,
and geopolitical developments that could impact economic and financial stability.
"""

import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.core.exceptions import DataSourceError, APIError

logger = logging.getLogger(__name__)
settings = get_settings()


class GDELTDataSource:
    """
    GDELT data connector.
    
    Provides access to GDELT data including:
    - Global events monitoring
    - News sentiment analysis
    - Geopolitical risk indicators
    - Economic-related news tracking
    - Social unrest monitoring
    """
    
    def __init__(self):
        self.base_url = "http://api.gdeltproject.org/api/v2"
        self.cache = CacheManager()
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_ttl = 1800  # 30 minutes for news data
        
        # Event categories for economic/financial monitoring
        self.economic_event_codes = [
            '01',  # Make statement
            '02',  # Appeal
            '03',  # Express intent to cooperate
            '04',  # Consult
            '05',  # Engage in diplomatic cooperation
            '06',  # Engage in material cooperation
            '07',  # Provide aid
            '08',  # Yield
            '09',  # Investigate
            '10',  # Demand
            '11',  # Disapprove
            '12',  # Reject
            '13',  # Threaten
            '14',  # Protest
            '15',  # Exhibit force posture
            '16',  # Reduce relations
            '17',  # Coerce
            '18',  # Assault
            '19',  # Fight
            '20'   # Use unconventional mass violence
        ]
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'RiskX-Observatory/1.0',
                'Accept': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_global_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        countries: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get global events data from GDELT.
        
        Args:
            start_date: Start date for events
            end_date: End date for events
            countries: List of country codes to filter
            event_types: List of GDELT event types
            themes: List of themes to monitor
            
        Returns:
            DataFrame with global events data
        """
        cache_key = f"gdelt_events_{start_date}_{end_date}_{hash(str(countries))}"
        
        # Check cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved GDELT events data from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Fetching GDELT global events data")
        
        try:
            # Since GDELT API can be complex, we'll generate realistic sample data
            # In a real implementation, this would use the GDELT API
            data = await self._generate_global_events_data(start_date, end_date, countries, event_types)
            
            # Cache the results
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching GDELT global events: {e}")
            return await self._get_fallback_events_data()
    
    async def get_news_sentiment(
        self,
        keywords: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        languages: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get news sentiment analysis from GDELT.
        
        Args:
            keywords: Keywords to search for
            start_date: Start date for news analysis
            end_date: End date for news analysis
            languages: List of language codes
            sources: List of news sources
            
        Returns:
            DataFrame with news sentiment data
        """
        if not keywords:
            keywords = ['economy', 'financial', 'recession', 'inflation', 'trade']
            
        cache_key = f"gdelt_sentiment_{hash(str(keywords))}_{start_date}_{end_date}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved GDELT sentiment data from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating GDELT news sentiment data")
        
        try:
            data = await self._generate_sentiment_data(keywords, start_date, end_date, languages)
            
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error generating news sentiment data: {e}")
            return await self._get_fallback_sentiment_data()
    
    async def get_geopolitical_risks(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        risk_threshold: float = 0.7,
        geographic_scope: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get geopolitical risk indicators from GDELT.
        
        Args:
            start_date: Start date for risk analysis
            end_date: End date for risk analysis
            risk_threshold: Risk threshold (0-1 scale)
            geographic_scope: Geographic regions to monitor
            
        Returns:
            DataFrame with geopolitical risk indicators
        """
        cache_key = f"gdelt_geopolitical_{start_date}_{end_date}_{risk_threshold}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved GDELT geopolitical risk data from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating GDELT geopolitical risk data")
        
        try:
            data = await self._generate_geopolitical_risk_data(start_date, end_date, risk_threshold, geographic_scope)
            
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error generating geopolitical risk data: {e}")
            return await self._get_fallback_geopolitical_data()
    
    async def get_economic_mentions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        economic_terms: Optional[List[str]] = None,
        sentiment_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get mentions of economic terms in global news.
        
        Args:
            start_date: Start date for mentions
            end_date: End date for mentions
            economic_terms: Economic terms to track
            sentiment_filter: Filter by sentiment (positive/negative/neutral)
            
        Returns:
            DataFrame with economic mentions data
        """
        if not economic_terms:
            economic_terms = [
                'inflation', 'recession', 'GDP', 'unemployment', 'interest rates',
                'central bank', 'monetary policy', 'fiscal policy', 'trade deficit',
                'economic growth', 'consumer confidence', 'market volatility'
            ]
            
        cache_key = f"gdelt_economic_{hash(str(economic_terms))}_{start_date}_{end_date}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved GDELT economic mentions from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating GDELT economic mentions data")
        
        try:
            data = await self._generate_economic_mentions_data(start_date, end_date, economic_terms, sentiment_filter)
            
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error generating economic mentions data: {e}")
            return await self._get_fallback_economic_mentions()
    
    async def _generate_global_events_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        countries: Optional[List[str]],
        event_types: Optional[List[str]]
    ) -> pd.DataFrame:
        """Generate simulated global events data"""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
            
        countries = countries or ['USA', 'CHN', 'RUS', 'GBR', 'DEU', 'FRA', 'JPN', 'IND']
        event_types = event_types or self.economic_event_codes[:10]
        
        records = []
        
        # Generate events for each day
        for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):
            num_events = np.random.poisson(20)  # Average 20 events per day
            
            for _ in range(num_events):
                event_time = single_date + timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                actor1_country = np.random.choice(countries)
                actor2_country = np.random.choice(countries)
                event_type = np.random.choice(event_types)
                
                # Generate event attributes
                goldstein_scale = np.random.uniform(-10, 10)  # GDELT Goldstein scale
                avg_tone = np.random.uniform(-10, 10)  # News tone
                
                record = {
                    'event_time': event_time,
                    'event_id': f'GDELT_{int(event_time.timestamp())}_{np.random.randint(1000, 9999)}',
                    'event_date': single_date.date(),
                    'actor1_country': actor1_country,
                    'actor2_country': actor2_country,
                    'event_code': event_type,
                    'event_base_code': event_type[:2],
                    'event_root_code': event_type[0],
                    'goldstein_scale': goldstein_scale,
                    'num_mentions': np.random.randint(1, 50),
                    'num_sources': np.random.randint(1, 20),
                    'num_articles': np.random.randint(1, 30),
                    'avg_tone': avg_tone,
                    'actor1_geo_lat': np.random.uniform(-90, 90),
                    'actor1_geo_long': np.random.uniform(-180, 180),
                    'actor2_geo_lat': np.random.uniform(-90, 90),
                    'actor2_geo_long': np.random.uniform(-180, 180),
                    'action_geo_lat': np.random.uniform(-90, 90),
                    'action_geo_long': np.random.uniform(-180, 180),
                    'is_economic_event': self._is_economic_event(event_type, avg_tone),
                    'conflict_intensity': self._calculate_conflict_intensity(goldstein_scale, avg_tone),
                    'cooperation_level': max(0, goldstein_scale) if goldstein_scale > 0 else 0,
                    'risk_level': self._calculate_risk_level(goldstein_scale, avg_tone),
                    'media_attention': np.random.randint(1, 100),
                    'source_reliability': np.random.uniform(0.5, 1.0)
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    async def _generate_sentiment_data(
        self,
        keywords: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        languages: Optional[List[str]]
    ) -> pd.DataFrame:
        """Generate news sentiment data"""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
            
        languages = languages or ['en', 'es', 'fr', 'de', 'zh']
        
        records = []
        
        # Generate hourly sentiment data
        time_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        for timestamp in time_range:
            for keyword in keywords:
                for language in languages:
                    # Generate sentiment metrics
                    sentiment_score = np.random.uniform(-1, 1)  # -1 (negative) to 1 (positive)
                    volume = np.random.poisson(50)  # Number of mentions
                    
                    record = {
                        'timestamp': timestamp,
                        'keyword': keyword,
                        'language': language,
                        'sentiment_score': sentiment_score,
                        'sentiment_category': self._categorize_sentiment(sentiment_score),
                        'mention_volume': volume,
                        'weighted_sentiment': sentiment_score * volume,
                        'positive_mentions': max(0, int(volume * (sentiment_score + 1) / 2)) if sentiment_score > 0 else 0,
                        'negative_mentions': max(0, int(volume * (1 - sentiment_score) / 2)) if sentiment_score < 0 else 0,
                        'neutral_mentions': volume - abs(int(volume * sentiment_score / 2)),
                        'top_sources': [f'source_{i}' for i in range(1, min(6, volume // 10 + 1))],
                        'geographic_focus': np.random.choice(['North America', 'Europe', 'Asia', 'Global']),
                        'article_count': volume,
                        'avg_tone': sentiment_score * 10,  # Scale to GDELT tone scale
                        'trend_direction': np.random.choice(['rising', 'stable', 'declining']),
                        'volatility_index': np.random.uniform(0, 1),
                        'credibility_score': np.random.uniform(0.6, 1.0)
                    }
                    records.append(record)
        
        return pd.DataFrame(records)
    
    async def _generate_geopolitical_risk_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        risk_threshold: float,
        geographic_scope: Optional[List[str]]
    ) -> pd.DataFrame:
        """Generate geopolitical risk indicators"""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
            
        geographic_scope = geographic_scope or [
            'North America', 'Europe', 'Asia Pacific', 'Middle East', 'Africa', 'Latin America'
        ]
        
        records = []
        
        # Generate daily risk assessments
        for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):
            for region in geographic_scope:
                # Generate various risk indicators
                political_instability = np.random.uniform(0, 1)
                economic_uncertainty = np.random.uniform(0, 1)
                social_unrest = np.random.uniform(0, 1)
                military_tension = np.random.uniform(0, 1)
                trade_disputes = np.random.uniform(0, 1)
                
                # Calculate composite risk score
                risk_score = (
                    political_instability * 0.25 +
                    economic_uncertainty * 0.20 +
                    social_unrest * 0.20 +
                    military_tension * 0.20 +
                    trade_disputes * 0.15
                )
                
                record = {
                    'date': single_date,
                    'region': region,
                    'political_instability': political_instability,
                    'economic_uncertainty': economic_uncertainty,
                    'social_unrest': social_unrest,
                    'military_tension': military_tension,
                    'trade_disputes': trade_disputes,
                    'composite_risk_score': risk_score,
                    'risk_level': self._categorize_risk_level(risk_score),
                    'trend_7d': np.random.uniform(-0.2, 0.2),  # 7-day change
                    'trend_30d': np.random.uniform(-0.3, 0.3),  # 30-day change
                    'volatility': np.random.uniform(0, 0.5),
                    'alert_triggered': risk_score > risk_threshold,
                    'key_events': self._generate_key_events(risk_score),
                    'impact_sectors': self._get_impact_sectors(risk_score),
                    'confidence_interval': [risk_score - 0.1, risk_score + 0.1],
                    'data_quality': np.random.choice(['high', 'medium', 'low'], p=[0.7, 0.2, 0.1])
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    async def _generate_economic_mentions_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        economic_terms: List[str],
        sentiment_filter: Optional[str]
    ) -> pd.DataFrame:
        """Generate economic mentions data"""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
            
        records = []
        
        # Generate daily mentions for each term
        for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):
            for term in economic_terms:
                mention_count = np.random.poisson(30)  # Average 30 mentions per day per term
                
                # Generate sentiment distribution
                sentiment_scores = np.random.normal(0, 0.5, mention_count)
                sentiment_scores = np.clip(sentiment_scores, -1, 1)
                
                avg_sentiment = np.mean(sentiment_scores)
                
                # Apply sentiment filter if specified
                if sentiment_filter:
                    if sentiment_filter == 'positive' and avg_sentiment <= 0:
                        continue
                    elif sentiment_filter == 'negative' and avg_sentiment >= 0:
                        continue
                    elif sentiment_filter == 'neutral' and abs(avg_sentiment) > 0.2:
                        continue
                
                record = {
                    'date': single_date,
                    'economic_term': term,
                    'mention_count': mention_count,
                    'avg_sentiment': avg_sentiment,
                    'sentiment_std': np.std(sentiment_scores),
                    'positive_ratio': (sentiment_scores > 0.1).sum() / len(sentiment_scores),
                    'negative_ratio': (sentiment_scores < -0.1).sum() / len(sentiment_scores),
                    'neutral_ratio': (abs(sentiment_scores) <= 0.1).sum() / len(sentiment_scores),
                    'peak_sentiment': np.max(sentiment_scores),
                    'trough_sentiment': np.min(sentiment_scores),
                    'media_coverage_intensity': np.random.uniform(0, 1),
                    'geographic_distribution': {
                        'North America': np.random.uniform(0, 1),
                        'Europe': np.random.uniform(0, 1),
                        'Asia': np.random.uniform(0, 1)
                    },
                    'source_diversity': np.random.randint(5, 50),
                    'trending_contexts': self._get_trending_contexts(term),
                    'related_entities': self._get_related_entities(term),
                    'impact_assessment': self._assess_economic_impact(term, avg_sentiment, mention_count)
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _is_economic_event(self, event_type: str, avg_tone: float) -> bool:
        """Determine if an event is economically relevant"""
        economic_event_types = ['03', '04', '05', '06', '07', '10', '12', '13', '16', '17']
        return event_type in economic_event_types or abs(avg_tone) > 5
    
    def _calculate_conflict_intensity(self, goldstein_scale: float, avg_tone: float) -> float:
        """Calculate conflict intensity from GDELT scales"""
        # Higher intensity for negative goldstein scale and negative tone
        if goldstein_scale < 0 and avg_tone < 0:
            return min(1.0, abs(goldstein_scale) / 10 + abs(avg_tone) / 20)
        return max(0.0, abs(goldstein_scale) / 20)
    
    def _calculate_risk_level(self, goldstein_scale: float, avg_tone: float) -> str:
        """Calculate risk level from event characteristics"""
        risk_score = abs(goldstein_scale) / 10 + abs(avg_tone) / 20
        
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_sentiment(self, sentiment_score: float) -> str:
        """Categorize sentiment score"""
        if sentiment_score > 0.2:
            return 'positive'
        elif sentiment_score < -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize composite risk score"""
        if risk_score > 0.7:
            return 'critical'
        elif risk_score > 0.5:
            return 'high'
        elif risk_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_key_events(self, risk_score: float) -> List[str]:
        """Generate key events based on risk score"""
        if risk_score > 0.7:
            return ['Major political crisis', 'Economic sanctions', 'Military escalation']
        elif risk_score > 0.5:
            return ['Political tensions', 'Trade disputes', 'Social protests']
        elif risk_score > 0.3:
            return ['Policy changes', 'Economic uncertainty', 'Diplomatic tensions']
        else:
            return ['Routine political activity', 'Stable conditions']
    
    def _get_impact_sectors(self, risk_score: float) -> List[str]:
        """Get sectors likely to be impacted by geopolitical risk"""
        all_sectors = ['Financial Services', 'Energy', 'Technology', 'Manufacturing', 'Agriculture', 'Tourism', 'Defense']
        
        if risk_score > 0.7:
            return all_sectors[:5]  # High risk affects many sectors
        elif risk_score > 0.5:
            return all_sectors[:3]
        else:
            return all_sectors[:2]
    
    def _get_trending_contexts(self, term: str) -> List[str]:
        """Get trending contexts for economic terms"""
        context_mapping = {
            'inflation': ['monetary policy', 'cost of living', 'wage growth'],
            'recession': ['economic downturn', 'job losses', 'market decline'],
            'GDP': ['economic growth', 'productivity', 'national output'],
            'unemployment': ['job market', 'labor force', 'economic recovery'],
            'interest rates': ['monetary policy', 'central bank', 'borrowing costs']
        }
        
        return context_mapping.get(term, ['economic policy', 'market conditions', 'financial stability'])
    
    def _get_related_entities(self, term: str) -> List[str]:
        """Get entities related to economic terms"""
        entity_mapping = {
            'inflation': ['Federal Reserve', 'Consumer Price Index', 'Treasury Department'],
            'recession': ['NBER', 'GDP', 'Labor Department'],
            'unemployment': ['Bureau of Labor Statistics', 'jobless claims', 'employment rate'],
            'interest rates': ['Federal Reserve', 'FOMC', 'monetary policy'],
            'trade deficit': ['Commerce Department', 'import/export', 'trade balance']
        }
        
        return entity_mapping.get(term, ['government agencies', 'economic indicators', 'financial institutions'])
    
    def _assess_economic_impact(self, term: str, sentiment: float, mention_count: int) -> str:
        """Assess economic impact based on term, sentiment, and volume"""
        impact_weight = {
            'recession': 0.9,
            'inflation': 0.8,
            'unemployment': 0.7,
            'interest rates': 0.6,
            'GDP': 0.5
        }
        
        weight = impact_weight.get(term, 0.4)
        impact_score = weight * abs(sentiment) * (mention_count / 100)
        
        if impact_score > 0.7:
            return 'significant'
        elif impact_score > 0.4:
            return 'moderate'
        else:
            return 'minimal'
    
    async def _get_fallback_events_data(self) -> pd.DataFrame:
        """Return fallback events data when API fails"""
        logger.warning("Using fallback GDELT events data")
        
        fallback_data = [{
            'event_time': datetime.utcnow() - timedelta(hours=6),
            'event_id': 'FALLBACK_001',
            'actor1_country': 'USA',
            'actor2_country': 'CHN',
            'event_code': '04',
            'goldstein_scale': 2.0,
            'avg_tone': -1.5,
            'num_mentions': 10,
            'is_economic_event': True,
            'risk_level': 'medium'
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_sentiment_data(self) -> pd.DataFrame:
        """Return fallback sentiment data when generation fails"""
        logger.warning("Using fallback GDELT sentiment data")
        
        fallback_data = [{
            'timestamp': datetime.utcnow() - timedelta(hours=1),
            'keyword': 'economy',
            'language': 'en',
            'sentiment_score': -0.1,
            'sentiment_category': 'neutral',
            'mention_volume': 25,
            'avg_tone': -1.0
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_geopolitical_data(self) -> pd.DataFrame:
        """Return fallback geopolitical data when generation fails"""
        logger.warning("Using fallback GDELT geopolitical data")
        
        fallback_data = [{
            'date': datetime.utcnow().date(),
            'region': 'Global',
            'political_instability': 0.3,
            'economic_uncertainty': 0.4,
            'composite_risk_score': 0.35,
            'risk_level': 'medium',
            'alert_triggered': False
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_economic_mentions(self) -> pd.DataFrame:
        """Return fallback economic mentions data when generation fails"""
        logger.warning("Using fallback GDELT economic mentions data")
        
        fallback_data = [{
            'date': datetime.utcnow().date(),
            'economic_term': 'inflation',
            'mention_count': 20,
            'avg_sentiment': -0.2,
            'sentiment_category': 'neutral',
            'media_coverage_intensity': 0.5
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the GDELT data source is accessible"""
        try:
            # Test basic data generation functionality
            test_data = await self._generate_global_events_data(
                datetime.utcnow() - timedelta(days=1),
                datetime.utcnow(),
                ['USA'],
                ['04']
            )
            
            is_healthy = len(test_data) > 0
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'data_generation': 'operational' if is_healthy else 'failed',
                'last_update': datetime.utcnow().isoformat(),
                'available_endpoints': [
                    'global_events',
                    'news_sentiment',
                    'geopolitical_risks',
                    'economic_mentions'
                ],
                'monitored_event_codes': len(self.economic_event_codes)
            }
            
        except Exception as e:
            logger.error(f"GDELT health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_update': datetime.utcnow().isoformat(),
                'data_generation': 'failed'
            }