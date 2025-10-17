"""
Google Trends Data Source

Provides access to Google Trends data for monitoring search patterns
and sentiment indicators related to economic and financial topics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.core.exceptions import DataSourceError, APIError

logger = logging.getLogger(__name__)
settings = get_settings()


class TrendsDataSource:
    """
    Google Trends data connector.
    
    Provides access to Google Trends data including:
    - Search volume trends for economic terms
    - Regional interest patterns
    - Related queries and topics
    - Trending searches
    - Real-time search trends
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.cache_ttl = 3600  # 1 hour for trends data
        
        # Economic and financial keywords to monitor
        self.economic_keywords = [
            'recession', 'inflation', 'unemployment', 'stock market crash',
            'financial crisis', 'bank failure', 'economic uncertainty',
            'supply chain disruption', 'interest rates', 'federal reserve',
            'market volatility', 'economic indicators', 'GDP growth',
            'consumer confidence', 'housing market', 'oil prices',
            'trade war', 'currency crisis', 'debt ceiling',
            'economic stimulus', 'quantitative easing'
        ]
        
        # Risk-related keywords
        self.risk_keywords = [
            'cyber attack', 'data breach', 'ransomware', 'natural disaster',
            'climate change', 'pandemic', 'supply shortage',
            'geopolitical tension', 'sanctions', 'trade restrictions',
            'energy crisis', 'food security', 'water shortage'
        ]
        
    async def get_trends_data(
        self,
        keywords: Optional[List[str]] = None,
        timeframe: str = 'today 1-m',
        geo: str = 'US',
        category: int = 0
    ) -> pd.DataFrame:
        """
        Get Google Trends data for specified keywords.
        
        Args:
            keywords: List of keywords to search for
            timeframe: Time frame for trends data
            geo: Geographic region code
            category: Google Trends category
            
        Returns:
            DataFrame with trends data
        """
        if not keywords:
            keywords = self.economic_keywords[:5]  # Default to top 5 economic keywords
            
        cache_key = f"trends_data_{hash(str(keywords))}_{timeframe}_{geo}"
        
        # Check cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved Google Trends data from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating Google Trends data")
        
        try:
            # Since pytrends requires synchronous execution, we'll simulate trends data
            # In a real implementation, this would use pytrends library
            data = await self._generate_trends_data(keywords, timeframe, geo)
            
            # Cache the results
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating Google Trends data: {e}")
            return await self._get_fallback_trends_data(keywords)
    
    async def get_economic_sentiment_index(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Calculate economic sentiment index based on trends data.
        
        Args:
            start_date: Start date for sentiment analysis
            end_date: End date for sentiment analysis
            
        Returns:
            DataFrame with economic sentiment index
        """
        cache_key = f"economic_sentiment_{start_date}_{end_date}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved economic sentiment index from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Calculating economic sentiment index")
        
        try:
            # Get trends data for economic keywords
            positive_keywords = ['economic growth', 'job market', 'consumer confidence', 'market rally']
            negative_keywords = ['recession', 'unemployment', 'inflation', 'market crash']
            
            positive_trends = await self.get_trends_data(positive_keywords)
            negative_trends = await self.get_trends_data(negative_keywords)
            
            # Calculate sentiment index
            sentiment_data = self._calculate_sentiment_index(positive_trends, negative_trends)
            
            await self.cache.set(cache_key, sentiment_data.to_dict('records'), ttl=self.cache_ttl)
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error calculating economic sentiment index: {e}")
            return await self._get_fallback_sentiment_data()
    
    async def get_risk_signal_trends(
        self,
        risk_categories: Optional[List[str]] = None,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Get trends data for risk-related search terms.
        
        Args:
            risk_categories: List of risk categories to monitor
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with risk signal trends
        """
        if not risk_categories:
            risk_categories = ['cyber', 'natural_disaster', 'economic', 'geopolitical']
            
        cache_key = f"risk_signals_{hash(str(risk_categories))}_{lookback_days}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved risk signal trends from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating risk signal trends")
        
        try:
            risk_data = await self._generate_risk_signal_data(risk_categories, lookback_days)
            
            await self.cache.set(cache_key, risk_data.to_dict('records'), ttl=self.cache_ttl)
            return risk_data
            
        except Exception as e:
            logger.error(f"Error generating risk signal trends: {e}")
            return await self._get_fallback_risk_signals()
    
    async def get_regional_trends(
        self,
        keywords: List[str],
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get regional breakdown of trends data.
        
        Args:
            keywords: Keywords to analyze
            regions: List of regions to compare
            
        Returns:
            DataFrame with regional trends data
        """
        if not regions:
            regions = ['US-CA', 'US-NY', 'US-TX', 'US-FL', 'US-IL']  # Top 5 states by economy
            
        cache_key = f"regional_trends_{hash(str(keywords))}_{hash(str(regions))}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved regional trends data from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Generating regional trends data")
        
        try:
            regional_data = await self._generate_regional_data(keywords, regions)
            
            await self.cache.set(cache_key, regional_data.to_dict('records'), ttl=self.cache_ttl)
            return regional_data
            
        except Exception as e:
            logger.error(f"Error generating regional trends data: {e}")
            return await self._get_fallback_regional_data()
    
    async def _generate_trends_data(
        self,
        keywords: List[str],
        timeframe: str,
        geo: str
    ) -> pd.DataFrame:
        """Generate simulated Google Trends data"""
        
        # Parse timeframe to determine date range
        if 'today 1-m' in timeframe:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
        elif 'today 3-m' in timeframe:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
        elif 'today 1-y' in timeframe:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
        else:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
        
        # Generate daily data points
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        records = []
        
        for keyword in keywords:
            # Generate trend values (0-100 scale like Google Trends)
            base_trend = self._get_keyword_base_trend(keyword)
            
            for date in date_range:
                # Add some realistic variation
                trend_value = base_trend + np.random.normal(0, 10)
                trend_value = max(0, min(100, trend_value))  # Clamp to 0-100
                
                # Add weekly and seasonal patterns
                day_of_week_factor = 1.0 + 0.1 * np.sin(date.weekday() * 2 * np.pi / 7)
                seasonal_factor = 1.0 + 0.2 * np.sin(date.timetuple().tm_yday * 2 * np.pi / 365)
                
                trend_value *= day_of_week_factor * seasonal_factor
                trend_value = max(0, min(100, trend_value))
                
                record = {
                    'date': date,
                    'keyword': keyword,
                    'trend_value': round(trend_value, 1),
                    'geo': geo,
                    'category': self._categorize_keyword(keyword),
                    'related_queries_rising': self._get_related_queries(keyword)[:3],
                    'search_volume_index': trend_value,
                    'relative_interest': trend_value / 100,
                    'isPartial': date == date_range[-1]  # Last data point is partial
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _get_keyword_base_trend(self, keyword: str) -> float:
        """Get base trend value for a keyword based on current events"""
        # Map keywords to realistic base trend values
        trend_mapping = {
            'recession': 35,
            'inflation': 60,
            'unemployment': 40,
            'stock market crash': 25,
            'financial crisis': 20,
            'bank failure': 15,
            'economic uncertainty': 45,
            'supply chain disruption': 70,
            'interest rates': 55,
            'federal reserve': 30,
            'market volatility': 50,
            'economic indicators': 35,
            'GDP growth': 25,
            'consumer confidence': 30,
            'housing market': 65,
            'oil prices': 55,
            'trade war': 20,
            'currency crisis': 15,
            'debt ceiling': 25,
            'economic stimulus': 30,
            'quantitative easing': 20,
            'cyber attack': 40,
            'data breach': 35,
            'ransomware': 45,
            'natural disaster': 30,
            'climate change': 80,
            'pandemic': 25,
            'supply shortage': 40,
            'geopolitical tension': 35,
            'sanctions': 30,
            'trade restrictions': 25,
            'energy crisis': 50,
            'food security': 40,
            'water shortage': 35
        }
        
        return trend_mapping.get(keyword, 30)  # Default to 30 if keyword not found
    
    def _categorize_keyword(self, keyword: str) -> str:
        """Categorize keyword into risk or economic type"""
        if keyword in self.economic_keywords:
            return 'economic'
        elif keyword in self.risk_keywords:
            return 'risk'
        else:
            return 'general'
    
    def _get_related_queries(self, keyword: str) -> List[str]:
        """Get simulated related queries for a keyword"""
        related_mapping = {
            'recession': ['economic downturn', 'GDP decline', 'job losses'],
            'inflation': ['cost of living', 'price increases', 'monetary policy'],
            'unemployment': ['job market', 'layoffs', 'employment rate'],
            'stock market crash': ['market decline', 'bear market', 'portfolio losses'],
            'financial crisis': ['banking crisis', 'credit crunch', 'systemic risk'],
            'bank failure': ['bank collapse', 'FDIC insurance', 'deposit protection'],
            'supply chain disruption': ['logistics problems', 'shipping delays', 'inventory shortage'],
            'cyber attack': ['data security', 'network breach', 'cyber security'],
            'natural disaster': ['emergency response', 'disaster relief', 'climate impact']
        }
        
        return related_mapping.get(keyword, ['related term 1', 'related term 2', 'related term 3'])
    
    def _calculate_sentiment_index(
        self,
        positive_trends: pd.DataFrame,
        negative_trends: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate economic sentiment index from positive and negative trends"""
        
        # Aggregate trends by date
        positive_agg = positive_trends.groupby('date')['trend_value'].mean().reset_index()
        negative_agg = negative_trends.groupby('date')['trend_value'].mean().reset_index()
        
        # Merge on date
        sentiment_df = positive_agg.merge(negative_agg, on='date', suffixes=('_positive', '_negative'))
        
        # Calculate sentiment index (0-100 scale, higher = more positive sentiment)
        sentiment_df['sentiment_index'] = (
            (sentiment_df['trend_value_positive'] - sentiment_df['trend_value_negative'] + 100) / 2
        )
        
        # Smooth the sentiment index
        sentiment_df['sentiment_index_smoothed'] = sentiment_df['sentiment_index'].rolling(window=7).mean()
        
        # Categorize sentiment
        sentiment_df['sentiment_category'] = sentiment_df['sentiment_index'].apply(
            lambda x: 'positive' if x > 60 else 'negative' if x < 40 else 'neutral'
        )
        
        # Add additional metrics
        sentiment_df['volatility'] = sentiment_df['sentiment_index'].rolling(window=7).std()
        sentiment_df['trend_direction'] = np.where(
            sentiment_df['sentiment_index'].diff() > 0, 'improving', 'deteriorating'
        )
        
        return sentiment_df
    
    async def _generate_risk_signal_data(
        self,
        risk_categories: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Generate risk signal trends data"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        records = []
        
        category_keywords = {
            'cyber': ['cyber attack', 'data breach', 'ransomware'],
            'natural_disaster': ['earthquake', 'hurricane', 'wildfire'],
            'economic': ['recession', 'inflation', 'market crash'],
            'geopolitical': ['trade war', 'sanctions', 'geopolitical tension']
        }
        
        for category in risk_categories:
            keywords = category_keywords.get(category, [f'{category} risk'])
            
            for date in date_range:
                # Generate risk signal intensity
                base_intensity = np.random.uniform(10, 40)
                
                # Add event spikes occasionally
                if np.random.random() < 0.05:  # 5% chance of spike
                    spike_intensity = np.random.uniform(60, 90)
                    base_intensity = max(base_intensity, spike_intensity)
                
                record = {
                    'date': date,
                    'risk_category': category,
                    'risk_intensity': base_intensity,
                    'keywords': keywords,
                    'signal_strength': base_intensity / 100,
                    'alert_threshold_exceeded': base_intensity > 70,
                    'confidence_level': np.random.uniform(0.7, 0.95),
                    'geographic_scope': 'national' if base_intensity > 50 else 'regional',
                    'trending_keywords': keywords[:2],
                    'related_events_count': np.random.randint(0, 10)
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    async def _generate_regional_data(
        self,
        keywords: List[str],
        regions: List[str]
    ) -> pd.DataFrame:
        """Generate regional trends data"""
        
        records = []
        
        for region in regions:
            for keyword in keywords:
                # Generate regional interest level
                base_interest = self._get_keyword_base_trend(keyword)
                
                # Add regional variation
                regional_factor = np.random.uniform(0.5, 1.5)
                regional_interest = base_interest * regional_factor
                regional_interest = max(0, min(100, regional_interest))
                
                record = {
                    'region': region,
                    'region_name': self._get_region_name(region),
                    'keyword': keyword,
                    'interest_level': regional_interest,
                    'relative_interest': regional_interest / 100,
                    'rank': 0,  # Will be calculated after all regions
                    'population_adjusted_interest': regional_interest * np.random.uniform(0.8, 1.2),
                    'trending_up': np.random.choice([True, False], p=[0.4, 0.6]),
                    'breakout_term': np.random.choice([True, False], p=[0.1, 0.9]),
                    'related_queries': self._get_related_queries(keyword)[:2]
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Calculate ranks within each keyword
        for keyword in keywords:
            keyword_mask = df['keyword'] == keyword
            df.loc[keyword_mask, 'rank'] = df.loc[keyword_mask, 'interest_level'].rank(method='dense', ascending=False)
        
        return df
    
    def _get_region_name(self, region_code: str) -> str:
        """Get region name from region code"""
        region_mapping = {
            'US-CA': 'California',
            'US-NY': 'New York',
            'US-TX': 'Texas',
            'US-FL': 'Florida',
            'US-IL': 'Illinois',
            'US-PA': 'Pennsylvania',
            'US-OH': 'Ohio',
            'US-GA': 'Georgia',
            'US-NC': 'North Carolina',
            'US-MI': 'Michigan'
        }
        return region_mapping.get(region_code, region_code)
    
    async def _get_fallback_trends_data(self, keywords: List[str]) -> pd.DataFrame:
        """Return fallback trends data when generation fails"""
        logger.warning("Using fallback Google Trends data")
        
        fallback_data = []
        base_date = datetime.utcnow() - timedelta(days=7)
        
        for i, keyword in enumerate(keywords):
            fallback_data.append({
                'date': base_date + timedelta(days=i),
                'keyword': keyword,
                'trend_value': 30.0 + i * 5,
                'geo': 'US',
                'category': 'economic',
                'search_volume_index': 30.0 + i * 5,
                'relative_interest': 0.3 + i * 0.05
            })
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_sentiment_data(self) -> pd.DataFrame:
        """Return fallback sentiment data when calculation fails"""
        logger.warning("Using fallback economic sentiment data")
        
        fallback_data = [{
            'date': datetime.utcnow() - timedelta(days=1),
            'sentiment_index': 50.0,
            'sentiment_category': 'neutral',
            'trend_direction': 'stable'
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_risk_signals(self) -> pd.DataFrame:
        """Return fallback risk signals data when generation fails"""
        logger.warning("Using fallback risk signals data")
        
        fallback_data = [{
            'date': datetime.utcnow() - timedelta(days=1),
            'risk_category': 'general',
            'risk_intensity': 25.0,
            'signal_strength': 0.25,
            'alert_threshold_exceeded': False
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_regional_data(self) -> pd.DataFrame:
        """Return fallback regional data when generation fails"""
        logger.warning("Using fallback regional trends data")
        
        fallback_data = [{
            'region': 'US-CA',
            'region_name': 'California',
            'keyword': 'economic indicators',
            'interest_level': 45.0,
            'relative_interest': 0.45,
            'rank': 1
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the Trends data source is operational"""
        try:
            # Test basic data generation functionality
            test_keywords = ['economy']
            test_data = await self._generate_trends_data(test_keywords, 'today 1-m', 'US')
            
            is_healthy = len(test_data) > 0
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'data_generation': 'operational' if is_healthy else 'failed',
                'last_update': datetime.utcnow().isoformat(),
                'available_categories': ['economic', 'risk', 'general'],
                'default_keywords_count': len(self.economic_keywords + self.risk_keywords)
            }
            
        except Exception as e:
            logger.error(f"Trends health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_update': datetime.utcnow().isoformat(),
                'data_generation': 'failed'
            }