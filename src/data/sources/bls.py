"""
Bureau of Labor Statistics (BLS) Data Source

Provides access to employment, inflation, and labor market data
from the U.S. Bureau of Labor Statistics API.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from etl.utils.connectors import APIConnector
from src.cache.cache_manager import CacheManager
from src.core.config import get_settings


class BlsDataFetcher:
    """
    Fetches data from the Bureau of Labor Statistics API
    
    Key data series:
    - Employment data (PAYEMS, UNRATE, CIVPART)
    - Inflation data (CPIAUCSL, CPILFESL, CPILFENS) 
    - Labor productivity (OPHNFB, PRS85006112)
    - Producer prices (PPIACO, PPIFGS, PPIFIS)
    """
    
    def __init__(self):
        self.logger = logging.getLogger("bls_fetcher")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # BLS API configuration
        self.api_key = getattr(self.settings, 'BLS_API_KEY', None)
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        # Initialize connector
        connector_config = {
            'base_url': self.base_url,
            'headers': {
                'Content-Type': 'application/json',
                'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'
            },
            'timeout': 30,
            'rate_limit': 25,  # BLS allows 25 queries per day for unregistered users, 500 for registered
            'cache_ttl': 3600  # 1 hour cache
        }
        
        self.connector = APIConnector("bls", connector_config)
        
        # Key economic indicators
        self.key_series = {
            # Employment indicators
            'PAYEMS': 'All Employees, Total Nonfarm',
            'UNRATE': 'Unemployment Rate', 
            'CIVPART': 'Labor Force Participation Rate',
            'EMRATIO': 'Employment-Population Ratio',
            'LNS13023569': 'Employment Level - 25-54 years',
            'LNS14023569': 'Unemployment Rate - 25-54 years',
            
            # Inflation indicators
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
            'CPILFESL': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy',
            'CPILFENS': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy, NSA',
            'CPIENGSL': 'Consumer Price Index for All Urban Consumers: Energy',
            'CPIUFDSL': 'Consumer Price Index for All Urban Consumers: Food',
            
            # Producer prices
            'PPIACO': 'Producer Price Index for All Commodities',
            'PPIFGS': 'Producer Price Index: Finished Goods',
            'PPIFIS': 'Producer Price Index: Finished Goods Less Food and Energy',
            'PPIENG': 'Producer Price Index: Fuels and Related Products and Power',
            
            # Labor productivity
            'OPHNFB': 'Nonfarm Business Sector: Output Per Hour of All Persons',
            'PRS85006112': 'Nonfarm Business Sector: Unit Labor Costs',
            'PRS85006091': 'Nonfarm Business Sector: Real Hourly Compensation',
            
            # Wages and earnings
            'CES0500000003': 'Average Hourly Earnings of All Employees, Total Private',
            'CES0500000011': 'Average Weekly Hours of All Employees, Total Private',
            'AHETPI': 'Average Hourly Earnings of All Employees, Total Private',
            
            # Regional employment (key states)
            'LASST060000000000003': 'California Unemployment Rate',
            'LASST120000000000003': 'Florida Unemployment Rate', 
            'LASST360000000000003': 'New York Unemployment Rate',
            'LASST480000000000003': 'Texas Unemployment Rate'
        }
    
    async def fetch_latest_data(self, series_ids: Optional[List[str]] = None, 
                               start_year: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest data for specified series or all key series
        
        Args:
            series_ids: List of BLS series IDs to fetch
            start_year: Starting year for data (default: 5 years ago)
            
        Returns:
            Dictionary mapping series IDs to DataFrames
        """
        try:
            await self.connector.connect()
            
            if series_ids is None:
                series_ids = list(self.key_series.keys())
            
            if start_year is None:
                start_year = datetime.now().year - 5
            
            end_year = datetime.now().year
            
            all_data = {}
            
            # BLS API allows up to 25 series per request for registered users, 1 for unregistered
            batch_size = 25 if self.api_key else 1
            
            for i in range(0, len(series_ids), batch_size):
                batch_series = series_ids[i:i + batch_size]
                
                batch_data = await self._fetch_series_batch(batch_series, start_year, end_year)
                all_data.update(batch_data)
                
                # Rate limiting delay
                if i + batch_size < len(series_ids):
                    await asyncio.sleep(1)
            
            self.logger.info(f"Successfully fetched BLS data for {len(all_data)} series")
            
            # Cache the aggregated data
            await self._cache_data(all_data)
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error fetching BLS data: {str(e)}")
            
            # Try to return cached data
            cached_data = await self._get_cached_data(series_ids)
            if cached_data:
                self.logger.warning("Returning cached BLS data due to fetch error")
                return cached_data
            
            raise
    
    async def _fetch_series_batch(self, series_ids: List[str], start_year: int, end_year: int) -> Dict[str, pd.DataFrame]:
        """Fetch a batch of series data"""
        try:
            # Prepare request payload
            payload = {
                "seriesid": series_ids,
                "startyear": str(start_year),
                "endyear": str(end_year),
                "catalog": False,
                "calculations": True,
                "annualaverage": True
            }
            
            if self.api_key:
                payload["registrationkey"] = self.api_key
            
            # Make API request
            endpoint = ""  # BLS uses POST to base URL
            response_data = await self.connector.session.post(
                self.base_url.rstrip('/'),
                json=payload
            )
            
            if response_data.status != 200:
                raise Exception(f"BLS API returned status {response_data.status}")
            
            response_json = await response_data.json()
            
            if response_json.get('status') != 'REQUEST_SUCCEEDED':
                error_msg = response_json.get('message', ['Unknown error'])[0]
                raise Exception(f"BLS API error: {error_msg}")
            
            # Parse response data
            series_data = {}
            
            for series in response_json.get('Results', {}).get('series', []):
                series_id = series.get('seriesID')
                if not series_id:
                    continue
                
                data_points = []
                for data_point in series.get('data', []):
                    try:
                        # Parse date
                        year = int(data_point.get('year'))
                        period = data_point.get('period')
                        
                        if period.startswith('M'):
                            month = int(period[1:])
                            date = datetime(year, month, 1)
                        elif period == 'A01':  # Annual average
                            date = datetime(year, 12, 31)
                        else:
                            continue
                        
                        # Parse value
                        value_str = data_point.get('value', '').replace(',', '')
                        if value_str and value_str != '.':
                            value = float(value_str)
                        else:
                            value = None
                        
                        data_points.append({
                            'date': date,
                            'value': value,
                            'series_id': series_id,
                            'period': period,
                            'year': year,
                            'latest': data_point.get('latest', 'false') == 'true',
                            'footnotes': data_point.get('footnotes', [])
                        })
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error parsing data point for {series_id}: {str(e)}")
                        continue
                
                if data_points:
                    df = pd.DataFrame(data_points)
                    df = df.sort_values('date')
                    df['series_name'] = self.key_series.get(series_id, series_id)
                    
                    series_data[series_id] = df
                    
                    self.logger.debug(f"Parsed {len(df)} data points for series {series_id}")
            
            return series_data
            
        except Exception as e:
            self.logger.error(f"Error fetching BLS series batch: {str(e)}")
            raise
    
    async def fetch_series_by_category(self, category: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data by category
        
        Args:
            category: Category name (employment, inflation, productivity, etc.)
        """
        category_series = {
            'employment': ['PAYEMS', 'UNRATE', 'CIVPART', 'EMRATIO'],
            'inflation': ['CPIAUCSL', 'CPILFESL', 'CPIENGSL', 'CPIUFDSL'],
            'producer_prices': ['PPIACO', 'PPIFGS', 'PPIFIS', 'PPIENG'],
            'productivity': ['OPHNFB', 'PRS85006112', 'PRS85006091'],
            'wages': ['CES0500000003', 'CES0500000011', 'AHETPI'],
            'regional': ['LASST060000000000003', 'LASST120000000000003', 
                        'LASST360000000000003', 'LASST480000000000003']
        }
        
        series_ids = category_series.get(category.lower(), [])
        if not series_ids:
            raise ValueError(f"Unknown category: {category}")
        
        return await self.fetch_latest_data(series_ids)
    
    async def get_employment_indicators(self) -> Dict[str, Any]:
        """Get key employment indicators with risk assessment"""
        try:
            employment_data = await self.fetch_series_by_category('employment')
            
            if not employment_data:
                return {"error": "No employment data available"}
            
            indicators = {}
            
            # Process each series
            for series_id, df in employment_data.items():
                if df.empty:
                    continue
                
                latest_value = df.iloc[-1]['value']
                series_name = df.iloc[-1]['series_name']
                
                # Calculate month-over-month and year-over-year changes
                mom_change = None
                yoy_change = None
                
                if len(df) >= 2:
                    prev_value = df.iloc[-2]['value']
                    if prev_value and latest_value:
                        mom_change = ((latest_value - prev_value) / prev_value) * 100
                
                if len(df) >= 12:
                    year_ago_value = df.iloc[-13]['value'] if len(df) >= 13 else df.iloc[0]['value']
                    if year_ago_value and latest_value:
                        yoy_change = ((latest_value - year_ago_value) / year_ago_value) * 100
                
                indicators[series_id] = {
                    'name': series_name,
                    'latest_value': latest_value,
                    'latest_date': df.iloc[-1]['date'].isoformat(),
                    'mom_change': mom_change,
                    'yoy_change': yoy_change,
                    'risk_level': self._assess_employment_risk(series_id, latest_value, mom_change, yoy_change)
                }
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error getting employment indicators: {str(e)}")
            return {"error": str(e)}
    
    def _assess_employment_risk(self, series_id: str, value: float, mom_change: Optional[float], 
                               yoy_change: Optional[float]) -> str:
        """Assess risk level based on employment indicators"""
        if not value:
            return "unknown"
        
        # Risk thresholds based on historical data
        risk_thresholds = {
            'UNRATE': {  # Unemployment rate
                'low': 4.0,      # Below 4% = low risk
                'moderate': 6.0,  # 4-6% = moderate risk
                'high': 8.0      # Above 8% = high risk
            },
            'PAYEMS': {  # Employment growth (MoM %)
                'low_mom': 0.2,   # Above 0.2% MoM growth = low risk
                'high_mom': -0.2  # Below -0.2% MoM = high risk
            },
            'CIVPART': {  # Labor force participation
                'low': 62.0,      # Below 62% = high risk
                'moderate': 63.0  # 62-63% = moderate risk
            }
        }
        
        if series_id == 'UNRATE':
            thresholds = risk_thresholds['UNRATE']
            if value <= thresholds['low']:
                return "low"
            elif value <= thresholds['moderate']:
                return "moderate"
            else:
                return "high"
                
        elif series_id == 'PAYEMS' and mom_change is not None:
            thresholds = risk_thresholds['PAYEMS']
            if mom_change >= thresholds['low_mom']:
                return "low"
            elif mom_change <= thresholds['high_mom']:
                return "high"
            else:
                return "moderate"
                
        elif series_id == 'CIVPART':
            thresholds = risk_thresholds['CIVPART']
            if value >= thresholds['moderate']:
                return "low"
            elif value >= thresholds['low']:
                return "moderate"
            else:
                return "high"
        
        return "moderate"  # Default for unknown series
    
    async def _cache_data(self, data: Dict[str, pd.DataFrame]):
        """Cache fetched data"""
        try:
            cache_key = "bls_latest_data"
            
            # Convert DataFrames to serializable format
            serializable_data = {}
            for series_id, df in data.items():
                serializable_data[series_id] = {
                    'data': df.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
            
            await self.cache.set(cache_key, serializable_data, ttl=3600)
            self.logger.debug("BLS data cached successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache BLS data: {str(e)}")
    
    async def _get_cached_data(self, series_ids: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached data"""
        try:
            cache_key = "bls_latest_data"
            cached_data = await self.cache.get(cache_key)
            
            if not cached_data:
                return None
            
            # Convert back to DataFrames
            result_data = {}
            for series_id in series_ids:
                if series_id in cached_data:
                    df_data = cached_data[series_id]['data']
                    df = pd.DataFrame(df_data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    result_data[series_id] = df
            
            return result_data if result_data else None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached BLS data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check BLS API health"""
        try:
            # Test with a simple request
            test_data = await self.fetch_latest_data(['UNRATE'], start_year=datetime.now().year)
            
            return {
                "status": "healthy",
                "api_key_configured": bool(self.api_key),
                "test_series_count": len(test_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_key_configured": bool(self.api_key),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close connection"""
        if self.connector:
            await self.connector.close()