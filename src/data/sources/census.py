"""
U.S. Census Bureau Trade Data Source

Provides access to international trade data from the U.S. Census Bureau,
including imports, exports, and trade balance data by country and commodity.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from etl.utils.connectors import APIConnector
from src.cache.cache_manager import CacheManager
from src.core.config import get_settings


class CensusTradeDataFetcher:
    """
    Fetches international trade data from U.S. Census Bureau
    
    Key datasets:
    - Monthly trade data by country
    - Trade data by commodity (HS codes)
    - Trade balance calculations
    - Port-level trade statistics
    """
    
    def __init__(self):
        self.logger = logging.getLogger("census_trade_fetcher")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # Census API configuration
        self.api_key = getattr(self.settings, 'CENSUS_API_KEY', None)
        self.base_url = "https://api.census.gov/data"
        
        # Initialize connector
        connector_config = {
            'base_url': self.base_url,
            'headers': {
                'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'
            },
            'timeout': 30,
            'rate_limit': 500,  # Census allows up to 500 queries per day per IP
            'cache_ttl': 7200   # 2 hour cache for trade data
        }
        
        self.connector = APIConnector("census", connector_config)
        
        # Key trading partners for risk assessment
        self.key_partners = {
            '5700': 'China',
            '1220': 'Canada', 
            '2010': 'Mexico',
            '4280': 'Germany',
            '5880': 'Japan',
            '5800': 'South Korea',
            '4130': 'United Kingdom',
            '4810': 'Taiwan',
            '5330': 'India',
            '4100': 'Italy',
            '4070': 'France',
            '2140': 'Brazil',
            '5350': 'Vietnam',
            '3750': 'Netherlands',
            '4031': 'Belgium'
        }
        
        # Critical commodity sectors (HS codes)
        self.critical_commodities = {
            '84': 'Nuclear reactors, boilers, machinery',
            '85': 'Electrical machinery and equipment', 
            '87': 'Vehicles, parts and accessories',
            '90': 'Optical, photographic, measuring instruments',
            '39': 'Plastics and articles thereof',
            '72': 'Iron and steel',
            '76': 'Aluminum and articles thereof',
            '29': 'Organic chemicals',
            '30': 'Pharmaceutical products',
            '88': 'Aircraft, spacecraft, and parts thereof'
        }
    
    async def fetch_latest_trade_data(self, years: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest trade data by country and commodity
        
        Args:
            years: List of years to fetch (default: current year and previous year)
            
        Returns:
            Dictionary with trade data by dataset type
        """
        try:
            await self.connector.connect()
            
            if years is None:
                current_year = datetime.now().year
                years = [current_year - 1, current_year]  # Previous year and current year
            
            all_data = {}
            
            # Fetch trade by country data
            country_data = await self._fetch_trade_by_country(years)
            if country_data is not None:
                all_data['by_country'] = country_data
            
            # Fetch trade by commodity data
            commodity_data = await self._fetch_trade_by_commodity(years)
            if commodity_data is not None:
                all_data['by_commodity'] = commodity_data
            
            # Calculate trade balances and risk metrics
            if 'by_country' in all_data:
                balance_data = self._calculate_trade_balances(all_data['by_country'])
                all_data['trade_balances'] = balance_data
            
            self.logger.info(f"Successfully fetched Census trade data for {len(years)} years")
            
            # Cache the data
            await self._cache_data(all_data)
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Census trade data: {str(e)}")
            
            # Try to return cached data
            cached_data = await self._get_cached_data()
            if cached_data:
                self.logger.warning("Returning cached Census trade data due to fetch error")
                return cached_data
            
            raise
    
    async def _fetch_trade_by_country(self, years: List[int]) -> Optional[pd.DataFrame]:
        """Fetch trade data by country"""
        try:
            all_records = []
            
            for year in years:
                # Monthly trade data endpoint
                endpoint = f"timeseries/intltrade/exports/enduse"
                
                params = {
                    'get': 'ENDUSE,ENDUSE_LDESC,CTY_CODE,CTY_NAME,time,GEN_VAL_MO,GEN_VAL_YR',
                    'time': f"{year}",
                    'ENDUSE': '*',  # All end use categories
                    'CTY_CODE': ','.join(self.key_partners.keys())
                }
                
                if self.api_key:
                    params['key'] = self.api_key
                
                try:
                    response_data = await self.connector.fetch_data(endpoint, params)
                    
                    if isinstance(response_data, list) and len(response_data) > 1:
                        # First row is headers, rest is data
                        headers = response_data[0]
                        data_rows = response_data[1:]
                        
                        for row in data_rows:
                            if len(row) == len(headers):
                                record = dict(zip(headers, row))
                                record['year'] = year
                                record['data_type'] = 'exports'
                                all_records.append(record)
                
                except Exception as e:
                    self.logger.warning(f"Error fetching export data for {year}: {str(e)}")
                
                # Fetch imports data
                try:
                    endpoint = f"timeseries/intltrade/imports/enduse"
                    response_data = await self.connector.fetch_data(endpoint, params)
                    
                    if isinstance(response_data, list) and len(response_data) > 1:
                        headers = response_data[0]
                        data_rows = response_data[1:]
                        
                        for row in data_rows:
                            if len(row) == len(headers):
                                record = dict(zip(headers, row))
                                record['year'] = year
                                record['data_type'] = 'imports'
                                all_records.append(record)
                
                except Exception as e:
                    self.logger.warning(f"Error fetching import data for {year}: {str(e)}")
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            if not all_records:
                self.logger.warning("No trade by country data found")
                return None
            
            # Convert to DataFrame and clean
            df = pd.DataFrame(all_records)
            df = self._clean_trade_data(df)
            
            self.logger.debug(f"Fetched {len(df)} trade by country records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching trade by country data: {str(e)}")
            return None
    
    async def _fetch_trade_by_commodity(self, years: List[int]) -> Optional[pd.DataFrame]:
        """Fetch trade data by commodity"""
        try:
            all_records = []
            
            for year in years:
                # Fetch data for critical commodities
                for hs_code, commodity_name in self.critical_commodities.items():
                    
                    # Exports by commodity
                    try:
                        endpoint = f"timeseries/intltrade/exports/hs"
                        params = {
                            'get': 'COMM_LVL,COMMODITY,COMMODITY_LDESC,CTY_CODE,CTY_NAME,time,GEN_VAL_MO,GEN_VAL_YR',
                            'time': f"{year}",
                            'COMMODITY': hs_code,
                            'CTY_CODE': ','.join(list(self.key_partners.keys())[:5])  # Limit to top 5 partners
                        }
                        
                        if self.api_key:
                            params['key'] = self.api_key
                        
                        response_data = await self.connector.fetch_data(endpoint, params)
                        
                        if isinstance(response_data, list) and len(response_data) > 1:
                            headers = response_data[0]
                            data_rows = response_data[1:]
                            
                            for row in data_rows:
                                if len(row) == len(headers):
                                    record = dict(zip(headers, row))
                                    record['year'] = year
                                    record['data_type'] = 'exports'
                                    record['hs_sector'] = hs_code
                                    record['sector_name'] = commodity_name
                                    all_records.append(record)
                    
                    except Exception as e:
                        self.logger.warning(f"Error fetching export commodity data for {hs_code}/{year}: {str(e)}")
                    
                    # Imports by commodity
                    try:
                        endpoint = f"timeseries/intltrade/imports/hs"
                        response_data = await self.connector.fetch_data(endpoint, params)
                        
                        if isinstance(response_data, list) and len(response_data) > 1:
                            headers = response_data[0]
                            data_rows = response_data[1:]
                            
                            for row in data_rows:
                                if len(row) == len(headers):
                                    record = dict(zip(headers, row))
                                    record['year'] = year
                                    record['data_type'] = 'imports'
                                    record['hs_sector'] = hs_code
                                    record['sector_name'] = commodity_name
                                    all_records.append(record)
                    
                    except Exception as e:
                        self.logger.warning(f"Error fetching import commodity data for {hs_code}/{year}: {str(e)}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
            
            if not all_records:
                self.logger.warning("No trade by commodity data found")
                return None
            
            # Convert to DataFrame and clean
            df = pd.DataFrame(all_records)
            df = self._clean_trade_data(df)
            
            self.logger.debug(f"Fetched {len(df)} trade by commodity records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching trade by commodity data: {str(e)}")
            return None
    
    def _clean_trade_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize trade data"""
        try:
            # Convert time to date
            if 'time' in df.columns:
                df['date'] = pd.to_datetime(df['time'], format='%Y-%m', errors='coerce')
            
            # Convert trade values to numeric
            value_columns = ['GEN_VAL_MO', 'GEN_VAL_YR']
            for col in value_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add country names
            if 'CTY_CODE' in df.columns:
                df['country_name'] = df['CTY_CODE'].map(self.key_partners)
                df['country_name'] = df['country_name'].fillna(df.get('CTY_NAME', 'Unknown'))
            
            # Remove rows with null values in critical columns
            critical_cols = ['date', 'CTY_CODE', 'data_type']
            for col in critical_cols:
                if col in df.columns:
                    df = df.dropna(subset=[col])
            
            # Sort by date and country
            if 'date' in df.columns and 'CTY_CODE' in df.columns:
                df = df.sort_values(['date', 'CTY_CODE'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning trade data: {str(e)}")
            return df
    
    def _calculate_trade_balances(self, trade_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade balances and risk metrics"""
        try:
            if trade_df.empty:
                return pd.DataFrame()
            
            # Pivot to get exports and imports by country and date
            balance_records = []
            
            # Group by date and country
            for (date, country_code), group in trade_df.groupby(['date', 'CTY_CODE']):
                exports = group[group['data_type'] == 'exports']['GEN_VAL_MO'].sum()
                imports = group[group['data_type'] == 'imports']['GEN_VAL_MO'].sum()
                
                trade_balance = exports - imports
                total_trade = exports + imports
                
                # Calculate risk metrics
                import_dependency = imports / total_trade if total_trade > 0 else 0
                trade_intensity = total_trade
                
                balance_records.append({
                    'date': date,
                    'country_code': country_code,
                    'country_name': self.key_partners.get(country_code, 'Unknown'),
                    'exports': exports,
                    'imports': imports,
                    'trade_balance': trade_balance,
                    'total_trade': total_trade,
                    'import_dependency': import_dependency,
                    'trade_intensity': trade_intensity,
                    'risk_level': self._assess_trade_risk(country_code, trade_balance, import_dependency, trade_intensity)
                })
            
            balance_df = pd.DataFrame(balance_records)
            
            # Calculate month-over-month changes
            if len(balance_df) > 0:
                balance_df = balance_df.sort_values(['country_code', 'date'])
                balance_df['trade_balance_change'] = balance_df.groupby('country_code')['trade_balance'].pct_change()
                balance_df['import_change'] = balance_df.groupby('country_code')['imports'].pct_change()
            
            return balance_df
            
        except Exception as e:
            self.logger.error(f"Error calculating trade balances: {str(e)}")
            return pd.DataFrame()
    
    def _assess_trade_risk(self, country_code: str, trade_balance: float, 
                          import_dependency: float, trade_intensity: float) -> str:
        """Assess trade-related risk levels"""
        try:
            # Risk factors
            risk_score = 0
            
            # High import dependency increases risk
            if import_dependency > 0.7:
                risk_score += 2
            elif import_dependency > 0.5:
                risk_score += 1
            
            # Large trade deficits increase risk
            if trade_balance < -50000:  # 50B deficit
                risk_score += 2
            elif trade_balance < -10000:  # 10B deficit
                risk_score += 1
            
            # High trade intensity with certain countries increases risk
            high_risk_countries = ['5700']  # China
            if country_code in high_risk_countries:
                if trade_intensity > 100000:  # 100B+ trade
                    risk_score += 2
                elif trade_intensity > 50000:  # 50B+ trade
                    risk_score += 1
            
            # Convert score to risk level
            if risk_score >= 4:
                return "high"
            elif risk_score >= 2:
                return "moderate"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    async def get_trade_summary(self) -> Dict[str, Any]:
        """Get trade summary with risk indicators"""
        try:
            trade_data = await self.fetch_latest_trade_data()
            
            if not trade_data:
                return {"error": "No trade data available"}
            
            summary = {
                "trade_partners": {},
                "commodity_analysis": {},
                "risk_assessment": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Trade partner analysis
            if 'trade_balances' in trade_data:
                balance_df = trade_data['trade_balances']
                
                if not balance_df.empty:
                    # Latest month data
                    latest_month = balance_df['date'].max()
                    latest_data = balance_df[balance_df['date'] == latest_month]
                    
                    for _, row in latest_data.iterrows():
                        country_name = row['country_name']
                        summary["trade_partners"][country_name] = {
                            "exports": row['exports'],
                            "imports": row['imports'],
                            "trade_balance": row['trade_balance'],
                            "import_dependency": row['import_dependency'],
                            "risk_level": row['risk_level']
                        }
                    
                    # Overall risk assessment
                    high_risk_partners = latest_data[latest_data['risk_level'] == 'high']
                    total_high_risk_trade = high_risk_partners['total_trade'].sum()
                    total_trade = latest_data['total_trade'].sum()
                    
                    summary["risk_assessment"] = {
                        "high_risk_partner_count": len(high_risk_partners),
                        "high_risk_trade_share": total_high_risk_trade / total_trade if total_trade > 0 else 0,
                        "total_trade_deficit": latest_data['trade_balance'].sum(),
                        "avg_import_dependency": latest_data['import_dependency'].mean()
                    }
            
            # Commodity analysis
            if 'by_commodity' in trade_data:
                commodity_df = trade_data['by_commodity']
                
                if not commodity_df.empty:
                    latest_month = commodity_df['date'].max()
                    latest_commodities = commodity_df[commodity_df['date'] == latest_month]
                    
                    commodity_summary = latest_commodities.groupby(['hs_sector', 'sector_name']).agg({
                        'GEN_VAL_MO': 'sum'
                    }).reset_index()
                    
                    for _, row in commodity_summary.iterrows():
                        summary["commodity_analysis"][row['sector_name']] = {
                            "hs_code": row['hs_sector'],
                            "monthly_value": row['GEN_VAL_MO']
                        }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating trade summary: {str(e)}")
            return {"error": str(e)}
    
    async def _cache_data(self, data: Dict[str, pd.DataFrame]):
        """Cache trade data"""
        try:
            cache_key = "census_trade_data"
            
            # Convert DataFrames to serializable format
            serializable_data = {}
            for dataset_name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    serializable_data[dataset_name] = {
                        'data': df.to_dict('records'),
                        'last_updated': datetime.now().isoformat()
                    }
                else:
                    serializable_data[dataset_name] = data
            
            await self.cache.set(cache_key, serializable_data, ttl=7200)
            self.logger.debug("Census trade data cached successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache Census trade data: {str(e)}")
    
    async def _get_cached_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached trade data"""
        try:
            cache_key = "census_trade_data"
            cached_data = await self.cache.get(cache_key)
            
            if not cached_data:
                return None
            
            # Convert back to DataFrames
            result_data = {}
            for dataset_name, dataset_info in cached_data.items():
                if isinstance(dataset_info, dict) and 'data' in dataset_info:
                    df_data = dataset_info['data']
                    df = pd.DataFrame(df_data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    result_data[dataset_name] = df
                else:
                    result_data[dataset_name] = dataset_info
            
            return result_data if result_data else None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached Census trade data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Census API health"""
        try:
            # Test with a simple request
            current_year = datetime.now().year
            test_data = await self._fetch_trade_by_country([current_year])
            
            return {
                "status": "healthy",
                "api_key_configured": bool(self.api_key),
                "test_data_available": test_data is not None and len(test_data) > 0,
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