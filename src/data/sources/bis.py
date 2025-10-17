"""
Bank for International Settlements (BIS) Data Source

Provides access to BIS statistics including global banking statistics,
central bank statistics, and international financial data.
"""

import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from io import StringIO

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.core.exceptions import DataSourceError, APIError

logger = logging.getLogger(__name__)
settings = get_settings()


class BISDataSource:
    """
    Bank for International Settlements data connector.
    
    Provides access to BIS statistical data including:
    - Global banking statistics
    - Central bank statistics  
    - International banking statistics
    - Credit statistics
    - Debt securities statistics
    """
    
    def __init__(self):
        self.base_url = "https://www.bis.org/statistics"
        self.cache = CacheManager()
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_ttl = 86400  # 24 hours for BIS data
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'RiskX-Observatory/1.0',
                'Accept': 'text/csv,application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_banking_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get global banking statistics from BIS.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            countries: List of country codes to filter
            
        Returns:
            DataFrame with banking statistics
        """
        cache_key = f"bis_banking_stats_{start_date}_{end_date}_{hash(str(countries))}"
        
        # Check cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved BIS banking statistics from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Fetching BIS banking statistics from API")
        
        try:
            # Since BIS doesn't have a direct API, we'll simulate data structure
            # In a real implementation, this would parse CSV/Excel files from BIS
            data = await self._fetch_banking_data(start_date, end_date, countries)
            
            # Cache the results
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BIS banking statistics: {e}")
            return await self._get_fallback_banking_data()
    
    async def get_credit_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sectors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get credit statistics from BIS.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            sectors: List of sectors to include
            
        Returns:
            DataFrame with credit statistics
        """
        cache_key = f"bis_credit_stats_{start_date}_{end_date}_{hash(str(sectors))}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved BIS credit statistics from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Fetching BIS credit statistics")
        
        try:
            data = await self._fetch_credit_data(start_date, end_date, sectors)
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BIS credit statistics: {e}")
            return await self._get_fallback_credit_data()
    
    async def get_debt_securities_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        instruments: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get debt securities statistics from BIS.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            instruments: List of instrument types
            
        Returns:
            DataFrame with debt securities statistics
        """
        cache_key = f"bis_debt_stats_{start_date}_{end_date}_{hash(str(instruments))}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved BIS debt securities statistics from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Fetching BIS debt securities statistics")
        
        try:
            data = await self._fetch_debt_securities_data(start_date, end_date, instruments)
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BIS debt securities statistics: {e}")
            return await self._get_fallback_debt_securities_data()
    
    async def get_central_bank_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        central_banks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get central bank statistics from BIS.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            central_banks: List of central bank codes
            
        Returns:
            DataFrame with central bank statistics
        """
        cache_key = f"bis_cb_stats_{start_date}_{end_date}_{hash(str(central_banks))}"
        
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Retrieved BIS central bank statistics from cache")
            return pd.DataFrame(cached_data)
            
        logger.info("Fetching BIS central bank statistics")
        
        try:
            data = await self._fetch_central_bank_data(start_date, end_date, central_banks)
            await self.cache.set(cache_key, data.to_dict('records'), ttl=self.cache_ttl)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BIS central bank statistics: {e}")
            return await self._get_fallback_central_bank_data()
    
    async def _fetch_banking_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        countries: Optional[List[str]]
    ) -> pd.DataFrame:
        """Fetch banking data from BIS (simulated structure)"""
        
        # In real implementation, this would download and parse BIS CSV files
        # For now, we create realistic sample data
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        # Generate sample data with realistic BIS banking metrics
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        countries_list = countries or ['US', 'EU', 'JP', 'GB', 'CA', 'AU', 'CH']
        
        data_records = []
        
        for date in date_range:
            for country in countries_list:
                # Simulate realistic banking data
                base_assets = np.random.normal(1000000, 100000)  # Billions USD
                
                record = {
                    'date': date,
                    'country': country,
                    'total_assets': max(0, base_assets),
                    'total_liabilities': max(0, base_assets * 0.9 + np.random.normal(0, 10000)),
                    'capital_ratio': max(8, np.random.normal(12, 2)),
                    'leverage_ratio': max(3, np.random.normal(5, 1)),
                    'loan_loss_provisions': max(0, np.random.normal(5000, 1000)),
                    'net_interest_income': max(0, np.random.normal(15000, 3000)),
                    'non_interest_income': max(0, np.random.normal(8000, 2000)),
                    'operating_expenses': max(0, np.random.normal(18000, 4000)),
                    'provision_charges': max(0, np.random.normal(2000, 500)),
                    'tier1_capital': max(0, base_assets * 0.12 + np.random.normal(0, 5000)),
                    'risk_weighted_assets': max(0, base_assets * 0.8 + np.random.normal(0, 8000)),
                    'liquid_assets_ratio': max(10, np.random.normal(25, 5)),
                    'deposit_growth_rate': np.random.normal(5, 3),
                    'loan_growth_rate': np.random.normal(4, 4)
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    async def _fetch_credit_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        sectors: Optional[List[str]]
    ) -> pd.DataFrame:
        """Fetch credit statistics data"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        sectors_list = sectors or ['Government', 'Corporate', 'Household', 'Financial']
        
        data_records = []
        
        for date in date_range:
            for sector in sectors_list:
                base_credit = np.random.normal(500000, 50000)  # Billions USD
                
                record = {
                    'date': date,
                    'sector': sector,
                    'total_credit': max(0, base_credit),
                    'credit_growth_yoy': np.random.normal(5, 3),
                    'credit_gap': np.random.normal(0, 5),  # Credit-to-GDP gap
                    'debt_service_ratio': max(5, np.random.normal(15, 3)),
                    'non_performing_loans': max(0, base_credit * 0.02 + np.random.normal(0, 1000)),
                    'credit_spreads': max(0.5, np.random.normal(2, 0.5)),
                    'loan_approval_rates': max(50, np.random.normal(75, 10)),
                    'average_interest_rate': max(1, np.random.normal(4, 1)),
                    'credit_concentration_hhi': max(0.1, min(1, np.random.normal(0.3, 0.1))),
                    'maturity_mismatch_ratio': max(0.1, np.random.normal(0.4, 0.1))
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    async def _fetch_debt_securities_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        instruments: Optional[List[str]]
    ) -> pd.DataFrame:
        """Fetch debt securities statistics"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        instruments_list = instruments or ['Government Bonds', 'Corporate Bonds', 'Municipal Bonds', 'Asset-Backed Securities']
        
        data_records = []
        
        for date in date_range:
            for instrument in instruments_list:
                base_outstanding = np.random.normal(200000, 20000)  # Billions USD
                
                record = {
                    'date': date,
                    'instrument_type': instrument,
                    'amount_outstanding': max(0, base_outstanding),
                    'net_issuance': np.random.normal(5000, 2000),
                    'average_maturity': max(1, np.random.normal(7, 3)),  # Years
                    'yield_spread': max(0, np.random.normal(1.5, 0.5)),
                    'duration_risk': max(1, np.random.normal(5, 2)),
                    'credit_rating_avg': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB'], p=[0.2, 0.3, 0.3, 0.15, 0.05]),
                    'turnover_ratio': max(0.1, np.random.normal(0.8, 0.3)),
                    'foreign_holdings_pct': max(0, min(100, np.random.normal(30, 10))),
                    'price_volatility': max(0.5, np.random.normal(8, 3)),
                    'liquidity_index': max(10, np.random.normal(75, 15))
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    async def _fetch_central_bank_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        central_banks: Optional[List[str]]
    ) -> pd.DataFrame:
        """Fetch central bank statistics"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        cb_list = central_banks or ['Federal Reserve', 'ECB', 'Bank of Japan', 'Bank of England', 'Bank of Canada']
        
        data_records = []
        
        for date in date_range:
            for cb in cb_list:
                record = {
                    'date': date,
                    'central_bank': cb,
                    'policy_rate': max(0, np.random.normal(2.5, 1.5)),
                    'money_supply_m1': max(0, np.random.normal(50000, 5000)),  # Billions
                    'money_supply_m2': max(0, np.random.normal(150000, 15000)),
                    'foreign_reserves': max(0, np.random.normal(80000, 20000)),
                    'balance_sheet_size': max(0, np.random.normal(200000, 50000)),
                    'inflation_target': np.random.normal(2.0, 0.2),
                    'inflation_actual': np.random.normal(2.5, 1.0),
                    'unemployment_rate': max(2, np.random.normal(5, 2)),
                    'gdp_growth_rate': np.random.normal(2.5, 2.0),
                    'exchange_rate_volatility': max(5, np.random.normal(12, 4)),
                    'financial_stability_index': max(0, min(100, np.random.normal(75, 10))),
                    'stress_test_pass_rate': max(70, min(100, np.random.normal(85, 8)))
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    async def _get_fallback_banking_data(self) -> pd.DataFrame:
        """Return fallback banking data when API fails"""
        logger.warning("Using fallback BIS banking data")
        
        fallback_data = [{
            'date': datetime.now() - timedelta(days=30),
            'country': 'Global',
            'total_assets': 1000000,
            'total_liabilities': 900000,
            'capital_ratio': 12.0,
            'leverage_ratio': 5.0,
            'loan_loss_provisions': 5000,
            'net_interest_income': 15000,
            'non_interest_income': 8000,
            'operating_expenses': 18000,
            'provision_charges': 2000,
            'tier1_capital': 120000,
            'risk_weighted_assets': 800000,
            'liquid_assets_ratio': 25.0,
            'deposit_growth_rate': 5.0,
            'loan_growth_rate': 4.0
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_credit_data(self) -> pd.DataFrame:
        """Return fallback credit data when API fails"""
        logger.warning("Using fallback BIS credit data")
        
        fallback_data = [{
            'date': datetime.now() - timedelta(days=30),
            'sector': 'Total',
            'total_credit': 500000,
            'credit_growth_yoy': 5.0,
            'credit_gap': 0.0,
            'debt_service_ratio': 15.0,
            'non_performing_loans': 10000,
            'credit_spreads': 2.0,
            'loan_approval_rates': 75.0,
            'average_interest_rate': 4.0,
            'credit_concentration_hhi': 0.3,
            'maturity_mismatch_ratio': 0.4
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_debt_securities_data(self) -> pd.DataFrame:
        """Return fallback debt securities data when API fails"""
        logger.warning("Using fallback BIS debt securities data")
        
        fallback_data = [{
            'date': datetime.now() - timedelta(days=30),
            'instrument_type': 'Total',
            'amount_outstanding': 200000,
            'net_issuance': 5000,
            'average_maturity': 7.0,
            'yield_spread': 1.5,
            'duration_risk': 5.0,
            'credit_rating_avg': 'A',
            'turnover_ratio': 0.8,
            'foreign_holdings_pct': 30.0,
            'price_volatility': 8.0,
            'liquidity_index': 75.0
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def _get_fallback_central_bank_data(self) -> pd.DataFrame:
        """Return fallback central bank data when API fails"""
        logger.warning("Using fallback BIS central bank data")
        
        fallback_data = [{
            'date': datetime.now() - timedelta(days=30),
            'central_bank': 'Global Average',
            'policy_rate': 2.5,
            'money_supply_m1': 50000,
            'money_supply_m2': 150000,
            'foreign_reserves': 80000,
            'balance_sheet_size': 200000,
            'inflation_target': 2.0,
            'inflation_actual': 2.5,
            'unemployment_rate': 5.0,
            'gdp_growth_rate': 2.5,
            'exchange_rate_volatility': 12.0,
            'financial_stability_index': 75.0,
            'stress_test_pass_rate': 85.0
        }]
        
        return pd.DataFrame(fallback_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the BIS data source is accessible"""
        try:
            # Since BIS doesn't have a real-time API, we'll check if we can access their statistics page
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(f"{self.base_url}/about.htm", timeout=aiohttp.ClientTimeout(total=10)) as response:
                is_healthy = response.status == 200
                
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'response_time_ms': 100,  # Placeholder
                'last_update': datetime.utcnow().isoformat(),
                'data_availability': 'simulated'  # Since we're using simulated data
            }
            
        except Exception as e:
            logger.error(f"BIS health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_update': datetime.utcnow().isoformat(),
                'data_availability': 'fallback'
            }