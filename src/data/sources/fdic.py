"""
FDIC Banking Data Source

Provides access to FDIC banking data including institution profiles,
financial ratios, and regulatory compliance metrics.
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


class FdicDataFetcher:
    """
    Fetches banking data from FDIC APIs
    
    Key datasets:
    - Institution Directory (banks and their characteristics)
    - Summary of Deposits (branch-level deposit data)
    - Bank Call Reports (financial condition data)
    - Failed Bank List
    """
    
    def __init__(self):
        self.logger = logging.getLogger("fdic_fetcher")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # FDIC API configuration (no API key required)
        self.base_url = "https://banks.data.fdic.gov/api"
        
        # Initialize connector
        connector_config = {
            'base_url': self.base_url,
            'headers': {
                'Accept': 'application/json',
                'User-Agent': 'RiskX/1.0 (contact@riskx.ai)'
            },
            'timeout': 30,
            'rate_limit': 1000,  # FDIC allows high rate limits
            'cache_ttl': 3600    # 1 hour cache
        }
        
        self.connector = APIConnector("fdic", connector_config)
        
        # Key financial metrics for risk assessment
        self.risk_metrics = {
            'capital_ratios': [
                'TIER1RISKCAT',  # Tier 1 Risk-Based Capital Ratio
                'TOTCAPRISKCAT', # Total Risk-Based Capital Ratio
                'LEVERAGE',      # Leverage Ratio
                'RWATA'          # Risk Weighted Assets to Total Assets
            ],
            'asset_quality': [
                'NPTL',          # Noncurrent Loans and Leases to Total Loans
                'LNLSALLW',      # Net Loan and Lease Losses to Total Loans
                'OTHERREPOPCT',  # Other Real Estate Owned to Total Assets
                'ALLLOTL'        # Allowance for Loan Losses to Total Loans
            ],
            'earnings': [
                'ROA',           # Return on Assets
                'ROE',           # Return on Equity
                'NIM',           # Net Interest Margin
                'NONINTEXP'      # Noninterest Expense to Average Assets
            ],
            'liquidity': [
                'LIQUIDASSETST', # Liquid Assets to Total Assets
                'INTBEARING',    # Interest-Bearing Deposits to Total Deposits
                'COREDEP',       # Core Deposits to Total Assets
                'LNLSNET'        # Net Loans and Leases to Total Assets
            ]
        }
        
        # Asset size categories for analysis
        self.size_categories = {
            'large': 50000000,      # $50B+ (systemically important)
            'regional': 10000000,   # $10B+ (regional banks)
            'community': 1000000,   # $1B+ (community banks)
            'small': 300000         # $300M+ (small banks)
        }
    
    async def fetch_institution_data(self, limit: int = 5000) -> pd.DataFrame:
        """
        Fetch institution directory data
        
        Args:
            limit: Maximum number of institutions to fetch
            
        Returns:
            DataFrame with institution information
        """
        try:
            await self.connector.connect()
            
            endpoint = "institutions"
            params = {
                'limit': limit,
                'format': 'json',
                'filters': 'ACTIVE:1',  # Only active institutions
                'fields': 'CERT,NAME,CITY,STALP,ASSET,DEP,NETINC,ROA,ROE,TIER1RISKCAT,TOTCAPRISKCAT,LEVERAGE,DATEUPDT,INSDATE,EFFDATE,PROCDATE,CERT,CBSA,CBSA_DIV_NO,CBSA_DIV,CBSA_METRO,CBSA_METRO_NAME,CSA,CSA_NAME,COUNTY,COUNTY_NAME,FED,FDICSUPV,FEDIRID,FED_RSSD,FLDOFF,HCTMULT,INSAGNT1,INSAGNT2,INSCOML,INSDIF,INSSAVE,INSURED,MUTUAL,NEWCERT,OAKAR,OFFDOM,OFFFOR,OTSDIST,REGAGNT,REPDTE,ROAQ,ROAPTX,ROAY,RSSDHCR,STNAME,SUBCHAPS,TRACT,UNINUM,WEBADDR'
            }
            
            data = await self.connector.fetch_data(endpoint, params)
            
            if not data or 'data' not in data:
                raise Exception("No institution data received")
            
            institutions_df = pd.DataFrame(data['data'])
            
            # Clean and process data
            institutions_df = self._process_institution_data(institutions_df)
            
            self.logger.info(f"Fetched data for {len(institutions_df)} institutions")
            return institutions_df
            
        except Exception as e:
            self.logger.error(f"Error fetching institution data: {str(e)}")
            
            # Try cached data
            cached_data = await self._get_cached_institutions()
            if cached_data is not None:
                self.logger.warning("Returning cached institution data")
                return cached_data
            
            raise
    
    def _process_institution_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean institution data"""
        try:
            # Convert numeric columns
            numeric_columns = ['ASSET', 'DEP', 'NETINC', 'ROA', 'ROE', 'TIER1RISKCAT', 'TOTCAPRISKCAT', 'LEVERAGE']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert dates
            date_columns = ['DATEUPDT', 'INSDATE', 'EFFDATE', 'PROCDATE', 'REPDTE']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Add size categories
            df['size_category'] = df['ASSET'].apply(self._categorize_bank_size)
            
            # Add risk indicators
            df['capital_risk'] = df.apply(self._assess_capital_risk, axis=1)
            df['performance_risk'] = df.apply(self._assess_performance_risk, axis=1)
            
            # Filter out inactive or problem institutions
            df = df[df['ASSET'] > 0]  # Must have positive assets
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing institution data: {str(e)}")
            return df
    
    def _categorize_bank_size(self, assets: float) -> str:
        """Categorize bank by asset size"""
        if pd.isna(assets):
            return 'unknown'
        
        assets_thousands = assets  # FDIC reports in thousands
        
        if assets_thousands >= self.size_categories['large']:
            return 'large'
        elif assets_thousands >= self.size_categories['regional']:
            return 'regional'
        elif assets_thousands >= self.size_categories['community']:
            return 'community'
        elif assets_thousands >= self.size_categories['small']:
            return 'small'
        else:
            return 'micro'
    
    def _assess_capital_risk(self, row: pd.Series) -> str:
        """Assess capital adequacy risk"""
        try:
            tier1_ratio = row.get('TIER1RISKCAT', np.nan)
            total_ratio = row.get('TOTCAPRISKCAT', np.nan)
            leverage_ratio = row.get('LEVERAGE', np.nan)
            
            risk_score = 0
            
            # Tier 1 Capital Ratio thresholds
            if not pd.isna(tier1_ratio):
                if tier1_ratio < 6.0:  # Below well-capitalized threshold
                    risk_score += 3
                elif tier1_ratio < 8.0:  # Below well-capitalized for large banks
                    risk_score += 1
            
            # Total Capital Ratio thresholds  
            if not pd.isna(total_ratio):
                if total_ratio < 10.0:  # Below well-capitalized threshold
                    risk_score += 3
                elif total_ratio < 12.0:
                    risk_score += 1
            
            # Leverage Ratio thresholds
            if not pd.isna(leverage_ratio):
                if leverage_ratio < 4.0:  # Below well-capitalized threshold
                    risk_score += 2
                elif leverage_ratio < 5.0:
                    risk_score += 1
            
            # Convert score to risk level
            if risk_score >= 5:
                return "high"
            elif risk_score >= 2:
                return "moderate"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _assess_performance_risk(self, row: pd.Series) -> str:
        """Assess performance risk based on ROA and ROE"""
        try:
            roa = row.get('ROA', np.nan)
            roe = row.get('ROE', np.nan)
            
            risk_score = 0
            
            # ROA thresholds (annual percentage)
            if not pd.isna(roa):
                if roa < 0:  # Negative ROA
                    risk_score += 3
                elif roa < 0.5:  # Below 0.5%
                    risk_score += 2
                elif roa < 1.0:  # Below 1.0%
                    risk_score += 1
            
            # ROE thresholds
            if not pd.isna(roe):
                if roe < 0:  # Negative ROE
                    risk_score += 3
                elif roe < 5.0:  # Below 5%
                    risk_score += 2
                elif roe < 10.0:  # Below 10%
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
    
    async def fetch_failed_banks(self) -> pd.DataFrame:
        """Fetch failed bank list"""
        try:
            endpoint = "failures"
            params = {
                'format': 'json',
                'limit': 1000
            }
            
            data = await self.connector.fetch_data(endpoint, params)
            
            if not data or 'data' not in data:
                return pd.DataFrame()
            
            failed_df = pd.DataFrame(data['data'])
            
            # Process failed bank data
            if not failed_df.empty:
                if 'FAILDATE' in failed_df.columns:
                    failed_df['FAILDATE'] = pd.to_datetime(failed_df['FAILDATE'], errors='coerce')
                
                # Add recent failure indicator
                recent_threshold = datetime.now() - timedelta(days=365)
                failed_df['recent_failure'] = failed_df['FAILDATE'] > recent_threshold
            
            self.logger.info(f"Fetched {len(failed_df)} failed bank records")
            return failed_df
            
        except Exception as e:
            self.logger.error(f"Error fetching failed banks: {str(e)}")
            return pd.DataFrame()
    
    async def get_banking_health_indicators(self) -> Dict[str, Any]:
        """Get banking sector health indicators"""
        try:
            # Fetch institution data
            institutions_df = await self.fetch_institution_data()
            
            if institutions_df.empty:
                return {"error": "No institution data available"}
            
            # Fetch failed banks
            failed_df = await self.fetch_failed_banks()
            
            # Calculate sector metrics
            health_indicators = self._calculate_sector_health(institutions_df, failed_df)
            
            return health_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating banking health indicators: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_sector_health(self, institutions_df: pd.DataFrame, failed_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate banking sector health metrics"""
        try:
            sector_health = {
                "sector_overview": {},
                "size_analysis": {},
                "risk_distribution": {},
                "performance_metrics": {},
                "recent_failures": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Sector overview
            total_institutions = len(institutions_df)
            total_assets = institutions_df['ASSET'].sum()
            total_deposits = institutions_df['DEP'].sum()
            
            sector_health["sector_overview"] = {
                "total_institutions": total_institutions,
                "total_assets_thousands": total_assets,
                "total_deposits_thousands": total_deposits,
                "average_asset_size": total_assets / total_institutions if total_institutions > 0 else 0
            }
            
            # Size analysis
            size_analysis = institutions_df.groupby('size_category').agg({
                'CERT': 'count',
                'ASSET': 'sum',
                'DEP': 'sum'
            }).reset_index()
            
            for _, row in size_analysis.iterrows():
                sector_health["size_analysis"][row['size_category']] = {
                    "institution_count": int(row['CERT']),
                    "total_assets": row['ASSET'],
                    "total_deposits": row['DEP'],
                    "market_share": (row['ASSET'] / total_assets * 100) if total_assets > 0 else 0
                }
            
            # Risk distribution
            capital_risk_dist = institutions_df['capital_risk'].value_counts()
            performance_risk_dist = institutions_df['performance_risk'].value_counts()
            
            sector_health["risk_distribution"] = {
                "capital_risk": {
                    "high": int(capital_risk_dist.get('high', 0)),
                    "moderate": int(capital_risk_dist.get('moderate', 0)),
                    "low": int(capital_risk_dist.get('low', 0))
                },
                "performance_risk": {
                    "high": int(performance_risk_dist.get('high', 0)),
                    "moderate": int(performance_risk_dist.get('moderate', 0)),
                    "low": int(performance_risk_dist.get('low', 0))
                }
            }
            
            # Performance metrics
            sector_health["performance_metrics"] = {
                "median_roa": institutions_df['ROA'].median(),
                "median_roe": institutions_df['ROE'].median(),
                "median_tier1_ratio": institutions_df['TIER1RISKCAT'].median(),
                "median_leverage_ratio": institutions_df['LEVERAGE'].median(),
                "profitable_institutions_pct": (institutions_df['ROA'] > 0).mean() * 100
            }
            
            # Recent failures analysis
            if not failed_df.empty:
                recent_failures = failed_df[failed_df['recent_failure'] == True] if 'recent_failure' in failed_df.columns else failed_df
                
                sector_health["recent_failures"] = {
                    "total_failures_year": len(recent_failures),
                    "failure_rate": (len(recent_failures) / total_institutions * 100) if total_institutions > 0 else 0
                }
                
                if 'COST' in failed_df.columns:
                    sector_health["recent_failures"]["total_cost"] = failed_df['COST'].sum()
            
            return sector_health
            
        except Exception as e:
            self.logger.error(f"Error calculating sector health: {str(e)}")
            return {"error": str(e)}
    
    async def _cache_data(self, data: Dict[str, Any], cache_key: str):
        """Cache FDIC data"""
        try:
            await self.cache.set(cache_key, data, ttl=3600)
            self.logger.debug(f"FDIC data cached with key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to cache FDIC data: {str(e)}")
    
    async def _get_cached_institutions(self) -> Optional[pd.DataFrame]:
        """Get cached institution data"""
        try:
            cache_key = "fdic_institutions"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data and 'data' in cached_data:
                df = pd.DataFrame(cached_data['data'])
                if 'DATEUPDT' in df.columns:
                    df['DATEUPDT'] = pd.to_datetime(df['DATEUPDT'])
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached FDIC data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check FDIC API health"""
        try:
            # Test with a simple request for a few institutions
            endpoint = "institutions"
            params = {
                'limit': 10,
                'format': 'json',
                'filters': 'ACTIVE:1'
            }
            
            data = await self.connector.fetch_data(endpoint, params)
            
            return {
                "status": "healthy",
                "institutions_available": len(data.get('data', [])) if data else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close connection"""
        if self.connector:
            await self.connector.close()