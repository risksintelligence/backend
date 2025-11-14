"""
Historical data backfill job for RIS
Fetches 5+ years of data from FRED, Yahoo Finance, and other sources
"""
import asyncio
import asyncpg
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import os
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data series configuration with 5+ year lookback
SERIES_CONFIG = {
    # Financial Stress Indicators
    'VIX': {
        'provider': 'FRED',
        'series_id': 'VIXCLS',
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'index'
    },
    'TREASURY_10Y': {
        'provider': 'FRED', 
        'series_id': 'DGS10',
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'percent'
    },
    'TREASURY_2Y': {
        'provider': 'FRED',
        'series_id': 'DGS2',
        'years_back': 10, 
        'frequency': 'daily',
        'unit': 'percent'
    },
    'CREDIT_SPREAD_BAA': {
        'provider': 'FRED',
        'series_id': 'BAA10YM',  # Moody's Baa - 10Y Treasury
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'percentage_points'
    },
    
    # Supply Chain Indicators
    'BALTIC_DRY_INDEX': {
        'provider': 'FRED',
        'series_id': 'BDIY',
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'index'
    },
    'PMI_MANUFACTURING': {
        'provider': 'FRED',
        'series_id': 'NAPM',  # ISM Manufacturing PMI
        'years_back': 10,
        'frequency': 'monthly',
        'unit': 'index'
    },
    'WTI_OIL': {
        'provider': 'FRED',
        'series_id': 'DCOILWTICO',
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'dollars_per_barrel'
    },
    
    # Macro Indicators
    'CPI_ALL_ITEMS': {
        'provider': 'FRED',
        'series_id': 'CPIAUCSL',
        'years_back': 10,
        'frequency': 'monthly', 
        'unit': 'index_1982_1984_100'
    },
    'UNEMPLOYMENT_RATE': {
        'provider': 'FRED',
        'series_id': 'UNRATE',
        'years_back': 10,
        'frequency': 'monthly',
        'unit': 'percent'
    },
    
    # Additional series for robustness
    'SP500': {
        'provider': 'FRED',
        'series_id': 'SP500',
        'years_back': 10,
        'frequency': 'daily',
        'unit': 'index'
    },
    'INDUSTRIAL_PRODUCTION': {
        'provider': 'FRED',
        'series_id': 'INDPRO',
        'years_back': 10,
        'frequency': 'monthly',
        'unit': 'index_2017_100'
    }
}

class HistoricalDataBackfill:
    def __init__(self, postgres_dsn: str, fred_api_key: str):
        self.postgres_dsn = postgres_dsn
        self.fred_api_key = fred_api_key
        self.session = None
        self.db_pool = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.db_pool = await asyncpg.create_pool(self.postgres_dsn)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()
    
    async def fetch_fred_series(self, series_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical data from FRED API"""
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'asc'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"FRED API error for {series_id}: {response.status}")
                return []
                
            data = await response.json()
            observations = data.get('observations', [])
            
            # Filter out missing values
            valid_observations = []
            for obs in observations:
                if obs['value'] != '.' and obs['value'] is not None:
                    try:
                        float_value = float(obs['value'])
                        valid_observations.append({
                            'date': obs['date'],
                            'value': float_value
                        })
                    except (ValueError, TypeError):
                        continue
                        
            logger.info(f"Fetched {len(valid_observations)} valid observations for {series_id}")
            return valid_observations
    
    async def store_observations(self, series_key: str, series_config: Dict, observations: List[Dict]) -> int:
        """Store observations in raw_observations table"""
        if not observations:
            return 0
            
        insert_count = 0
        async with self.db_pool.acquire() as conn:
            for obs in observations:
                try:
                    await conn.execute("""
                        INSERT INTO raw_observations 
                        (series_id, source, observed_at, value, unit, source_url, fetched_at)
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        ON CONFLICT (series_id, observed_at) 
                        DO UPDATE SET 
                            value = EXCLUDED.value,
                            fetched_at = NOW()
                    """, 
                    series_key,
                    series_config['provider'],
                    datetime.strptime(obs['date'], '%Y-%m-%d'),
                    obs['value'],
                    series_config['unit'],
                    f"https://fred.stlouisfed.org/series/{series_config['series_id']}"
                    )
                    insert_count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert observation for {series_key}: {e}")
                    continue
                    
        logger.info(f"Stored {insert_count} observations for {series_key}")
        return insert_count
    
    async def backfill_series(self, series_key: str, series_config: Dict) -> int:
        """Backfill historical data for a single series"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * series_config['years_back'])).strftime('%Y-%m-%d')
        
        logger.info(f"Backfilling {series_key} from {start_date} to {end_date}")
        
        if series_config['provider'] == 'FRED':
            observations = await self.fetch_fred_series(
                series_config['series_id'], 
                start_date, 
                end_date
            )
            return await self.store_observations(series_key, series_config, observations)
        else:
            logger.warning(f"Provider {series_config['provider']} not implemented yet")
            return 0
    
    async def run_backfill(self, series_list: List[str] = None) -> Dict[str, int]:
        """Run backfill for specified series or all series"""
        if series_list is None:
            series_list = list(SERIES_CONFIG.keys())
            
        results = {}
        
        for series_key in series_list:
            if series_key not in SERIES_CONFIG:
                logger.warning(f"Unknown series: {series_key}")
                continue
                
            try:
                count = await self.backfill_series(series_key, SERIES_CONFIG[series_key])
                results[series_key] = count
                
                # Small delay to respect API rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to backfill {series_key}: {e}")
                results[series_key] = 0
                
        return results
    
    async def verify_data_coverage(self) -> Dict[str, Dict]:
        """Verify we have sufficient data coverage for each series"""
        async with self.db_pool.acquire() as conn:
            coverage_info = {}
            
            for series_key in SERIES_CONFIG.keys():
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_observations,
                        MIN(observed_at) as earliest_date,
                        MAX(observed_at) as latest_date,
                        EXTRACT(EPOCH FROM (MAX(observed_at) - MIN(observed_at))) / (365.25 * 24 * 3600) as years_span
                    FROM raw_observations 
                    WHERE series_id = $1
                """, series_key)
                
                if result:
                    coverage_info[series_key] = {
                        'total_observations': result['total_observations'],
                        'earliest_date': result['earliest_date'],
                        'latest_date': result['latest_date'],
                        'years_span': float(result['years_span']) if result['years_span'] else 0,
                        'sufficient_for_ml': result['years_span'] and result['years_span'] >= 5
                    }
                else:
                    coverage_info[series_key] = {
                        'total_observations': 0,
                        'sufficient_for_ml': False
                    }
                    
        return coverage_info

async def main():
    """Main backfill execution"""
    # Get environment variables
    postgres_dsn = os.environ.get('RIS_POSTGRES_DSN')
    fred_api_key = os.environ.get('RIS_FRED_API_KEY', '08cfc7737f0b2840a85fd6054d5ba7af')  # From env_reference.md
    
    if not postgres_dsn:
        logger.error("RIS_POSTGRES_DSN not set")
        return
        
    logger.info("Starting historical data backfill...")
    logger.info(f"Target: 5+ years of data for {len(SERIES_CONFIG)} series")
    
    async with HistoricalDataBackfill(postgres_dsn, fred_api_key) as backfill:
        # Run the backfill
        results = await backfill.run_backfill()
        
        # Report results
        total_observations = sum(results.values())
        logger.info(f"Backfill completed: {total_observations} total observations")
        
        for series_key, count in results.items():
            logger.info(f"  {series_key}: {count} observations")
        
        # Verify coverage
        logger.info("Verifying data coverage...")
        coverage = await backfill.verify_data_coverage()
        
        insufficient_series = []
        for series_key, info in coverage.items():
            years = info.get('years_span', 0)
            sufficient = info.get('sufficient_for_ml', False)
            obs_count = info.get('total_observations', 0)
            
            status = "✅ SUFFICIENT" if sufficient else "❌ INSUFFICIENT"
            logger.info(f"  {series_key}: {obs_count} obs, {years:.1f} years {status}")
            
            if not sufficient:
                insufficient_series.append(series_key)
        
        if insufficient_series:
            logger.warning(f"Insufficient data for ML training: {insufficient_series}")
            logger.info("Consider extending backfill period or checking data availability")
        else:
            logger.info("✅ All series have sufficient data (5+ years) for ML training")

if __name__ == "__main__":
    asyncio.run(main())