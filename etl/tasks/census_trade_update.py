"""
Census Trade Data Update Task

ETL task for fetching and processing U.S. trade data from the Census Bureau.
Handles imports, exports, and trade balance calculations with real-time updates.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.data.sources.census import CensusDataSource
from src.data.processors.cleaner import DataCleaner
from src.data.processors.validator import DataValidator
from src.data.storage.database import DatabaseManager
from etl.utils.connectors import create_notification
from etl.utils.validators import validate_trade_data_quality

logger = logging.getLogger(__name__)
settings = get_settings()


class CensusTradeUpdateTask:
    """
    ETL task for Census Bureau trade data updates.
    
    Fetches monthly trade statistics including imports, exports, 
    trade balance, and commodity-level details.
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.census_source = CensusDataSource()
        self.data_cleaner = DataCleaner()
        self.validator = DataValidator()
        self.db_manager = DatabaseManager()
        self.task_name = "census_trade_update"
        
    async def execute(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False,
        notify_on_completion: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the Census trade data update task.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            force_refresh: Force refresh of cached data
            notify_on_completion: Send notification when complete
            
        Returns:
            Dictionary with task execution results
        """
        task_start_time = datetime.utcnow()
        logger.info(f"Starting {self.task_name} ETL task")
        
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=365)  # Last year of data
            
            # Initialize results tracking
            results = {
                'task_name': self.task_name,
                'start_time': task_start_time,
                'status': 'running',
                'records_processed': 0,
                'errors': [],
                'warnings': [],
                'data_quality_score': 0.0
            }
            
            # Step 1: Fetch trade data
            logger.info("Fetching Census trade data")
            trade_data = await self._fetch_trade_data(start_date, end_date, force_refresh)
            
            if trade_data.empty:
                logger.warning("No trade data retrieved")
                results['status'] = 'completed_with_warnings'
                results['warnings'].append("No trade data available for specified period")
                return results
            
            # Step 2: Clean and validate data
            logger.info(f"Processing {len(trade_data)} trade records")
            cleaned_data = await self._clean_trade_data(trade_data)
            validated_data = await self._validate_trade_data(cleaned_data)
            
            # Step 3: Calculate derived metrics
            enhanced_data = await self._calculate_trade_metrics(validated_data)
            
            # Step 4: Store in database
            storage_result = await self._store_trade_data(enhanced_data)
            
            # Step 5: Update cache with latest data
            await self._update_cache(enhanced_data)
            
            # Step 6: Generate summary statistics
            summary_stats = await self._generate_summary_statistics(enhanced_data)
            
            # Update results
            results.update({
                'status': 'completed',
                'end_time': datetime.utcnow(),
                'duration_seconds': (datetime.utcnow() - task_start_time).total_seconds(),
                'records_processed': len(enhanced_data),
                'records_stored': storage_result.get('records_inserted', 0),
                'data_quality_score': await self._calculate_data_quality_score(enhanced_data),
                'summary_statistics': summary_stats
            })
            
            logger.info(f"Completed {self.task_name} ETL task successfully")
            
            # Send notification if requested
            if notify_on_completion:
                await self._send_completion_notification(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {self.task_name} ETL task: {e}")
            results.update({
                'status': 'failed',
                'end_time': datetime.utcnow(),
                'error': str(e)
            })
            
            # Send error notification
            await self._send_error_notification(results, str(e))
            
            return results
    
    async def _fetch_trade_data(
        self,
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool
    ) -> pd.DataFrame:
        """Fetch trade data from Census Bureau"""
        
        try:
            # Use async context manager for proper resource handling
            async with self.census_source as source:
                # Fetch monthly trade statistics
                monthly_trade = await source.get_monthly_trade_data(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Fetch commodity-level details for recent months
                recent_start = max(start_date, end_date - timedelta(days=90))
                commodity_trade = await source.get_commodity_trade_data(
                    start_date=recent_start,
                    end_date=end_date,
                    commodity_level='2'  # 2-digit HS codes
                )
                
                # Fetch country-level trade data
                country_trade = await source.get_country_trade_data(
                    start_date=recent_start,
                    end_date=end_date,
                    top_partners=20  # Top 20 trading partners
                )
                
                # Combine datasets
                combined_data = self._combine_trade_datasets(
                    monthly_trade, commodity_trade, country_trade
                )
                
                return combined_data
                
        except Exception as e:
            logger.error(f"Error fetching Census trade data: {e}")
            return pd.DataFrame()
    
    def _combine_trade_datasets(
        self,
        monthly_data: pd.DataFrame,
        commodity_data: pd.DataFrame,
        country_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine different trade datasets into unified format"""
        
        combined_records = []
        
        # Process monthly aggregate data
        for _, row in monthly_data.iterrows():
            record = {
                'data_type': 'monthly_aggregate',
                'period': row.get('period'),
                'imports_value': row.get('imports_value', 0),
                'exports_value': row.get('exports_value', 0),
                'trade_balance': row.get('trade_balance', 0),
                'country': 'TOTAL',
                'commodity_code': 'TOTAL',
                'commodity_description': 'All Commodities',
                'source': 'census_monthly'
            }
            combined_records.append(record)
        
        # Process commodity-level data
        for _, row in commodity_data.iterrows():
            record = {
                'data_type': 'commodity_detail',
                'period': row.get('period'),
                'imports_value': row.get('imports_value', 0),
                'exports_value': row.get('exports_value', 0),
                'trade_balance': row.get('exports_value', 0) - row.get('imports_value', 0),
                'country': 'TOTAL',
                'commodity_code': row.get('commodity_code'),
                'commodity_description': row.get('commodity_description'),
                'source': 'census_commodity'
            }
            combined_records.append(record)
        
        # Process country-level data
        for _, row in country_data.iterrows():
            record = {
                'data_type': 'country_detail',
                'period': row.get('period'),
                'imports_value': row.get('imports_value', 0),
                'exports_value': row.get('exports_value', 0),
                'trade_balance': row.get('exports_value', 0) - row.get('imports_value', 0),
                'country': row.get('country'),
                'commodity_code': 'TOTAL',
                'commodity_description': 'All Commodities',
                'source': 'census_country'
            }
            combined_records.append(record)
        
        return pd.DataFrame(combined_records)
    
    async def _clean_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize trade data"""
        
        logger.info("Cleaning trade data")
        
        # Use the data cleaner for standard cleaning operations
        cleaned_data = await self.data_cleaner.clean_dataframe(
            data,
            remove_duplicates=True,
            standardize_columns=True,
            handle_missing_values=True
        )
        
        # Trade-specific cleaning
        cleaned_data = cleaned_data.copy()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['imports_value', 'exports_value', 'trade_balance']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce').fillna(0)
        
        # Clean period column
        if 'period' in cleaned_data.columns:
            cleaned_data['period'] = pd.to_datetime(cleaned_data['period'], errors='coerce')
        
        # Standardize country codes
        if 'country' in cleaned_data.columns:
            cleaned_data['country'] = cleaned_data['country'].str.upper().str.strip()
        
        # Clean commodity codes and descriptions
        if 'commodity_code' in cleaned_data.columns:
            cleaned_data['commodity_code'] = cleaned_data['commodity_code'].str.strip()
        if 'commodity_description' in cleaned_data.columns:
            cleaned_data['commodity_description'] = cleaned_data['commodity_description'].str.title()
        
        # Remove rows with invalid data
        cleaned_data = cleaned_data.dropna(subset=['period'])
        
        logger.info(f"Cleaned data: {len(cleaned_data)} records remaining")
        return cleaned_data
    
    async def _validate_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate trade data quality and consistency"""
        
        logger.info("Validating trade data")
        
        # Use the data validator for standard validation
        validation_result = await self.validator.validate_dataframe(
            data,
            required_columns=['period', 'imports_value', 'exports_value'],
            data_types={
                'imports_value': 'numeric',
                'exports_value': 'numeric',
                'trade_balance': 'numeric'
            }
        )
        
        if not validation_result.is_valid:
            logger.warning(f"Data validation issues: {validation_result.errors}")
        
        # Trade-specific validation
        validated_data = data.copy()
        
        # Check for reasonable value ranges
        max_reasonable_value = 1e12  # $1 trillion
        for col in ['imports_value', 'exports_value']:
            if col in validated_data.columns:
                outliers = validated_data[validated_data[col] > max_reasonable_value]
                if not outliers.empty:
                    logger.warning(f"Found {len(outliers)} outliers in {col}")
                    # Cap extreme values
                    validated_data.loc[validated_data[col] > max_reasonable_value, col] = max_reasonable_value
        
        # Validate trade balance consistency
        if all(col in validated_data.columns for col in ['imports_value', 'exports_value', 'trade_balance']):
            calculated_balance = validated_data['exports_value'] - validated_data['imports_value']
            balance_diff = abs(validated_data['trade_balance'] - calculated_balance)
            
            # Flag significant discrepancies
            threshold = validated_data[['imports_value', 'exports_value']].max(axis=1) * 0.01  # 1% threshold
            discrepancies = balance_diff > threshold
            
            if discrepancies.any():
                logger.warning(f"Found {discrepancies.sum()} trade balance discrepancies")
                # Recalculate trade balance
                validated_data.loc[discrepancies, 'trade_balance'] = calculated_balance[discrepancies]
        
        logger.info(f"Validated data: {len(validated_data)} records passed validation")
        return validated_data
    
    async def _calculate_trade_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional trade metrics and indicators"""
        
        logger.info("Calculating trade metrics")
        
        enhanced_data = data.copy()
        
        # Calculate trade intensity (total trade as share of combined value)
        enhanced_data['total_trade_value'] = enhanced_data['imports_value'] + enhanced_data['exports_value']
        
        # Calculate trade coverage ratio (exports/imports)
        enhanced_data['trade_coverage_ratio'] = np.where(
            enhanced_data['imports_value'] > 0,
            enhanced_data['exports_value'] / enhanced_data['imports_value'],
            np.inf
        )
        
        # Calculate relative trade balance (balance/total trade)
        enhanced_data['relative_trade_balance'] = np.where(
            enhanced_data['total_trade_value'] > 0,
            enhanced_data['trade_balance'] / enhanced_data['total_trade_value'],
            0
        )
        
        # Add time-based metrics
        if 'period' in enhanced_data.columns:
            enhanced_data = enhanced_data.sort_values('period')
            
            # Calculate month-over-month growth rates
            for value_col in ['imports_value', 'exports_value', 'total_trade_value']:
                if value_col in enhanced_data.columns:
                    enhanced_data[f'{value_col}_mom_growth'] = enhanced_data.groupby(
                        ['country', 'commodity_code']
                    )[value_col].pct_change()
            
            # Calculate year-over-year growth rates
            for value_col in ['imports_value', 'exports_value', 'total_trade_value']:
                if value_col in enhanced_data.columns:
                    enhanced_data[f'{value_col}_yoy_growth'] = enhanced_data.groupby(
                        ['country', 'commodity_code']
                    )[value_col].pct_change(periods=12)
        
        # Add risk indicators
        enhanced_data['trade_concentration_risk'] = self._calculate_concentration_risk(enhanced_data)
        enhanced_data['volatility_indicator'] = self._calculate_trade_volatility(enhanced_data)
        
        # Add metadata
        enhanced_data['last_updated'] = datetime.utcnow()
        enhanced_data['data_source'] = 'census_bureau'
        enhanced_data['etl_batch_id'] = f"census_trade_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Enhanced data with metrics: {len(enhanced_data)} records")
        return enhanced_data
    
    def _calculate_concentration_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trade concentration risk indicator"""
        
        # Simple concentration measure based on data type
        concentration_risk = pd.Series(0.0, index=data.index)
        
        # Higher risk for country-specific trade
        country_mask = data['data_type'] == 'country_detail'
        concentration_risk[country_mask] = 0.5
        
        # Medium risk for commodity-specific trade
        commodity_mask = data['data_type'] == 'commodity_detail'
        concentration_risk[commodity_mask] = 0.3
        
        # Lower risk for aggregate trade
        aggregate_mask = data['data_type'] == 'monthly_aggregate'
        concentration_risk[aggregate_mask] = 0.1
        
        return concentration_risk
    
    def _calculate_trade_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trade volatility indicator"""
        
        volatility = pd.Series(0.0, index=data.index)
        
        # Calculate volatility based on growth rate standard deviation
        for entity_group in ['country', 'commodity_code']:
            if entity_group in data.columns and 'total_trade_value_mom_growth' in data.columns:
                group_volatility = data.groupby(entity_group)['total_trade_value_mom_growth'].transform('std')
                volatility = np.maximum(volatility, group_volatility.fillna(0))
        
        return volatility
    
    async def _store_trade_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Store trade data in database"""
        
        logger.info(f"Storing {len(data)} trade records in database")
        
        try:
            # Store in trade_data table
            result = await self.db_manager.bulk_insert(
                table_name='census_trade_data',
                data=data.to_dict('records'),
                on_conflict='update'  # Update if record already exists
            )
            
            logger.info(f"Successfully stored {result.get('records_inserted', 0)} trade records")
            return result
            
        except Exception as e:
            logger.error(f"Error storing trade data: {e}")
            return {'records_inserted': 0, 'error': str(e)}
    
    async def _update_cache(self, data: pd.DataFrame) -> None:
        """Update cache with latest trade data"""
        
        try:
            # Cache latest aggregate data
            latest_aggregate = data[
                (data['data_type'] == 'monthly_aggregate') & 
                (data['period'] == data['period'].max())
            ]
            
            if not latest_aggregate.empty:
                await self.cache.set(
                    'census_trade_latest',
                    latest_aggregate.to_dict('records'),
                    ttl=86400  # 24 hours
                )
            
            # Cache summary by country
            country_summary = data[data['data_type'] == 'country_detail'].groupby('country').agg({
                'imports_value': 'sum',
                'exports_value': 'sum',
                'trade_balance': 'sum'
            }).to_dict('index')
            
            await self.cache.set(
                'census_trade_country_summary',
                country_summary,
                ttl=86400
            )
            
            logger.info("Updated trade data cache")
            
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
    
    async def _generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the trade data"""
        
        try:
            summary = {
                'total_records': len(data),
                'date_range': {
                    'start': data['period'].min().isoformat() if not data['period'].empty else None,
                    'end': data['period'].max().isoformat() if not data['period'].empty else None
                },
                'data_types': data['data_type'].value_counts().to_dict(),
                'value_statistics': {
                    'total_imports': data['imports_value'].sum(),
                    'total_exports': data['exports_value'].sum(),
                    'trade_balance': data['trade_balance'].sum(),
                    'avg_trade_coverage_ratio': data['trade_coverage_ratio'].mean()
                },
                'coverage': {
                    'countries': data['country'].nunique(),
                    'commodities': data['commodity_code'].nunique()
                },
                'quality_indicators': {
                    'missing_values_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                    'zero_trade_records': len(data[data['total_trade_value'] == 0])
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            return {}
    
    async def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        
        try:
            score_components = []
            
            # Completeness score (no missing values in key columns)
            key_columns = ['period', 'imports_value', 'exports_value', 'trade_balance']
            completeness = 1.0 - (data[key_columns].isnull().sum().sum() / (len(data) * len(key_columns)))
            score_components.append(completeness * 30)  # 30% weight
            
            # Consistency score (trade balance matches calculated values)
            if all(col in data.columns for col in ['imports_value', 'exports_value', 'trade_balance']):
                calculated_balance = data['exports_value'] - data['imports_value']
                consistency = 1.0 - (abs(data['trade_balance'] - calculated_balance).sum() / data['total_trade_value'].sum())
                score_components.append(max(0, consistency) * 25)  # 25% weight
            
            # Timeliness score (data recency)
            if 'period' in data.columns and not data['period'].empty:
                latest_date = data['period'].max()
                days_old = (datetime.utcnow() - latest_date).days
                timeliness = max(0, 1 - (days_old / 365))  # Decrease score as data gets older
                score_components.append(timeliness * 20)  # 20% weight
            
            # Coverage score (breadth of data)
            coverage = min(1.0, len(data) / 1000)  # Assume 1000 records is good coverage
            score_components.append(coverage * 25)  # 25% weight
            
            return sum(score_components)
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 50.0  # Default score
    
    async def _send_completion_notification(self, results: Dict[str, Any]) -> None:
        """Send notification upon task completion"""
        
        try:
            message = f"""
            Census Trade ETL Task Completed Successfully
            
            Records Processed: {results.get('records_processed', 0)}
            Duration: {results.get('duration_seconds', 0):.1f} seconds
            Data Quality Score: {results.get('data_quality_score', 0):.1f}/100
            
            Summary Statistics:
            - Total Imports: ${results.get('summary_statistics', {}).get('value_statistics', {}).get('total_imports', 0):,.0f}
            - Total Exports: ${results.get('summary_statistics', {}).get('value_statistics', {}).get('total_exports', 0):,.0f}
            - Trade Balance: ${results.get('summary_statistics', {}).get('value_statistics', {}).get('trade_balance', 0):,.0f}
            """
            
            await create_notification(
                title="Census Trade ETL Completed",
                message=message,
                severity="info",
                task_name=self.task_name
            )
            
        except Exception as e:
            logger.error(f"Error sending completion notification: {e}")
    
    async def _send_error_notification(self, results: Dict[str, Any], error_message: str) -> None:
        """Send notification upon task failure"""
        
        try:
            message = f"""
            Census Trade ETL Task Failed
            
            Error: {error_message}
            Duration: {results.get('duration_seconds', 0):.1f} seconds
            
            Please check logs for detailed error information.
            """
            
            await create_notification(
                title="Census Trade ETL Failed",
                message=message,
                severity="error",
                task_name=self.task_name
            )
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")


# Convenience function for external usage
async def run_census_trade_update(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Run the Census trade update ETL task.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        force_refresh: Force refresh of cached data
        
    Returns:
        Dictionary with task execution results
    """
    task = CensusTradeUpdateTask()
    return await task.execute(start_date, end_date, force_refresh)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        result = await run_census_trade_update()
        print(f"Task completed with status: {result['status']}")
        print(f"Processed {result['records_processed']} records")
        
    asyncio.run(main())