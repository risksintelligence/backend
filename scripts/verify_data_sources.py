#!/usr/bin/env python3
"""
Data Source Integration Verification Script

Tests all data sources to verify real API connections, fallback mechanisms,
and cache integrity for the RiskX platform.
"""
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sources.fred import FREDConnector
from src.data.sources.bea import BEAConnector
from src.data.sources.census import CensusTradeDataFetcher
from src.data.sources.bls import BlsDataFetcher
from src.data.sources.fdic import FdicDataFetcher
from src.data.sources.cisa import CISADataSource
from src.data.sources.noaa import NOAADataSource
from src.data.sources.usgs import USGSDataSource
from src.cache.cache_manager import CacheManager
from src.core.config import get_settings
# from src.core.logging import setup_logging

logger = logging.getLogger(__name__)


class DataSourceVerifier:
    """Comprehensive data source verification and testing."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        
        # Initialize data sources
        self.data_sources = {
            'fred': FREDConnector(cache_manager=self.cache_manager),
            'bea': BEAConnector(),
            'census': CensusTradeDataFetcher(),
            'bls': BlsDataFetcher(),
            'fdic': FdicDataFetcher(),
            'cisa': CISADataSource(),
            'noaa': NOAADataSource(),
            'usgs': USGSDataSource()
        }
        
        # Test results
        self.results = {}
    
    async def verify_all_sources(self) -> Dict[str, Any]:
        """
        Verify all data sources for API connectivity, cache functionality,
        and data quality.
        
        Returns:
            Comprehensive verification results
        """
        logger.info("Starting comprehensive data source verification")
        
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'summary': {
                'total_sources': len(self.data_sources),
                'api_connected': 0,
                'cache_working': 0,
                'data_quality_passed': 0,
                'fully_operational': 0
            }
        }
        
        # Test each data source
        for source_name, source_instance in self.data_sources.items():
            logger.info(f"Verifying {source_name}...")
            
            source_results = await self._verify_single_source(source_name, source_instance)
            verification_results['sources'][source_name] = source_results
            
            # Update summary
            if source_results['api_connected']:
                verification_results['summary']['api_connected'] += 1
            if source_results['cache_working']:
                verification_results['summary']['cache_working'] += 1
            if source_results['data_quality_ok']:
                verification_results['summary']['data_quality_passed'] += 1
            if source_results['fully_operational']:
                verification_results['summary']['fully_operational'] += 1
        
        # Calculate success rates
        total = verification_results['summary']['total_sources']
        verification_results['summary']['success_rates'] = {
            'api_connectivity': f"{verification_results['summary']['api_connected']}/{total}",
            'cache_functionality': f"{verification_results['summary']['cache_working']}/{total}",
            'data_quality': f"{verification_results['summary']['data_quality_passed']}/{total}",
            'overall_operational': f"{verification_results['summary']['fully_operational']}/{total}"
        }
        
        logger.info("Data source verification completed")
        return verification_results
    
    async def _verify_single_source(self, source_name: str, source_instance: Any) -> Dict[str, Any]:
        """
        Verify a single data source.
        
        Args:
            source_name: Name of the data source
            source_instance: Instance of the data source class
            
        Returns:
            Verification results for the source
        """
        results = {
            'source_name': source_name,
            'api_connected': False,
            'cache_working': False,
            'data_retrieved': False,
            'data_quality_ok': False,
            'fallback_available': False,
            'fully_operational': False,
            'error_messages': [],
            'performance_metrics': {}
        }
        
        start_time = datetime.now()
        
        try:
            # Test 1: API Connectivity
            logger.info(f"Testing {source_name} API connectivity...")
            api_test_result = await self._test_api_connectivity(source_instance)
            results['api_connected'] = api_test_result['connected']
            if not api_test_result['connected']:
                results['error_messages'].append(f"API connectivity failed: {api_test_result['error']}")
            
            # Test 2: Data Retrieval
            logger.info(f"Testing {source_name} data retrieval...")
            data_test_result = await self._test_data_retrieval(source_instance)
            results['data_retrieved'] = data_test_result['success']
            results['data_quality_ok'] = data_test_result['quality_ok']
            if not data_test_result['success']:
                results['error_messages'].append(f"Data retrieval failed: {data_test_result['error']}")
            
            # Test 3: Cache Functionality
            logger.info(f"Testing {source_name} cache functionality...")
            cache_test_result = await self._test_cache_functionality(source_name, source_instance)
            results['cache_working'] = cache_test_result['working']
            if not cache_test_result['working']:
                results['error_messages'].append(f"Cache functionality failed: {cache_test_result['error']}")
            
            # Test 4: Fallback Mechanisms
            logger.info(f"Testing {source_name} fallback mechanisms...")
            fallback_test_result = await self._test_fallback_mechanisms(source_instance)
            results['fallback_available'] = fallback_test_result['available']
            
            # Calculate performance metrics
            end_time = datetime.now()
            results['performance_metrics'] = {
                'total_test_time_seconds': (end_time - start_time).total_seconds(),
                'api_response_time': api_test_result.get('response_time', 0),
                'cache_response_time': cache_test_result.get('response_time', 0)
            }
            
            # Determine if fully operational
            results['fully_operational'] = (
                results['api_connected'] and 
                results['cache_working'] and 
                results['data_quality_ok']
            )
            
        except Exception as e:
            logger.error(f"Error verifying {source_name}: {e}")
            results['error_messages'].append(f"Verification error: {str(e)}")
        
        return results
    
    async def _test_api_connectivity(self, source_instance: Any) -> Dict[str, Any]:
        """Test API connectivity for a data source."""
        try:
            start_time = datetime.now()
            
            # Try to establish connection
            if hasattr(source_instance, 'connector'):
                connected = await source_instance.connector.connect()
            elif hasattr(source_instance, 'test_connection'):
                connected = await source_instance.test_connection()
            else:
                # Fallback: try to fetch a small amount of data
                test_data = await self._get_test_data(source_instance)
                connected = test_data is not None
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'connected': connected,
                'response_time': response_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'connected': False,
                'response_time': 0,
                'error': str(e)
            }
    
    async def _test_data_retrieval(self, source_instance: Any) -> Dict[str, Any]:
        """Test data retrieval capabilities."""
        try:
            # Get test data
            test_data = await self._get_test_data(source_instance)
            
            if test_data is None:
                return {'success': False, 'quality_ok': False, 'error': 'No data returned'}
            
            # Basic data quality checks
            quality_ok = self._check_data_quality(test_data)
            
            return {
                'success': True,
                'quality_ok': quality_ok,
                'error': None,
                'data_sample': str(test_data)[:200] if test_data else None
            }
            
        except Exception as e:
            return {'success': False, 'quality_ok': False, 'error': str(e)}
    
    async def _test_cache_functionality(self, source_name: str, source_instance: Any) -> Dict[str, Any]:
        """Test cache functionality."""
        try:
            start_time = datetime.now()
            
            # Test cache write/read
            test_key = f"verification_test_{source_name}_{int(datetime.now().timestamp())}"
            test_value = {'test': 'data', 'timestamp': datetime.now().isoformat()}
            
            # Write to cache
            if hasattr(source_instance, 'cache'):
                source_instance.cache.set(test_key, test_value, ttl=300)
                
                # Read from cache
                cached_value = source_instance.cache.get(test_key)
                
                # Cleanup
                source_instance.cache.delete(test_key)
                
                working = cached_value is not None
            else:
                working = False
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'working': working,
                'response_time': response_time,
                'error': None
            }
            
        except Exception as e:
            return {'working': False, 'response_time': 0, 'error': str(e)}
    
    async def _test_fallback_mechanisms(self, source_instance: Any) -> Dict[str, Any]:
        """Test fallback mechanisms."""
        try:
            # Check if fallback data is available
            if hasattr(source_instance, '_get_fallback_data'):
                fallback_data = await source_instance._get_fallback_data()
                available = fallback_data is not None
            elif hasattr(source_instance, 'get_cached_data'):
                fallback_data = await source_instance.get_cached_data()
                available = fallback_data is not None
            else:
                available = False
            
            return {'available': available, 'error': None}
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    async def _get_test_data(self, source_instance: Any) -> Any:
        """Get a small amount of test data from a source."""
        try:
            # Try different methods to get test data
            if hasattr(source_instance, 'get_economic_indicators'):
                return await source_instance.get_economic_indicators()
            elif hasattr(source_instance, 'fetch_latest_data'):
                return await source_instance.fetch_latest_data()
            elif hasattr(source_instance, 'get_latest_indicators'):
                return await source_instance.get_latest_indicators()
            elif hasattr(source_instance, 'fetch_recent_data'):
                return await source_instance.fetch_recent_data()
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get test data: {e}")
            return None
    
    def _check_data_quality(self, data: Any) -> bool:
        """Basic data quality checks."""
        try:
            if data is None:
                return False
            
            # Check if data has content
            if hasattr(data, '__len__') and len(data) == 0:
                return False
            
            # If it's a dictionary, check for basic structure
            if isinstance(data, dict):
                return len(data) > 0
            
            # If it's a DataFrame, check for rows and columns
            if hasattr(data, 'shape'):
                return data.shape[0] > 0 and data.shape[1] > 0
            
            return True
            
        except:
            return False


async def main():
    """Main verification script entry point."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Verify RiskX data source integrations')
    parser.add_argument('--source', help='Verify specific source only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Save results to file')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize verifier
        verifier = DataSourceVerifier()
        
        # Run verification
        if args.source:
            logger.info(f"Verifying single source: {args.source}")
            if args.source not in verifier.data_sources:
                print(f"Error: Unknown source '{args.source}'")
                print(f"Available sources: {list(verifier.data_sources.keys())}")
                sys.exit(1)
            
            source_instance = verifier.data_sources[args.source]
            result = await verifier._verify_single_source(args.source, source_instance)
            results = {'sources': {args.source: result}}
        else:
            logger.info("Verifying all data sources")
            results = await verifier.verify_all_sources()
        
        # Display results
        print("\n" + "="*80)
        print("DATA SOURCE VERIFICATION RESULTS")
        print("="*80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\nSUMMARY:")
            print(f"Total Sources: {summary['total_sources']}")
            print(f"API Connected: {summary['success_rates']['api_connectivity']}")
            print(f"Cache Working: {summary['success_rates']['cache_functionality']}")
            print(f"Data Quality OK: {summary['success_rates']['data_quality']}")
            print(f"Fully Operational: {summary['success_rates']['overall_operational']}")
        
        print(f"\nDETAILED RESULTS:")
        for source_name, source_result in results['sources'].items():
            status = "✅ OPERATIONAL" if source_result['fully_operational'] else "⚠️ ISSUES"
            print(f"\n{source_name.upper()}: {status}")
            print(f"  API Connected: {'✅' if source_result['api_connected'] else '❌'}")
            print(f"  Cache Working: {'✅' if source_result['cache_working'] else '❌'}")
            print(f"  Data Quality: {'✅' if source_result['data_quality_ok'] else '❌'}")
            print(f"  Fallback Available: {'✅' if source_result['fallback_available'] else '❌'}")
            
            if source_result['error_messages']:
                print(f"  Errors: {'; '.join(source_result['error_messages'])}")
            
            perf = source_result.get('performance_metrics', {})
            if perf:
                print(f"  Performance: {perf.get('total_test_time_seconds', 0):.2f}s total")
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        
        # Exit with error if any sources have issues
        if 'summary' in results:
            operational_count = results['summary']['fully_operational']
            total_count = results['summary']['total_sources']
            
            if operational_count < total_count:
                print(f"\n⚠️ WARNING: {total_count - operational_count} sources have issues")
                sys.exit(1)
            else:
                print(f"\n✅ SUCCESS: All {total_count} sources are fully operational")
        
    except Exception as e:
        logger.error(f"Verification script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())