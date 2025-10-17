"""
Utilities package for RiskX platform.
Provides commonly used utility functions, decorators, constants, and monitoring tools.
"""

from .helpers import (
    # Date and time utilities
    utc_now, parse_date_string, format_date_for_display,
    get_business_days_between, get_date_range,
    
    # Numeric utilities
    safe_divide, round_to_precision, calculate_percentage_change,
    normalize_value, clamp, calculate_moving_average, calculate_percentile,
    
    # String utilities
    clean_string, truncate_string, snake_to_camel, camel_to_snake,
    extract_numbers, format_number,
    
    # Data structure utilities
    deep_merge_dicts, flatten_dict, unflatten_dict, filter_dict,
    chunk_list, remove_duplicates,
    
    # Validation utilities
    is_valid_email, is_valid_url, is_numeric, validate_range,
    
    # Hashing utilities
    generate_hash, generate_cache_key,
    
    # File utilities
    safe_filename, get_file_extension,
    
    # Configuration utilities
    parse_bool, parse_list, get_nested_value, set_nested_value,
    
    # Performance utilities
    measure_execution_time, memory_usage_mb
)

from .constants import (
    # Application info
    APP_NAME, APP_DESCRIPTION, APP_VERSION, API_VERSION, API_PREFIX,
    
    # Color scheme
    Colors,
    
    # Risk assessment
    RiskLevels, RiskCategories,
    
    # Data sources
    DataSources, EconomicIndicators,
    
    # Configuration
    CacheConfig, ModelConfig, SimulationConfig, APIConfig,
    LogConfig, SecurityConfig, StorageConfig, NetworkConfig,
    
    # Business rules
    BusinessRules, ErrorCodes, RegexPatterns, FeatureFlags
)

from .decorators import (
    # Caching decorators
    cache_result, async_cache_result,
    
    # Performance decorators
    measure_performance, async_measure_performance,
    
    # Retry decorators
    retry_on_failure, async_retry_on_failure,
    
    # Validation decorators
    validate_input, rate_limit, require_api_key,
    
    # Error handling decorators
    handle_exceptions, deprecated, timeout,
    
    # Utility decorators
    singleton, log_function_calls,
    
    # Convenience combinations
    cached_with_performance, robust_api_call
)

from .monitoring import (
    # Data classes
    SystemMetrics, ApplicationMetrics, HealthStatus,
    
    # Monitoring classes
    SystemMonitor, ApplicationMonitor, HealthChecker,
    
    # Built-in health checks
    database_health_check, cache_health_check, external_api_health_check,
    
    # Global instances
    system_monitor, app_monitor, health_checker,
    
    # Setup functions
    setup_monitoring, get_monitoring_summary
)

__all__ = [
    # Helpers
    'utc_now', 'parse_date_string', 'format_date_for_display',
    'get_business_days_between', 'get_date_range',
    'safe_divide', 'round_to_precision', 'calculate_percentage_change',
    'normalize_value', 'clamp', 'calculate_moving_average', 'calculate_percentile',
    'clean_string', 'truncate_string', 'snake_to_camel', 'camel_to_snake',
    'extract_numbers', 'format_number',
    'deep_merge_dicts', 'flatten_dict', 'unflatten_dict', 'filter_dict',
    'chunk_list', 'remove_duplicates',
    'is_valid_email', 'is_valid_url', 'is_numeric', 'validate_range',
    'generate_hash', 'generate_cache_key',
    'safe_filename', 'get_file_extension',
    'parse_bool', 'parse_list', 'get_nested_value', 'set_nested_value',
    'measure_execution_time', 'memory_usage_mb',
    
    # Constants
    'APP_NAME', 'APP_DESCRIPTION', 'APP_VERSION', 'API_VERSION', 'API_PREFIX',
    'Colors',
    'RiskLevels', 'RiskCategories',
    'DataSources', 'EconomicIndicators',
    'CacheConfig', 'ModelConfig', 'SimulationConfig', 'APIConfig',
    'LogConfig', 'SecurityConfig', 'StorageConfig', 'NetworkConfig',
    'BusinessRules', 'ErrorCodes', 'RegexPatterns', 'FeatureFlags',
    
    # Decorators
    'cache_result', 'async_cache_result',
    'measure_performance', 'async_measure_performance',
    'retry_on_failure', 'async_retry_on_failure',
    'validate_input', 'rate_limit', 'require_api_key',
    'handle_exceptions', 'deprecated', 'timeout',
    'singleton', 'log_function_calls',
    'cached_with_performance', 'robust_api_call',
    
    # Monitoring
    'SystemMetrics', 'ApplicationMetrics', 'HealthStatus',
    'SystemMonitor', 'ApplicationMonitor', 'HealthChecker',
    'database_health_check', 'cache_health_check', 'external_api_health_check',
    'system_monitor', 'app_monitor', 'health_checker',
    'setup_monitoring', 'get_monitoring_summary'
]