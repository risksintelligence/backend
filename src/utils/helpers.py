"""
General utility functions for RiskX platform.
Provides commonly used helper functions for data processing, formatting, and validation.
"""

import re
import hashlib
import json
import math
import statistics
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple
from decimal import Decimal, ROUND_HALF_UP
from urllib.parse import urlparse, parse_qs
import logging

logger = logging.getLogger('riskx.utils.helpers')

T = TypeVar('T')


# Date and Time Utilities

def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def parse_date_string(date_str: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """Parse date string with multiple format attempts."""
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y%m%d'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_str}")
    return None


def format_date_for_display(dt: datetime, format_type: str = 'default') -> str:
    """Format datetime for display purposes."""
    formats = {
        'default': '%Y-%m-%d %H:%M:%S',
        'date_only': '%Y-%m-%d',
        'time_only': '%H:%M:%S',
        'iso': '%Y-%m-%dT%H:%M:%SZ',
        'friendly': '%B %d, %Y at %I:%M %p',
        'compact': '%Y%m%d_%H%M%S'
    }
    
    return dt.strftime(formats.get(format_type, formats['default']))


def get_business_days_between(start_date: date, end_date: date) -> int:
    """Calculate number of business days between two dates."""
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days


def get_date_range(start_date: date, end_date: date, 
                  frequency: str = 'daily') -> List[date]:
    """Generate list of dates between start and end date."""
    dates = []
    current_date = start_date
    
    delta_map = {
        'daily': timedelta(days=1),
        'weekly': timedelta(weeks=1),
        'monthly': timedelta(days=30),  # Approximate
        'quarterly': timedelta(days=90)  # Approximate
    }
    
    delta = delta_map.get(frequency, timedelta(days=1))
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta
    
    return dates


# Numeric Utilities

def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def round_to_precision(value: float, precision: int = 2) -> float:
    """Round number to specified precision using banker's rounding."""
    if math.isnan(value) or math.isinf(value):
        return value
    
    decimal_value = Decimal(str(value))
    rounded = decimal_value.quantize(
        Decimal('0.' + '0' * precision),
        rounding=ROUND_HALF_UP
    )
    return float(rounded)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0.0
    
    return ((new_value - old_value) / abs(old_value)) * 100


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range."""
    if max_val == min_val:
        return 0.5  # If no range, return middle value
    
    return (value - min_val) / (max_val - min_val)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max bounds."""
    return max(min_val, min(value, max_val))


def calculate_moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average with specified window size."""
    if window <= 0 or window > len(values):
        return values.copy()
    
    moving_averages = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        moving_averages.append(statistics.mean(window_values))
    
    return moving_averages


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


# String Utilities

def clean_string(text: str, remove_extra_spaces: bool = True, 
                strip_chars: Optional[str] = None) -> str:
    """Clean and normalize string input."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Strip specified characters
    if strip_chars:
        text = text.strip(strip_chars)
    else:
        text = text.strip()
    
    # Remove extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    return text


def truncate_string(text: str, max_length: int, 
                   suffix: str = '...') -> str:
    """Truncate string to maximum length with optional suffix."""
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])


def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text string."""
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]


def format_number(value: Union[int, float], format_type: str = 'default') -> str:
    """Format number for display purposes."""
    if math.isnan(value) or math.isinf(value):
        return 'N/A'
    
    formats = {
        'default': '{:,.2f}',
        'integer': '{:,}',
        'percentage': '{:.1%}',
        'currency': '${:,.2f}',
        'scientific': '{:.2e}',
        'compact': '{:.1f}K' if abs(value) >= 1000 else '{:.1f}'
    }
    
    fmt = formats.get(format_type, formats['default'])
    
    if format_type == 'compact' and abs(value) >= 1000000:
        fmt = '{:.1f}M'
        value = value / 1000000
    elif format_type == 'compact' and abs(value) >= 1000:
        value = value / 1000
    
    return fmt.format(value)


# Data Structure Utilities

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(nested_dict: Dict[str, Any], 
                separator: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary using separator."""
    def _flatten(obj, parent_key='', sep='.'):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(nested_dict, sep=separator)


def unflatten_dict(flat_dict: Dict[str, Any], 
                  separator: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary using separator."""
    result = {}
    
    for key, value in flat_dict.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def filter_dict(data: Dict[str, Any], 
               condition: Callable[[str, Any], bool]) -> Dict[str, Any]:
    """Filter dictionary based on condition function."""
    return {k: v for k, v in data.items() if condition(k, v)}


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates(lst: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """Remove duplicates from list while preserving order."""
    if key is None:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in lst:
            k = key(item)
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result


# Validation Utilities

def is_valid_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception:
        return False


def is_numeric(value: Any) -> bool:
    """Check if value is numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def validate_range(value: float, min_val: Optional[float] = None, 
                  max_val: Optional[float] = None) -> bool:
    """Validate value is within specified range."""
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


# Hashing and Security Utilities

def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """Generate hash of string data."""
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
        else:
            key_parts.append(f"{key}:{value}")
    
    key_string = '|'.join(key_parts)
    return generate_hash(key_string)


# File and Path Utilities

def safe_filename(filename: str, max_length: int = 255) -> str:
    """Create safe filename by removing/replacing invalid characters."""
    # Remove invalid characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    safe_chars = safe_chars.strip('. ')
    
    # Truncate if too long
    if len(safe_chars) > max_length:
        name, ext = os.path.splitext(safe_chars)
        max_name_length = max_length - len(ext)
        safe_chars = name[:max_name_length] + ext
    
    return safe_chars or 'untitled'


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return os.path.splitext(filename)[1].lower()


# Retry and Error Handling Utilities

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    exceptions: Tuple = (Exception,)):
    """Decorator to retry function on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def safe_execute(func: Callable, default_value: Any = None, 
                log_errors: bool = True) -> Any:
    """Safely execute function and return default on error."""
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing function {func.__name__}: {e}")
        return default_value


# Configuration and Environment Utilities

def parse_bool(value: Union[str, bool, int, None]) -> bool:
    """Parse various types to boolean."""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, int):
        return bool(value)
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    return False


def parse_list(value: Union[str, List, None], separator: str = ',') -> List[str]:
    """Parse string or list to list of strings."""
    if value is None:
        return []
    
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    return [str(value)]


def get_nested_value(data: Dict[str, Any], key_path: str, 
                    default: Any = None, separator: str = '.') -> Any:
    """Get nested dictionary value using dot notation."""
    keys = key_path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: Dict[str, Any], key_path: str, 
                    value: Any, separator: str = '.') -> None:
    """Set nested dictionary value using dot notation."""
    keys = key_path.split(separator)
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


# Performance and Monitoring Utilities

def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"Function {func.__name__} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise
    return wrapper


def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024