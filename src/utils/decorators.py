"""
Custom decorators for RiskX platform.
Provides reusable decorators for caching, performance monitoring, security, and error handling.
"""

import time
import functools
import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar, cast
from datetime import datetime, timedelta

from ..core.exceptions import (
    RiskXBaseException, handle_exception, CacheError, 
    ProcessingError, AuthenticationError, RateLimitExceededError
)
from ..core.logging import performance_logger, security_logger
from ..core.security import security_middleware, audit_log
from .constants import CacheConfig, ModelConfig, SecurityConfig

logger = logging.getLogger('riskx.utils.decorators')

F = TypeVar('F', bound=Callable[..., Any])


def cache_result(ttl: int = CacheConfig.DEFAULT_TTL, 
                key_prefix: str = "", 
                use_args: bool = True,
                use_kwargs: bool = True) -> Callable[[F], F]:
    """
    Decorator to cache function results with TTL.
    Uses in-memory cache with fallback to prevent external dependencies.
    """
    _cache: Dict[str, Dict[str, Any]] = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key_parts = [key_prefix, func.__name__]
            
            if use_args:
                cache_key_parts.extend(str(arg) for arg in args)
            
            if use_kwargs:
                cache_key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            
            cache_key = "|".join(cache_key_parts)
            
            # Check cache
            current_time = time.time()
            if cache_key in _cache:
                cache_entry = _cache[cache_key]
                if current_time - cache_entry["timestamp"] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_entry["value"]
                else:
                    # Expired entry
                    del _cache[cache_key]
            
            # Cache miss - execute function
            logger.debug(f"Cache miss for {func.__name__}")
            try:
                result = func(*args, **kwargs)
                
                # Store in cache
                _cache[cache_key] = {
                    "value": result,
                    "timestamp": current_time
                }
                
                # Clean up old entries (simple LRU-like behavior)
                if len(_cache) > CacheConfig.MAX_CACHE_ENTRIES:
                    oldest_key = min(_cache.keys(), 
                                   key=lambda k: _cache[k]["timestamp"])
                    del _cache[oldest_key]
                
                return result
                
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                raise
        
        return cast(F, wrapper)
    
    return decorator


def async_cache_result(ttl: int = CacheConfig.DEFAULT_TTL,
                      key_prefix: str = "",
                      use_args: bool = True,
                      use_kwargs: bool = True) -> Callable[[F], F]:
    """Async version of cache_result decorator."""
    _cache: Dict[str, Dict[str, Any]] = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key (same logic as sync version)
            cache_key_parts = [key_prefix, func.__name__]
            
            if use_args:
                cache_key_parts.extend(str(arg) for arg in args)
            
            if use_kwargs:
                cache_key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            
            cache_key = "|".join(cache_key_parts)
            
            # Check cache
            current_time = time.time()
            if cache_key in _cache:
                cache_entry = _cache[cache_key]
                if current_time - cache_entry["timestamp"] < ttl:
                    logger.debug(f"Async cache hit for {func.__name__}")
                    return cache_entry["value"]
                else:
                    del _cache[cache_key]
            
            # Cache miss - execute async function
            logger.debug(f"Async cache miss for {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                
                _cache[cache_key] = {
                    "value": result,
                    "timestamp": current_time
                }
                
                # Clean up old entries
                if len(_cache) > CacheConfig.MAX_CACHE_ENTRIES:
                    oldest_key = min(_cache.keys(), 
                                   key=lambda k: _cache[k]["timestamp"])
                    del _cache[oldest_key]
                
                return result
                
            except Exception as e:
                logger.error(f"Error in async cached function {func.__name__}: {e}")
                raise
        
        return cast(F, wrapper)
    
    return decorator


def measure_performance(log_slow_threshold_ms: float = 1000.0,
                       include_args: bool = False) -> Callable[[F], F]:
    """
    Decorator to measure and log function performance.
    Logs warning for functions exceeding threshold.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Log performance metrics
                log_data = {
                    'function': func.__name__,
                    'execution_time_ms': execution_time,
                    'module': func.__module__
                }
                
                if include_args and args:
                    log_data['args_count'] = len(args)
                if include_args and kwargs:
                    log_data['kwargs_count'] = len(kwargs)
                
                # Log to performance logger
                performance_logger.logger.info(
                    f"Function {func.__name__} executed",
                    extra=log_data
                )
                
                # Log warning for slow functions
                if execution_time > log_slow_threshold_ms:
                    logger.warning(
                        f"Slow function execution: {func.__name__} took {execution_time:.2f}ms"
                    )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {func.__name__} failed after {execution_time:.2f}ms: {e}"
                )
                raise
        
        return cast(F, wrapper)
    
    return decorator


def async_measure_performance(log_slow_threshold_ms: float = 1000.0,
                            include_args: bool = False) -> Callable[[F], F]:
    """Async version of measure_performance decorator."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                log_data = {
                    'function': func.__name__,
                    'execution_time_ms': execution_time,
                    'module': func.__module__,
                    'async': True
                }
                
                if include_args and args:
                    log_data['args_count'] = len(args)
                if include_args and kwargs:
                    log_data['kwargs_count'] = len(kwargs)
                
                performance_logger.logger.info(
                    f"Async function {func.__name__} executed",
                    extra=log_data
                )
                
                if execution_time > log_slow_threshold_ms:
                    logger.warning(
                        f"Slow async function: {func.__name__} took {execution_time:.2f}ms"
                    )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Async function {func.__name__} failed after {execution_time:.2f}ms: {e}"
                )
                raise
        
        return cast(F, wrapper)
    
    return decorator


def retry_on_failure(max_attempts: int = 3,
                    delay: float = 1.0,
                    backoff_factor: float = 2.0,
                    exceptions: tuple = (Exception,),
                    on_retry: Optional[Callable] = None) -> Callable[[F], F]:
    """
    Decorator to retry function execution on failure.
    Supports exponential backoff and custom retry callbacks.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {sleep_time:.2f}s"
                        )
                        
                        if on_retry:
                            try:
                                on_retry(attempt, e, sleep_time)
                            except Exception as retry_error:
                                logger.error(f"Error in retry callback: {retry_error}")
                        
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            if last_exception:
                raise last_exception
        
        return cast(F, wrapper)
    
    return decorator


def async_retry_on_failure(max_attempts: int = 3,
                          delay: float = 1.0,
                          backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,),
                          on_retry: Optional[Callable] = None) -> Callable[[F], F]:
    """Async version of retry_on_failure decorator."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        
                        logger.warning(
                            f"Async attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {sleep_time:.2f}s"
                        )
                        
                        if on_retry:
                            try:
                                if asyncio.iscoroutinefunction(on_retry):
                                    await on_retry(attempt, e, sleep_time)
                                else:
                                    on_retry(attempt, e, sleep_time)
                            except Exception as retry_error:
                                logger.error(f"Error in async retry callback: {retry_error}")
                        
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_attempts} async attempts failed for {func.__name__}: {e}"
                        )
            
            if last_exception:
                raise last_exception
        
        return cast(F, wrapper)
    
    return decorator


def validate_input(**validators) -> Callable[[F], F]:
    """
    Decorator to validate function input arguments.
    validators should be a dict of param_name: validation_function pairs.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter mapping
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    try:
                        # Call validator function
                        if not validator(value):
                            raise ValueError(f"Validation failed for parameter '{param_name}' with value: {value}")
                    except Exception as e:
                        raise ValueError(f"Validation error for parameter '{param_name}': {e}")
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def rate_limit(requests_per_minute: int = 60,
              key_func: Optional[Callable] = None) -> Callable[[F], F]:
    """
    Decorator to implement rate limiting on function calls.
    Uses sliding window approach.
    """
    call_times: Dict[str, list] = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                rate_key = key_func(*args, **kwargs)
            else:
                rate_key = f"{func.__module__}.{func.__name__}"
            
            current_time = time.time()
            window_start = current_time - 60  # 1 minute window
            
            # Clean old entries
            if rate_key in call_times:
                call_times[rate_key] = [
                    call_time for call_time in call_times[rate_key]
                    if call_time > window_start
                ]
            else:
                call_times[rate_key] = []
            
            # Check rate limit
            if len(call_times[rate_key]) >= requests_per_minute:
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {func.__name__}: {requests_per_minute} requests per minute"
                )
            
            # Record current call
            call_times[rate_key].append(current_time)
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def require_api_key(api_key_param: str = "api_key") -> Callable[[F], F]:
    """
    Decorator to require API key authentication.
    Validates API key from function parameters.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from kwargs
            api_key = kwargs.get(api_key_param)
            
            if not api_key:
                raise AuthenticationError(f"Missing required parameter: {api_key_param}")
            
            # Validate API key (simplified validation)
            if not isinstance(api_key, str) or len(api_key) < 16:
                raise AuthenticationError("Invalid API key format")
            
            # Log authentication attempt
            security_logger.log_authentication_attempt(
                user_id=api_key[:8] + "...",  # Partial key for logging
                success=True,
                ip_address="unknown",
                user_agent="api_call"
            )
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def handle_exceptions(default_return: Any = None,
                     log_errors: bool = True,
                     reraise: bool = False) -> Callable[[F], F]:
    """
    Decorator to handle exceptions gracefully.
    Can return default value or reraise as RiskX exceptions.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except RiskXBaseException:
                # Already a RiskX exception, just reraise
                raise
                
            except Exception as e:
                if log_errors:
                    logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                
                if reraise:
                    # Convert to RiskX exception
                    riskx_exception = handle_exception(e)
                    raise riskx_exception
                else:
                    return default_return
        
        return cast(F, wrapper)
    
    return decorator


def deprecated(reason: str = "", version: str = "") -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.
    Logs warning when deprecated function is called.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"Function {func.__name__} is deprecated"
            
            if version:
                warning_msg += f" since version {version}"
            
            if reason:
                warning_msg += f": {reason}"
            
            logger.warning(warning_msg)
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def singleton(cls: type) -> type:
    """
    Decorator to implement singleton pattern for classes.
    Ensures only one instance of the class exists.
    """
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def log_function_calls(include_args: bool = False,
                      include_result: bool = False,
                      max_arg_length: int = 100) -> Callable[[F], F]:
    """
    Decorator to log function calls for debugging and auditing.
    Can optionally include arguments and return values.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_data = {
                'function': func.__name__,
                'module': func.__module__
            }
            
            if include_args:
                if args:
                    truncated_args = [
                        str(arg)[:max_arg_length] + "..." if len(str(arg)) > max_arg_length else str(arg)
                        for arg in args
                    ]
                    log_data['args'] = truncated_args
                
                if kwargs:
                    truncated_kwargs = {
                        k: str(v)[:max_arg_length] + "..." if len(str(v)) > max_arg_length else str(v)
                        for k, v in kwargs.items()
                    }
                    log_data['kwargs'] = truncated_kwargs
            
            logger.debug(f"Calling function {func.__name__}", extra=log_data)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    result_str = str(result)
                    if len(result_str) > max_arg_length:
                        result_str = result_str[:max_arg_length] + "..."
                    log_data['result'] = result_str
                
                logger.debug(f"Function {func.__name__} completed", extra=log_data)
                return result
                
            except Exception as e:
                log_data['error'] = str(e)
                logger.error(f"Function {func.__name__} failed", extra=log_data)
                raise
        
        return cast(F, wrapper)
    
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to add timeout to function execution.
    Raises TimeoutError if function takes longer than specified seconds.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return cast(F, wrapper)
    
    return decorator


# Convenience combinations
def cached_with_performance(ttl: int = CacheConfig.DEFAULT_TTL,
                           key_prefix: str = "",
                           slow_threshold_ms: float = 1000.0) -> Callable[[F], F]:
    """Convenience decorator combining caching and performance monitoring."""
    def decorator(func: F) -> F:
        # Apply decorators in reverse order
        decorated = measure_performance(slow_threshold_ms)(func)
        decorated = cache_result(ttl, key_prefix)(decorated)
        return decorated
    
    return decorator


def robust_api_call(max_retries: int = 3,
                   cache_ttl: int = CacheConfig.SHORT_TTL,
                   timeout_seconds: float = 30.0) -> Callable[[F], F]:
    """Convenience decorator for robust API calls with retry, cache, and timeout."""
    def decorator(func: F) -> F:
        # Apply decorators in reverse order
        decorated = timeout(timeout_seconds)(func)
        decorated = retry_on_failure(max_retries)(decorated)
        decorated = cache_result(cache_ttl)(decorated)
        decorated = measure_performance()(decorated)
        return decorated
    
    return decorator