"""
Data storage utilities package for RiskX platform.
Provides database, cache, and file storage interfaces with fallback mechanisms.
"""

from .database import (
    DatabaseManager,
    DatabaseConfig,
    ConnectionPool,
    create_database_manager,
    get_database_connection
)

from .cache import (
    CacheManager,
    CacheConfig,
    CacheBackend,
    RedisCacheBackend,
    MemoryCacheBackend,
    FileCacheBackend,
    create_cache_manager,
    get_cache_instance
)

from .files import (
    FileManager,
    FileConfig,
    StorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
    create_file_manager,
    get_file_storage
)

__all__ = [
    # Database utilities
    'DatabaseManager',
    'DatabaseConfig',
    'ConnectionPool',
    'create_database_manager',
    'get_database_connection',
    
    # Cache utilities
    'CacheManager',
    'CacheConfig',
    'CacheBackend',
    'RedisCacheBackend',
    'MemoryCacheBackend',
    'FileCacheBackend',
    'create_cache_manager',
    'get_cache_instance',
    
    # File storage utilities
    'FileManager',
    'FileConfig',
    'StorageBackend',
    'LocalStorageBackend',
    'S3StorageBackend',
    'create_file_manager',
    'get_file_storage'
]