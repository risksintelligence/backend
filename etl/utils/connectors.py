"""
ETL Data Connectors

Provides base classes and implementations for connecting to various data sources
including APIs, databases, and file systems with robust error handling and caching.
"""

import aiohttp
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import time
import json
import hashlib

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager


class BaseConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"connector.{name}")
        self.cache = CacheManager()
        self.settings = get_settings()
        
        # Rate limiting configuration
        self.rate_limit = self.config.get('rate_limit', 100)  # requests per minute
        self.rate_window = 60  # seconds
        self.request_timestamps = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    async def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from the source"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health status of the connection"""
        pass
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        
        # Remove timestamps older than the rate window
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < self.rate_window
        ]
        
        # Check if we can make another request
        if len(self.request_timestamps) >= self.rate_limit:
            return False
        
        # Record this request
        self.request_timestamps.append(now)
        return True
    
    async def _wait_for_rate_limit(self):
        """Wait until we can make another request"""
        while not self._check_rate_limit():
            await asyncio.sleep(1)
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """Generate a cache key for the request"""
        key_data = f"{self.name}:{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()


class APIConnector(BaseConnector):
    """HTTP API connector with rate limiting and caching"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.base_url = self.config.get('base_url', '')
        self.headers = self.config.get('headers', {})
        self.timeout = self.config.get('timeout', 30)
        self.session = None
        
        # Cache configuration
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        self.use_cache = self.config.get('use_cache', True)
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        try:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.headers
            )
            
            self.logger.info(f"Connected to {self.name} API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.name}: {str(e)}")
            return False
    
    async def fetch_data(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from API endpoint with caching and rate limiting"""
        try:
            # Check cache first
            if self.use_cache:
                cache_key = self._generate_cache_key(endpoint, params)
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    self.logger.debug(f"Cache hit for {endpoint}")
                    return cached_data
            
            # Ensure we have a session
            if not self.session:
                await self.connect()
            
            # Wait for rate limit
            await self._wait_for_rate_limit()
            
            # Build URL
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            # Make request
            self.logger.debug(f"Fetching from {url} with params: {params}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the response
                    if self.use_cache:
                        await self.cache.set(cache_key, data, ttl=self.cache_ttl)
                    
                    self.logger.debug(f"Successfully fetched data from {endpoint}")
                    return data
                
                else:
                    error_text = await response.text()
                    self.logger.error(f"API error {response.status}: {error_text}")
                    
                    # Try to return cached data on error
                    if self.use_cache:
                        cache_key = self._generate_cache_key(endpoint, params)
                        cached_data = await self.cache.get(cache_key)
                        if cached_data:
                            self.logger.warning(f"Returning stale cache data for {endpoint}")
                            return cached_data
                    
                    raise Exception(f"API returned {response.status}: {error_text}")
        
        except Exception as e:
            self.logger.error(f"Error fetching from {endpoint}: {str(e)}")
            
            # Try fallback cache
            if self.use_cache:
                cache_key = self._generate_cache_key(endpoint, params)
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    self.logger.warning(f"Returning fallback cache data for {endpoint}")
                    return cached_data
            
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        health_endpoint = self.config.get('health_endpoint', '')
        
        if not health_endpoint:
            return {"status": "unknown", "message": "No health endpoint configured"}
        
        try:
            start_time = time.time()
            data = await self.fetch_data(health_endpoint)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info(f"Closed {self.name} connection")


class DatabaseConnector(BaseConnector):
    """Database connector for PostgreSQL"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.connection_string = self.config.get('connection_string')
        self.pool = None
    
    async def connect(self) -> bool:
        """Initialize database connection pool"""
        try:
            import asyncpg
            
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            self.logger.info(f"Connected to {self.name} database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.name} database: {str(e)}")
            return False
    
    async def fetch_data(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        try:
            if not self.pool:
                await self.connect()
            
            async with self.pool.acquire() as connection:
                rows = await connection.fetch(query, *(params or ()))
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Database query error: {str(e)}")
            raise
    
    async def execute(self, query: str, params: tuple = None) -> int:
        """Execute query and return affected rows"""
        try:
            if not self.pool:
                await self.connect()
            
            async with self.pool.acquire() as connection:
                result = await connection.execute(query, *(params or ()))
                return int(result.split()[-1]) if result else 0
                
        except Exception as e:
            self.logger.error(f"Database execute error: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            result = await self.fetch_data("SELECT 1 as health_check")
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                "connection_pool_size": len(self.pool._holders) if self.pool else 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.logger.info(f"Closed {self.name} database connection")


class ConnectorFactory:
    """Factory for creating connector instances"""
    
    @staticmethod
    def create_connector(connector_type: str, name: str, config: Dict[str, Any]) -> BaseConnector:
        """Create a connector instance based on type"""
        if connector_type.lower() == 'api':
            return APIConnector(name, config)
        elif connector_type.lower() == 'database':
            return DatabaseConnector(name, config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")


class ConnectorManager:
    """Manages multiple connectors"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self.logger = logging.getLogger("connector_manager")
    
    def add_connector(self, name: str, connector: BaseConnector):
        """Add a connector to the manager"""
        self.connectors[name] = connector
        self.logger.info(f"Added connector: {name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all registered connectors"""
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = await connector.connect()
            except Exception as e:
                self.logger.error(f"Failed to connect {name}: {str(e)}")
                results[name] = False
        return results
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check all connectors"""
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {str(e)}")
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        return results
    
    async def close_all(self):
        """Close all connectors"""
        for name, connector in self.connectors.items():
            try:
                await connector.close()
            except Exception as e:
                self.logger.error(f"Error closing {name}: {str(e)}")
        
        self.connectors.clear()
        self.logger.info("All connectors closed")
    
    def get_connector(self, name: str) -> Optional[BaseConnector]:
        """Get a connector by name"""
        return self.connectors.get(name)