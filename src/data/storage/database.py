"""
Database storage utilities for RiskX platform.
Provides PostgreSQL database management with connection pooling and fallback mechanisms.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import pandas as pd
from datetime import datetime

from ...core.exceptions import DatabaseError, ConfigurationError
from ...utils.helpers import safe_divide, parse_bool
from ...utils.constants import StorageConfig

logger = logging.getLogger('riskx.data.storage.database')


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "riskx"
    username: str = "postgres"
    password: str = ""
    min_connections: int = 1
    max_connections: int = 20
    connection_timeout: int = 30
    query_timeout: int = 300
    ssl_mode: str = "prefer"
    application_name: str = "RiskX"
    
    @classmethod
    def from_url(cls, database_url: str) -> 'DatabaseConfig':
        """Create config from database URL."""
        try:
            import urllib.parse as urlparse
            
            parsed = urlparse.urlparse(database_url)
            
            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/') if parsed.path else "riskx",
                username=parsed.username or "postgres",
                password=parsed.password or "",
                ssl_mode="require" if "sslmode" not in database_url else "prefer"
            )
        except Exception as e:
            logger.error(f"Error parsing database URL: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return cls.from_url(database_url)
        
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'riskx'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONNECTIONS', '1')),
            max_connections=int(os.getenv('DB_MAX_CONNECTIONS', '20')),
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', '30')),
            query_timeout=int(os.getenv('DB_QUERY_TIMEOUT', '300')),
            ssl_mode=os.getenv('DB_SSL_MODE', 'prefer'),
            application_name=os.getenv('DB_APPLICATION_NAME', 'RiskX')
        )


class ConnectionPool:
    """Database connection pool manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool = None
        self._async_pool = None
        self._is_connected = False
    
    def initialize_sync_pool(self):
        """Initialize synchronous connection pool."""
        try:
            connection_string = (
                f"host={self.config.host} "
                f"port={self.config.port} "
                f"dbname={self.config.database} "
                f"user={self.config.username} "
                f"password={self.config.password} "
                f"sslmode={self.config.ssl_mode} "
                f"application_name={self.config.application_name}"
            )
            
            self._pool = ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                dsn=connection_string
            )
            
            # Test connection
            with self.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            
            self._is_connected = True
            logger.info("Synchronous database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize sync database pool: {e}")
            self._is_connected = False
            raise DatabaseError(f"Database pool initialization failed: {e}")
    
    async def initialize_async_pool(self):
        """Initialize asynchronous connection pool."""
        try:
            connection_string = (
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
                f"?sslmode={self.config.ssl_mode}&application_name={self.config.application_name}"
            )
            
            self._async_pool = await asyncpg.create_pool(
                connection_string,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.query_timeout
            )
            
            # Test connection
            async with self._async_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            self._is_connected = True
            logger.info("Asynchronous database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database pool: {e}")
            self._is_connected = False
            raise DatabaseError(f"Async database pool initialization failed: {e}")
    
    @contextmanager
    def get_sync_connection(self):
        """Get synchronous database connection."""
        if not self._pool:
            raise DatabaseError("Sync connection pool not initialized")
        
        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get asynchronous database connection."""
        if not self._async_pool:
            raise DatabaseError("Async connection pool not initialized")
        
        async with self._async_pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Async database connection error: {e}")
                raise DatabaseError(f"Async database operation failed: {e}")
    
    def close(self):
        """Close all connections."""
        try:
            if self._pool:
                self._pool.closeall()
                self._pool = None
            
            if self._async_pool:
                asyncio.create_task(self._async_pool.close())
                self._async_pool = None
            
            self._is_connected = False
            logger.info("Database connection pools closed")
            
        except Exception as e:
            logger.error(f"Error closing database pools: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected


class DatabaseManager:
    """Comprehensive database management with fallback mechanisms."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self._table_cache = {}
        self._fallback_data = {}
        self._stats = {
            'queries_executed': 0,
            'queries_failed': 0,
            'fallback_used': 0,
            'cache_hits': 0
        }
    
    def initialize(self):
        """Initialize database connections."""
        try:
            self.pool.initialize_sync_pool()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed, using fallback mode: {e}")
    
    async def initialize_async(self):
        """Initialize async database connections."""
        try:
            await self.pool.initialize_async_pool()
            logger.info("Async database manager initialized successfully")
        except Exception as e:
            logger.warning(f"Async database initialization failed, using fallback mode: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None, 
                     fetch_mode: str = 'all') -> Optional[List[Dict[str, Any]]]:
        """Execute SQL query with fallback handling."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, using fallback data")
                return self._get_fallback_data(query)
            
            with self.pool.get_sync_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch_mode == 'all':
                        result = cursor.fetchall()
                    elif fetch_mode == 'one':
                        result = cursor.fetchone()
                        result = [result] if result else []
                    elif fetch_mode == 'many':
                        result = cursor.fetchmany(1000)
                    else:
                        result = []
                    
                    conn.commit()
                    self._stats['queries_executed'] += 1
                    
                    # Convert to list of dictionaries
                    return [dict(row) for row in result] if result else []
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Query execution failed: {e}")
            logger.info("Attempting to use fallback data")
            return self._get_fallback_data(query)
    
    async def execute_query_async(self, query: str, params: Optional[List] = None,
                                 fetch_mode: str = 'all') -> Optional[List[Dict[str, Any]]]:
        """Execute async SQL query with fallback handling."""
        try:
            if not self.pool.is_connected:
                logger.warning("Async database not connected, using fallback data")
                return self._get_fallback_data(query)
            
            async with self.pool.get_async_connection() as conn:
                if fetch_mode == 'all':
                    result = await conn.fetch(query, *(params or []))
                elif fetch_mode == 'one':
                    result = await conn.fetchrow(query, *(params or []))
                    result = [result] if result else []
                else:
                    result = await conn.fetch(query, *(params or []))
                
                self._stats['queries_executed'] += 1
                
                # Convert to list of dictionaries
                return [dict(row) for row in result] if result else []
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Async query execution failed: {e}")
            logger.info("Attempting to use fallback data")
            return self._get_fallback_data(query)
    
    def insert_data(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                   on_conflict: str = 'ignore') -> bool:
        """Insert data into database table."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, storing to fallback")
                return self._store_fallback_data(table, data)
            
            if isinstance(data, dict):
                data = [data]
            
            if not data:
                return True
            
            # Generate INSERT query
            columns = list(data[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)
            
            if on_conflict == 'ignore':
                conflict_clause = 'ON CONFLICT DO NOTHING'
            elif on_conflict == 'update':
                update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns])
                conflict_clause = f'ON CONFLICT DO UPDATE SET {update_clause}'
            else:
                conflict_clause = ''
            
            query = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                {conflict_clause}
            """
            
            with self.pool.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    for row in data:
                        values = tuple(row[col] for col in columns)
                        cursor.execute(query, values)
                    
                    conn.commit()
                    self._stats['queries_executed'] += 1
                    logger.info(f"Inserted {len(data)} rows into {table}")
                    return True
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Data insertion failed: {e}")
            return self._store_fallback_data(table, data)
    
    def update_data(self, table: str, data: Dict[str, Any], 
                   where_clause: str, where_params: Optional[tuple] = None) -> bool:
        """Update data in database table."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, cannot update")
                return False
            
            # Generate UPDATE query
            set_clause = ', '.join([f"{col} = %s" for col in data.keys()])
            values = tuple(data.values())
            
            if where_params:
                params = values + where_params
            else:
                params = values
            
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            
            with self.pool.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows_affected = cursor.rowcount
                    conn.commit()
                    
                    self._stats['queries_executed'] += 1
                    logger.info(f"Updated {rows_affected} rows in {table}")
                    return rows_affected > 0
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Data update failed: {e}")
            return False
    
    def delete_data(self, table: str, where_clause: str, 
                   where_params: Optional[tuple] = None) -> bool:
        """Delete data from database table."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, cannot delete")
                return False
            
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            with self.pool.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, where_params)
                    rows_affected = cursor.rowcount
                    conn.commit()
                    
                    self._stats['queries_executed'] += 1
                    logger.info(f"Deleted {rows_affected} rows from {table}")
                    return rows_affected > 0
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Data deletion failed: {e}")
            return False
    
    def read_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Read data into pandas DataFrame."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, returning empty DataFrame")
                return pd.DataFrame()
            
            with self.pool.get_sync_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                self._stats['queries_executed'] += 1
                logger.info(f"Read DataFrame with {len(df)} rows")
                return df
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"DataFrame read failed: {e}")
            return pd.DataFrame()
    
    def write_dataframe(self, df: pd.DataFrame, table: str, 
                       if_exists: str = 'append', index: bool = False) -> bool:
        """Write pandas DataFrame to database."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, cannot write DataFrame")
                return False
            
            with self.pool.get_sync_connection() as conn:
                df.to_sql(table, conn, if_exists=if_exists, index=index, method='multi')
                conn.commit()
                
                self._stats['queries_executed'] += 1
                logger.info(f"Wrote DataFrame with {len(df)} rows to {table}")
                return True
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"DataFrame write failed: {e}")
            return False
    
    def table_exists(self, table: str) -> bool:
        """Check if table exists in database."""
        if table in self._table_cache:
            self._stats['cache_hits'] += 1
            return self._table_cache[table]
        
        try:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """
            result = self.execute_query(query, (table,), fetch_mode='one')
            exists = result[0]['exists'] if result else False
            
            self._table_cache[table] = exists
            return exists
        
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table: str, 
                                   primary_key: Optional[str] = None) -> bool:
        """Create table from DataFrame structure."""
        try:
            if not self.pool.is_connected:
                logger.warning("Database not connected, cannot create table")
                return False
            
            # Generate CREATE TABLE statement
            type_mapping = {
                'object': 'TEXT',
                'int64': 'INTEGER',
                'float64': 'REAL',
                'bool': 'BOOLEAN',
                'datetime64[ns]': 'TIMESTAMP'
            }
            
            columns = []
            for col, dtype in df.dtypes.items():
                sql_type = type_mapping.get(str(dtype), 'TEXT')
                is_pk = ' PRIMARY KEY' if col == primary_key else ''
                columns.append(f"{col} {sql_type}{is_pk}")
            
            create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(columns)})"
            
            with self.pool.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_sql)
                    conn.commit()
                    
                    self._stats['queries_executed'] += 1
                    self._table_cache[table] = True
                    logger.info(f"Created table {table}")
                    return True
        
        except Exception as e:
            self._stats['queries_failed'] += 1
            logger.error(f"Table creation failed: {e}")
            return False
    
    def _get_fallback_data(self, query: str) -> List[Dict[str, Any]]:
        """Get fallback data when database is unavailable."""
        self._stats['fallback_used'] += 1
        
        # Extract table name from query for fallback lookup
        query_lower = query.lower()
        if 'from' in query_lower:
            try:
                table_part = query_lower.split('from')[1].strip()
                table_name = table_part.split()[0].strip()
                
                if table_name in self._fallback_data:
                    logger.info(f"Using fallback data for table: {table_name}")
                    return self._fallback_data[table_name]
            except Exception:
                pass
        
        # Return empty result if no fallback data available
        logger.warning("No fallback data available for query")
        return []
    
    def _store_fallback_data(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Store data as fallback when database is unavailable."""
        try:
            if isinstance(data, dict):
                data = [data]
            
            if table not in self._fallback_data:
                self._fallback_data[table] = []
            
            self._fallback_data[table].extend(data)
            logger.info(f"Stored {len(data)} rows as fallback data for {table}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to store fallback data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database operation statistics."""
        stats = self._stats.copy()
        stats['is_connected'] = self.pool.is_connected
        stats['fallback_data_tables'] = list(self._fallback_data.keys())
        stats['success_rate'] = safe_divide(
            stats['queries_executed'], 
            stats['queries_executed'] + stats['queries_failed']
        ) * 100
        return stats
    
    def close(self):
        """Close database connections."""
        self.pool.close()
        logger.info("Database manager closed")


# Global database manager instance
_database_manager = None

def create_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Create and initialize database manager."""
    global _database_manager
    
    if config is None:
        config = DatabaseConfig.from_env()
    
    _database_manager = DatabaseManager(config)
    _database_manager.initialize()
    
    return _database_manager

def get_database_connection() -> Optional[DatabaseManager]:
    """Get global database manager instance."""
    return _database_manager