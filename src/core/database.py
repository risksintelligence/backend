"""
Database initialization and session management.
"""
import logging
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.core.config import settings

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Database engine configuration
def _get_engine_config():
    """Get database engine configuration based on database type."""
    db_url = settings.effective_database_url
    
    config = {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "echo": settings.debug,  # Log SQL queries in debug mode
    }
    
    if "sqlite" in db_url:
        # SQLite-specific configuration
        config.update({
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False}
        })
    elif "postgresql" in db_url:
        # PostgreSQL-specific configuration
        config.update({
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_pool_overflow,
            "pool_timeout": settings.database_pool_timeout,
            "pool_reset_on_return": "commit"
        })
    
    return config

# Database engine
engine = create_engine(settings.effective_database_url, **_get_engine_config())

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    try:
        # Import all models to register them with Base
        from src.data.models import economic_data, risk_data, system_data
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_db_connection() -> bool:
    """
    Check database connection health.
    
    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Database event listeners for logging
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance (if using SQLite)."""
    if "sqlite" in settings.effective_database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        cursor.close()


@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries in debug mode."""
    if settings.debug:
        import time
        context._query_start_time = time.time()


@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query execution time in debug mode."""
    if settings.debug and hasattr(context, '_query_start_time'):
        import time
        total = time.time() - context._query_start_time
        if total > 0.1:  # Log queries taking more than 100ms
            logger.warning(f"Slow query ({total:.3f}s): {statement[:100]}...")