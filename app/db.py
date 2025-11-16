"""Database configuration and session management with production optimizations."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Production database engine configuration
def create_production_engine():
    """Create database engine with production optimizations."""
    if settings.database_url.startswith("sqlite"):
        # SQLite configuration (development only)
        logger.warning("Using SQLite - not recommended for production")
        return create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False}
        )
    else:
        # PostgreSQL production configuration
        return create_engine(
            settings.database_url,
            # Connection pooling configuration
            pool_size=20,                    # Base connection pool size
            max_overflow=30,                 # Additional connections when needed
            pool_timeout=30,                 # Timeout waiting for connection (seconds)
            pool_recycle=3600,              # Recycle connections every hour
            pool_pre_ping=True,             # Test connections before use
            
            # Performance settings
            echo=False,                     # Disable SQL query logging in production
            future=True,                    # Use SQLAlchemy 2.0 style
            
            # Connection timeout settings
            connect_args={
                "connect_timeout": 10,      # Connection timeout (PostgreSQL)
                "options": "-c statement_timeout=30000",  # Query timeout in ms
                "application_name": "rrio_backend",
            } if settings.database_url.startswith("postgresql") else {}
        )

# Create database engine
engine = create_production_engine()

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
