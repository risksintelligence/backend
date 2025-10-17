"""
Application configuration management.
"""
import os
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "RiskX"
    debug: bool = False
    environment: str = "development"
    secret_key: str = "dev-secret-key-change-in-production"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Database
    database_url: str = "sqlite:///./riskx_dev.db"
    postgres_url: Optional[str] = None  # For production PostgreSQL
    database_pool_size: int = 10
    database_pool_overflow: int = 20
    database_pool_timeout: int = 30
    
    # Cache
    redis_url: str = "redis://localhost:6379"
    redis_expire_time: int = 3600
    cache_ttl: int = 1800
    enable_redis: bool = True
    enable_postgres_cache: bool = True
    
    # External APIs
    fred_api_key: Optional[str] = None
    census_api_key: Optional[str] = None
    bea_api_key: Optional[str] = None
    bls_api_key: Optional[str] = None
    
    # ML Platform
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "riskx_experiments"
    
    # Security
    cors_origins: List[str] = [
        "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", 
        "http://localhost:3003", "http://localhost:8000", "http://127.0.0.1:3000",
        "https://riskx.onrender.com", "https://riskx-backend.onrender.com", 
        "https://riskx-frontend.onrender.com", "https://backend-1-il1e.onrender.com",
        "https://frontend-1-il1e.onrender.com", "https://*.onrender.com"
    ]
    allowed_hosts: List[str] = [
        "localhost", "127.0.0.1", "0.0.0.0", 
        "riskx.onrender.com", "riskx-backend.onrender.com", 
        "riskx-frontend.onrender.com", "backend-1-il1e.onrender.com",
        "frontend-1-il1e.onrender.com", "*.onrender.com"
    ]
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # WebSocket Configuration
    enable_websockets: bool = True  # Enabled for real-time risk broadcasting
    
    # Data
    data_update_interval: str = "daily"
    retrain_interval: str = "weekly"  # Renamed to avoid protected namespace
    
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": (),
        "extra": "ignore",
        "env_prefix": "",  # Allow direct env var names like DEBUG, ENVIRONMENT
        "env_nested_delimiter": "__"
    }
    
    @property
    def effective_database_url(self) -> str:
        """
        Get the effective database URL based on environment.
        
        Returns:
            Database URL to use
        """
        # Use PostgreSQL URL if provided (production)
        if self.postgres_url:
            return self.postgres_url
        
        # Use DATABASE_URL if set (Render deployment)
        if hasattr(self, '_database_url_override'):
            return self._database_url_override
            
        # Fall back to configured database_url (development)
        return self.database_url
    
    def set_database_url_override(self, url: str) -> None:
        """Set database URL override (for environment variables)."""
        self._database_url_override = url


# Global settings instance
settings = Settings()

# Override database URL from environment if present
import os
if os.getenv("DATABASE_URL"):
    settings.set_database_url_override(os.getenv("DATABASE_URL"))


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings