import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    environment: str = Field("local", env="RIS_ENV")
    
    # Database
    postgres_dsn: str = Field("sqlite:///./test.db", env="RIS_POSTGRES_DSN")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379/0", env="RIS_REDIS_URL")
    
    # Paths
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models")
    
    # Cache
    cache_ttl_seconds: int = Field(900, env="RIS_CACHE_TTL")
    
    # Data Provider API Keys
    fred_api_key: Optional[str] = Field(None, env="RIS_FRED_API_KEY")
    eia_api_key: Optional[str] = Field(None, env="RIS_EIA_API_KEY")
    census_api_key: Optional[str] = Field(None, env="RIS_CENSUS_API_KEY")
    bea_api_key: Optional[str] = Field(None, env="RIS_BEA_API_KEY")
    bls_api_key: Optional[str] = Field(None, env="RIS_BLS_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(None, env="RIS_ALPHA_VANTAGE_API_KEY")
    
    # Security
    reviewer_api_key: Optional[str] = Field(None, env="RIS_REVIEWER_API_KEY")
    allowed_origins: str = Field("http://localhost:3000", env="RIS_ALLOWED_ORIGINS")
    
    # Auth (for future use)
    auth_issuer: Optional[str] = Field(None, env="RIS_AUTH_ISSUER")
    auth_audience: Optional[str] = Field(None, env="RIS_AUTH_AUDIENCE")
    auth_jwks: Optional[str] = Field(None, env="RIS_AUTH_JWKS")
    
    # Worker settings
    worker_role: str = Field("ingestion", env="WORKER_ROLE")
    
    @property 
    def database_url(self) -> str:
        """Get database URL for SQLAlchemy."""
        return self.postgres_dsn
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment in ("production", "prod")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment in ("development", "dev", "local")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
