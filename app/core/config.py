import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    environment: str = Field(default_factory=lambda: os.getenv("RIS_ENV", "production"), env="RIS_ENV")
    
    # Database
    postgres_dsn: str = Field(default_factory=lambda: os.getenv("RIS_POSTGRES_DSN"), env="RIS_POSTGRES_DSN")
    
    # Redis
    redis_url: str = Field(default_factory=lambda: os.getenv("RIS_REDIS_URL"), env="RIS_REDIS_URL")
    
    @validator('postgres_dsn', 'redis_url')
    def validate_required_fields(cls, v, field):
        if not v:
            raise ValueError(f'{field.name} is required and must be set via environment variable')
        return v
    
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
    allowed_origins: str = Field("https://frontend-9t5o.onrender.com,https://backend-9t5o.onrender.com", env="RIS_ALLOWED_ORIGINS")
    jwt_secret: Optional[str] = Field(None, env="RIS_JWT_SECRET")
    
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
    
    def validate_production_config(self) -> None:
        """Validate that all required production settings are configured."""
        if self.is_production:
            required_fields = {
                "postgres_dsn": self.postgres_dsn,
                "redis_url": self.redis_url,
                "fred_api_key": self.fred_api_key,
                "reviewer_api_key": self.reviewer_api_key,
                "jwt_secret": self.jwt_secret,
            }
            
            missing_fields = [field for field, value in required_fields.items() if not value]
            
            if missing_fields:
                raise RuntimeError(
                    f"Production environment missing required configuration: {', '.join(missing_fields)}"
                )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production_config()
    return settings
