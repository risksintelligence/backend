import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    environment: str = Field(..., alias="ris_env")
    
    # Database
    postgres_dsn: str = Field(..., alias="ris_postgres_dsn")
    
    # Redis
    redis_url: Optional[str] = Field(None, alias="ris_redis_url")
    
    # Paths
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models")
    
    # Cache
    cache_ttl_seconds: int = Field(900, alias="ris_cache_ttl")
    
    # Data Provider API Keys
    fred_api_key: Optional[str] = Field(None, alias="ris_fred_api_key")
    eia_api_key: Optional[str] = Field(None, alias="ris_eia_api_key")
    census_api_key: Optional[str] = Field(None, alias="ris_census_api_key")
    bea_api_key: Optional[str] = Field(None, alias="ris_bea_api_key")
    bls_api_key: Optional[str] = Field(None, alias="ris_bls_api_key")
    alpha_vantage_api_key: Optional[str] = Field(None, alias="ris_alpha_vantage_api_key")
    wto_api_key: Optional[str] = Field(None, alias="ris_wto_api_key")
    
    # ACLED (Armed Conflict Location & Event Data) OAuth Authentication
    acled_email: Optional[str] = Field(None, alias="ris_acled_email")
    acled_password: Optional[str] = Field(None, alias="ris_acled_password")
    
    # MarineTraffic API Key
    marinetraffic_api_key: Optional[str] = Field(None, alias="ris_marinetraffic_api_key")
    
    # Free API replacements for S&P Global
    openroute_api_key: Optional[str] = Field(None, alias="ris_openroute_api_key")
    
    # Security
    reviewer_api_key: Optional[str] = Field(None, alias="ris_reviewer_api_key")
    allowed_origins: str = Field("https://frontend-1-tzlw.onrender.com,https://backend-1-s84g.onrender.com", alias="ris_allowed_origins")
    jwt_secret: Optional[str] = Field(None, alias="ris_jwt_secret")
    
    # Auth (for future use)
    auth_issuer: Optional[str] = Field(None, alias="ris_auth_issuer")
    auth_audience: Optional[str] = Field(None, alias="ris_auth_audience")
    auth_jwks: Optional[str] = Field(None, alias="ris_auth_jwks")
    
    # Monitoring & Error Tracking
    sentry_dsn: Optional[str] = Field(None, alias="ris_sentry_dsn")
    log_level: str = Field("INFO", alias="ris_log_level")
    enable_metrics: bool = Field(True, alias="ris_enable_metrics")
    cors_origins: Optional[str] = Field(None, alias="ris_cors_origins")
    
    # Worker settings
    worker_role: str = Field("ingestion", env="worker_role")
    
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
                "fred_api_key": self.fred_api_key,
                "reviewer_api_key": self.reviewer_api_key,
                "jwt_secret": self.jwt_secret,
            }
            
            # Redis is optional for development
            if self.redis_url:
                required_fields["redis_url"] = self.redis_url
            
            missing_fields = [field for field, value in required_fields.items() if not value]
            
            if missing_fields:
                raise RuntimeError(
                    f"Production environment missing required configuration: {', '.join(missing_fields)}"
                )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production_config()
    return settings


# Global settings instance
settings = get_settings()
