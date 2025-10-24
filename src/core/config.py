from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Production settings for RiskX Platform."""
    
    # Application
    app_name: str = "RiskX Risk Intelligence Platform"
    app_version: str = "1.0.0"
    environment: str = "production"
    
    # API Configuration
    api_v1_str: str = "/api/v1"
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "")
    
    # Cache
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # External APIs
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    bea_api_key: str = os.getenv("BEA_API_KEY", "")
    bls_api_key: str = os.getenv("BLS_API_KEY", "")
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra environment variables

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()