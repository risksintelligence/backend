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
    
    # Database (will be configured in next phase)
    database_url: str = ""
    
    # Cache (will be configured in next phase)  
    redis_url: str = ""
    
    # External APIs (will be configured in next phase)
    fred_api_key: str = ""
    bea_api_key: str = ""
    bls_api_key: str = ""
    census_api_key: str = ""
    
    # Security
    secret_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()