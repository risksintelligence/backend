"""Configuration management for RIS backend."""
import os
from typing import Optional

class Settings:
    """Application settings."""
    
    def __init__(self):
        self.RIS_POSTGRES_DSN: str = os.getenv("RIS_POSTGRES_DSN", "postgresql://ris_user:ris_password@ris-postgres:5432/ris_production")
        self.RIS_REDIS_URL: str = os.getenv("RIS_REDIS_URL", "redis://ris-redis:6379/0")
        self.RIS_ALLOWED_ORIGINS: str = os.getenv("RIS_ALLOWED_ORIGINS", "https://frontend-1-wvu7.onrender.com")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
