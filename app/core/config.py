from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    environment: str = Field("local", env="RRIO_ENV")
    redis_url: str = Field("redis://localhost:6379/0", env="RRIO_REDIS_URL")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    cache_ttl_seconds: int = Field(900, env="RRIO_CACHE_TTL")
    fred_api_key: Optional[str] = Field(None, env="RRIO_FRED_API_KEY")
    reviewer_api_key: Optional[str] = Field(None, env="RRIO_REVIEWER_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
