"""Production logging configuration."""
import logging
import logging.config
import sys
from pathlib import Path
from app.core.config import get_settings

settings = get_settings()

def setup_logging():
    """Configure structured logging for production."""
    
    log_level = "INFO" if settings.is_production else "DEBUG"
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "json": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter" if settings.is_production else "logging.Formatter"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if settings.is_production else "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "WARNING",
                "formatter": "json",
                "filename": "logs/app.log",
                "mode": "a"
            } if settings.is_production else None
        },
        "loggers": {
            "": {  # root logger
                "level": log_level,
                "handlers": ["console"] + (["file"] if settings.is_production else []),
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO", 
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "WARNING" if settings.is_production else "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Remove None handlers
    logging_config["handlers"] = {k: v for k, v in logging_config["handlers"].items() if v is not None}
    
    # Create logs directory if needed
    if settings.is_production:
        Path("logs").mkdir(exist_ok=True)
    
    logging.config.dictConfig(logging_config)
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for {settings.environment} environment")