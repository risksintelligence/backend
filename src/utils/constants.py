"""
Application constants for RiskX platform.
Defines all constant values used across the application for consistency and maintainability.
"""

from enum import Enum
from typing import Dict, List, Tuple


# Application Information
APP_NAME = "RiskX"
APP_DESCRIPTION = "AI Risk Intelligence Observatory"
APP_VERSION = "1.0.0"
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Professional Color Scheme (Strict Requirement)
class Colors:
    """Professional color palette for RiskX platform."""
    
    # Primary colors (MANDATORY)
    PRIMARY_NAVY = "#1e3a8a"        # Primary navigation, headers
    CHARCOAL_GRAY = "#374151"       # Text, secondary elements  
    PURE_WHITE = "#ffffff"          # Backgrounds, contrast
    
    # Supporting colors
    LIGHT_GRAY = "#f3f4f6"          # Subtle backgrounds
    BORDER_GRAY = "#d1d5db"         # Borders, dividers
    
    # Status colors (only for system states)
    SUCCESS_GREEN = "#059669"       # Success states only
    WARNING_AMBER = "#d97706"       # Warning states only
    DANGER_RED = "#dc2626"          # Error states only
    INFO_BLUE = "#2563eb"           # Information states only


# Risk Assessment Constants
class RiskLevels:
    """Risk level thresholds and classifications."""
    
    LOW_THRESHOLD = 25.0
    MODERATE_THRESHOLD = 50.0
    HIGH_THRESHOLD = 75.0
    CRITICAL_THRESHOLD = 90.0
    
    LEVEL_NAMES = {
        "low": "Low Risk",
        "moderate": "Moderate Risk", 
        "high": "High Risk",
        "critical": "Critical Risk"
    }
    
    LEVEL_COLORS = {
        "low": Colors.SUCCESS_GREEN,
        "moderate": Colors.WARNING_AMBER,
        "high": Colors.DANGER_RED,
        "critical": Colors.DANGER_RED
    }


class RiskCategories:
    """Risk category definitions and weights."""
    
    ECONOMIC = "economic"
    FINANCIAL = "financial"
    SUPPLY_CHAIN = "supply_chain"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    CYBER = "cyber"
    ENVIRONMENTAL = "environmental"
    
    ALL_CATEGORIES = [
        ECONOMIC, FINANCIAL, SUPPLY_CHAIN, OPERATIONAL,
        REGULATORY, CYBER, ENVIRONMENTAL
    ]
    
    DEFAULT_WEIGHTS = {
        ECONOMIC: 0.25,
        FINANCIAL: 0.25,
        SUPPLY_CHAIN: 0.20,
        OPERATIONAL: 0.15,
        REGULATORY: 0.10,
        CYBER: 0.05,
        ENVIRONMENTAL: 0.05
    }
    
    CATEGORY_DESCRIPTIONS = {
        ECONOMIC: "Macroeconomic indicators and trends",
        FINANCIAL: "Financial market conditions and stability",
        SUPPLY_CHAIN: "Supply chain disruptions and dependencies",
        OPERATIONAL: "Operational efficiency and capacity",
        REGULATORY: "Regulatory changes and compliance",
        CYBER: "Cybersecurity threats and vulnerabilities",
        ENVIRONMENTAL: "Environmental and climate-related risks"
    }


# Data Source Constants
class DataSources:
    """External data source configurations."""
    
    # Federal Reserve Economic Data (FRED)
    FRED_BASE_URL = "https://api.stlouisfed.org/fred"
    FRED_SERIES_LIMIT = 1000
    FRED_RATE_LIMIT = 120  # requests per minute
    
    # Bureau of Economic Analysis (BEA)
    BEA_BASE_URL = "https://apps.bea.gov/api/data"
    BEA_RATE_LIMIT = 100
    
    # Bureau of Labor Statistics (BLS)
    BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    BLS_RATE_LIMIT = 25
    
    # U.S. Census Bureau
    CENSUS_BASE_URL = "https://api.census.gov/data"
    CENSUS_RATE_LIMIT = 500
    
    # Federal Deposit Insurance Corporation (FDIC)
    FDIC_BASE_URL = "https://banks.data.fdic.gov/api"
    FDIC_RATE_LIMIT = 100
    
    # National Oceanic and Atmospheric Administration (NOAA)
    NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
    NOAA_RATE_LIMIT = 1000
    
    # Cybersecurity and Infrastructure Security Agency (CISA)
    CISA_BASE_URL = "https://www.cisa.gov/uscert/ncas/current-activity.xml"
    CISA_RATE_LIMIT = 60
    
    # Default timeout for all API calls (seconds)
    DEFAULT_TIMEOUT = 30
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    RETRY_BACKOFF = 2.0


# Economic Indicators
class EconomicIndicators:
    """Key economic indicators and their FRED series IDs."""
    
    # Labor Market
    UNEMPLOYMENT_RATE = "UNRATE"
    NONFARM_PAYROLLS = "PAYEMS"
    LABOR_FORCE_PARTICIPATION = "CIVPART"
    INITIAL_CLAIMS = "ICSA"
    
    # Inflation and Prices
    CPI_ALL_ITEMS = "CPIAUCSL"
    CORE_CPI = "CPILFESL"
    PCE_PRICE_INDEX = "PCEPI"
    CORE_PCE = "PCEPILFE"
    
    # Economic Growth
    GDP_REAL = "GDPC1"
    GDP_DEFLATOR = "GDPDEF"
    INDUSTRIAL_PRODUCTION = "INDPRO"
    CAPACITY_UTILIZATION = "TCU"
    
    # Interest Rates
    FEDERAL_FUNDS_RATE = "FEDFUNDS"
    TEN_YEAR_TREASURY = "GS10"
    THREE_MONTH_TREASURY = "TB3MS"
    YIELD_CURVE_SPREAD = "T10Y3M"
    
    # Financial Stress
    CREDIT_SPREAD = "BAA10Y"
    VIX_INDEX = "VIXCLS"
    HIGH_YIELD_SPREAD = "BAMLH0A0HYM2"
    TERM_PREMIUM = "THREEFYTP10"
    
    # Money and Banking
    M2_MONEY_SUPPLY = "M2SL"
    BANK_CREDIT = "TOTBKCR"
    COMMERCIAL_LOANS = "BUSLOANS"
    CONSUMER_CREDIT = "TOTALSL"
    
    # Trade and International
    TRADE_BALANCE = "BOPGSTB"
    EXPORTS = "EXPGS"
    IMPORTS = "IMPGS"
    DOLLAR_INDEX = "DTWEXBGS"


# Cache Configuration
class CacheConfig:
    """Cache configuration constants."""
    
    # Cache TTL (time to live) in seconds
    DEFAULT_TTL = 3600  # 1 hour
    SHORT_TTL = 300     # 5 minutes
    MEDIUM_TTL = 1800   # 30 minutes
    LONG_TTL = 86400    # 24 hours
    
    # Cache key prefixes
    RISK_SCORE_PREFIX = "risk_score"
    ECONOMIC_DATA_PREFIX = "economic_data"
    PREDICTION_PREFIX = "prediction"
    SIMULATION_PREFIX = "simulation"
    
    # Cache size limits
    MAX_CACHE_SIZE_MB = 512
    MAX_CACHE_ENTRIES = 10000
    
    # Fallback cache configuration
    FALLBACK_DATA_RETENTION_DAYS = 30
    FALLBACK_MAX_SIZE_MB = 1024


# Model Configuration
class ModelConfig:
    """Machine learning model configuration constants."""
    
    # Model types
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    LINEAR_REGRESSION = "linear_regression"
    
    # Default model parameters
    DEFAULT_FEATURES = [
        "unemployment_rate", "inflation_rate", "gdp_growth",
        "credit_spread", "yield_curve", "volatility_index"
    ]
    
    # Model performance thresholds
    MIN_ACCURACY = 0.75
    MIN_PRECISION = 0.70
    MIN_RECALL = 0.70
    MIN_F1_SCORE = 0.70
    
    # Prediction confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.50
    
    # Model retraining intervals
    RETRAIN_INTERVAL_DAYS = 7
    PERFORMANCE_CHECK_INTERVAL_DAYS = 1


# Simulation Parameters
class SimulationConfig:
    """Simulation and stress testing configuration."""
    
    # Scenario types
    ECONOMIC_SHOCK = "economic_shock"
    SUPPLY_DISRUPTION = "supply_disruption" 
    FINANCIAL_CRISIS = "financial_crisis"
    POLICY_CHANGE = "policy_change"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    
    # Stress test severity levels
    MILD_STRESS = {
        "unemployment_shock": 0.02,  # 2 percentage points
        "interest_rate_shock": 0.01,  # 1 percentage point
        "equity_decline": -0.10       # 10% decline
    }
    
    MODERATE_STRESS = {
        "unemployment_shock": 0.05,
        "interest_rate_shock": 0.025,
        "equity_decline": -0.25
    }
    
    SEVERE_STRESS = {
        "unemployment_shock": 0.10,
        "interest_rate_shock": 0.05,
        "equity_decline": -0.40
    }
    
    EXTREME_STRESS = {
        "unemployment_shock": 0.15,
        "interest_rate_shock": 0.075,
        "equity_decline": -0.60
    }
    
    # Monte Carlo simulation parameters
    DEFAULT_MC_RUNS = 1000
    MAX_MC_RUNS = 10000
    MIN_MC_RUNS = 100
    
    # Simulation time horizons
    SHORT_TERM_DAYS = 30
    MEDIUM_TERM_DAYS = 90
    LONG_TERM_DAYS = 365


# API Configuration
class APIConfig:
    """API configuration constants."""
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 100  # requests per minute
    AUTHENTICATED_RATE_LIMIT = 1000
    ADMIN_RATE_LIMIT = 5000
    
    # Request size limits
    MAX_REQUEST_SIZE_MB = 10
    MAX_JSON_PAYLOAD_SIZE = 1024 * 1024  # 1MB
    
    # Pagination
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Timeout settings
    REQUEST_TIMEOUT = 30
    LONG_RUNNING_TIMEOUT = 300  # 5 minutes
    
    # Response format
    DEFAULT_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d"


# Logging Configuration
class LogConfig:
    """Logging configuration constants."""
    
    # Log levels
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    # Log formats
    SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    JSON_FORMAT = "json"
    
    # Log rotation
    MAX_LOG_SIZE_MB = 50
    BACKUP_COUNT = 10
    
    # Log retention
    LOG_RETENTION_DAYS = 30
    SECURITY_LOG_RETENTION_DAYS = 90


# Security Configuration  
class SecurityConfig:
    """Security configuration constants."""
    
    # Password requirements
    MIN_PASSWORD_LENGTH = 12
    PASSWORD_COMPLEXITY_REQUIREMENTS = {
        "uppercase": True,
        "lowercase": True,
        "digits": True,
        "special_chars": True
    }
    
    # Session management
    SESSION_TIMEOUT_MINUTES = 30
    REFRESH_TOKEN_DAYS = 7
    
    # Rate limiting for security
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_LOCKOUT_MINUTES = 15
    
    # Input validation
    MAX_INPUT_LENGTH = 10000
    ALLOWED_FILE_EXTENSIONS = {'.json', '.csv', '.txt', '.pdf', '.png', '.jpg'}
    MAX_FILE_SIZE_MB = 10
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }


# File and Storage Configuration
class StorageConfig:
    """File and storage configuration constants."""
    
    # File paths
    DATA_DIR = "data"
    CACHE_DIR = "data/cache"
    LOGS_DIR = "logs"
    MODELS_DIR = "data/models"
    EXPORTS_DIR = "data/exports"
    
    # File naming conventions
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    BACKUP_PREFIX = "backup_"
    EXPORT_PREFIX = "export_"
    
    # File size limits
    MAX_EXPORT_SIZE_MB = 100
    MAX_BACKUP_SIZE_MB = 1000
    
    # Cleanup settings
    TEMP_FILE_RETENTION_HOURS = 24
    EXPORT_FILE_RETENTION_DAYS = 7
    BACKUP_RETENTION_DAYS = 30


# Network and HTTP Configuration
class NetworkConfig:
    """Network and HTTP configuration constants."""
    
    # HTTP status codes (commonly used)
    HTTP_OK = 200
    HTTP_CREATED = 201
    HTTP_BAD_REQUEST = 400
    HTTP_UNAUTHORIZED = 401
    HTTP_FORBIDDEN = 403
    HTTP_NOT_FOUND = 404
    HTTP_TOO_MANY_REQUESTS = 429
    HTTP_INTERNAL_ERROR = 500
    HTTP_SERVICE_UNAVAILABLE = 503
    
    # Timeouts
    CONNECT_TIMEOUT = 10
    READ_TIMEOUT = 30
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_STATUSES = [502, 503, 504]
    
    # User agent
    USER_AGENT = f"{APP_NAME}/{APP_VERSION}"


# Business Logic Constants
class BusinessRules:
    """Business logic constants and rules."""
    
    # Risk score calculation
    RISK_SCORE_MIN = 0.0
    RISK_SCORE_MAX = 100.0
    CONFIDENCE_MIN = 0.0
    CONFIDENCE_MAX = 1.0
    
    # Data quality thresholds
    MIN_DATA_QUALITY_SCORE = 0.70
    MIN_DATA_FRESHNESS_HOURS = 24
    MAX_DATA_AGE_DAYS = 30
    
    # Prediction horizons
    MIN_PREDICTION_DAYS = 1
    MAX_PREDICTION_DAYS = 365
    DEFAULT_PREDICTION_DAYS = 30
    
    # Model performance requirements
    MODEL_DRIFT_THRESHOLD = 0.10
    BIAS_THRESHOLD = 0.05
    FAIRNESS_THRESHOLD = 0.95
    
    # Simulation limits
    MAX_SCENARIO_DURATION_DAYS = 1095  # 3 years
    MAX_SIMULATION_PARAMETERS = 50
    MIN_MONTE_CARLO_CONFIDENCE = 0.90


# Error Codes
class ErrorCodes:
    """Standard error codes for the application."""
    
    # Authentication errors (1000-1099)
    AUTH_INVALID_CREDENTIALS = "AUTH_1001"
    AUTH_TOKEN_EXPIRED = "AUTH_1002"
    AUTH_TOKEN_INVALID = "AUTH_1003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_1004"
    
    # Validation errors (1100-1199)
    VALIDATION_REQUIRED_FIELD = "VALIDATION_1101"
    VALIDATION_INVALID_FORMAT = "VALIDATION_1102"
    VALIDATION_OUT_OF_RANGE = "VALIDATION_1103"
    VALIDATION_INVALID_TYPE = "VALIDATION_1104"
    
    # Data errors (1200-1299)
    DATA_NOT_FOUND = "DATA_1201"
    DATA_QUALITY_INSUFFICIENT = "DATA_1202"
    DATA_SOURCE_UNAVAILABLE = "DATA_1203"
    DATA_PROCESSING_FAILED = "DATA_1204"
    
    # Model errors (1300-1399)
    MODEL_NOT_FOUND = "MODEL_1301"
    MODEL_PREDICTION_FAILED = "MODEL_1302"
    MODEL_INSUFFICIENT_DATA = "MODEL_1303"
    MODEL_PERFORMANCE_DEGRADED = "MODEL_1304"
    
    # System errors (1400-1499)
    SYSTEM_UNAVAILABLE = "SYSTEM_1401"
    SYSTEM_OVERLOADED = "SYSTEM_1402"
    SYSTEM_CONFIGURATION_ERROR = "SYSTEM_1403"
    SYSTEM_INTERNAL_ERROR = "SYSTEM_1404"


# Regular Expressions
class RegexPatterns:
    """Common regular expression patterns."""
    
    EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    URL = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?$'
    PHONE = r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
    UUID = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    ALPHANUMERIC = r'^[a-zA-Z0-9]+$'
    NUMERIC = r'^-?\d*\.?\d+$'
    DATE_ISO = r'^\d{4}-\d{2}-\d{2}$'
    DATETIME_ISO = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?$'


# Feature Flags
class FeatureFlags:
    """Feature flag constants for gradual rollouts."""
    
    ENABLE_ADVANCED_ANALYTICS = True
    ENABLE_MONTE_CARLO_SIMULATION = True
    ENABLE_REAL_TIME_UPDATES = True
    ENABLE_DETAILED_LOGGING = True
    ENABLE_PERFORMANCE_MONITORING = True
    ENABLE_BIAS_DETECTION = True
    ENABLE_EXPLAINABLE_AI = True
    ENABLE_POLICY_SIMULATION = True