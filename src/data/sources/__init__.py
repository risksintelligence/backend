"""
Data Sources Module - All External API Integrations
"""

# Economic Data Sources
from .fred import get_key_indicators as get_fred_data, health_check as fred_health_check
from .bea import get_economic_accounts as get_bea_data, health_check as bea_health_check
from .bls import get_labor_statistics as get_bls_data, health_check as bls_health_check
from .census import get_population_data, get_household_income, health_check as census_health_check

# Risk Intelligence Data Sources
from .cisa import get_cybersecurity_threats as get_cisa_data, health_check as cisa_health_check
from .noaa import get_environmental_risks as get_noaa_data, health_check as noaa_health_check
from .usgs import get_geological_hazards as get_usgs_data, health_check as usgs_health_check
from .supply_chain import get_supply_chain_risks as get_supply_chain_data, health_check as supply_chain_health_check

__all__ = [
    # Economic Data
    "get_fred_data",
    "get_bea_data", 
    "get_bls_data",
    "get_population_data",
    "get_household_income",
    
    # Risk Intelligence Data
    "get_cisa_data",
    "get_noaa_data",
    "get_usgs_data",
    "get_supply_chain_data",
    
    # Health Checks
    "fred_health_check",
    "bea_health_check",
    "bls_health_check", 
    "census_health_check",
    "cisa_health_check",
    "noaa_health_check",
    "usgs_health_check",
    "supply_chain_health_check"
]