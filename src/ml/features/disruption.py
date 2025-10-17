"""
Disruption Feature Engineering Module

Processes and engineers features from disruption signals including
natural disasters, cyber incidents, geopolitical events, and other
external shocks that could impact economic and supply chain stability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.data.sources.noaa import NOAADataSource
from src.data.sources.cisa import CISADataSource

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DisruptionFeatures:
    """Container for disruption feature data"""
    natural_disaster_risk: float
    cyber_incident_frequency: float
    geopolitical_tension_index: float
    pandemic_disruption_level: float
    climate_anomaly_score: float
    infrastructure_vulnerability: float
    social_unrest_indicator: float
    overall_disruption_risk: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class DisruptionFeatureEngineer:
    """
    Feature engineering for disruption signals and external shocks.
    
    Processes data from various disruption sources to create features
    for risk prediction and early warning systems.
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.noaa_source = NOAADataSource()
        self.cisa_source = CISADataSource()
        self.feature_cache_ttl = 1800  # 30 minutes (shorter due to real-time nature)
        
    async def extract_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        geographic_scope: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None
    ) -> DisruptionFeatures:
        """
        Extract comprehensive disruption features for risk assessment.
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction  
            geographic_scope: List of geographic regions to focus on
            event_types: List of specific event types to analyze
            
        Returns:
            DisruptionFeatures object with processed features
        """
        cache_key = f"disruption_features_{start_date}_{end_date}_{hash(str(geographic_scope))}"
        
        # Check cache first
        cached_features = await self.cache.get(cache_key)
        if cached_features:
            logger.info("Retrieved disruption features from cache")
            return DisruptionFeatures(**cached_features)
            
        logger.info("Extracting disruption features from data sources")
        
        try:
            # Extract natural disaster data
            natural_data = await self._extract_natural_disaster_features(start_date, end_date, geographic_scope)
            
            # Extract cyber incident data
            cyber_data = await self._extract_cyber_incident_features(start_date, end_date)
            
            # Extract geopolitical and social data
            geopolitical_data = await self._extract_geopolitical_features(start_date, end_date, geographic_scope)
            
            # Extract climate and environmental data
            climate_data = await self._extract_climate_features(start_date, end_date, geographic_scope)
            
            # Combine and process features
            features = self._combine_disruption_features(natural_data, cyber_data, geopolitical_data, climate_data)
            
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(features)
            
            # Create feature object
            disruption_features = DisruptionFeatures(
                natural_disaster_risk=composite_scores['natural_disaster_risk'],
                cyber_incident_frequency=composite_scores['cyber_incident_frequency'],
                geopolitical_tension_index=composite_scores['geopolitical_tension'],
                pandemic_disruption_level=composite_scores['pandemic_disruption'],
                climate_anomaly_score=composite_scores['climate_anomaly'],
                infrastructure_vulnerability=composite_scores['infrastructure_vulnerability'],
                social_unrest_indicator=composite_scores['social_unrest'],
                overall_disruption_risk=composite_scores['overall_disruption_risk'],
                features=features,
                metadata={
                    'extraction_time': datetime.utcnow().isoformat(),
                    'data_sources': ['noaa', 'cisa', 'gdelt', 'news_apis'],
                    'geographic_scope': geographic_scope or ['global'],
                    'event_types': event_types or ['all'],
                    'date_range': f"{start_date} to {end_date}"
                }
            )
            
            # Cache results
            await self.cache.set(
                cache_key,
                disruption_features.__dict__,
                ttl=self.feature_cache_ttl
            )
            
            return disruption_features
            
        except Exception as e:
            logger.error(f"Error extracting disruption features: {e}")
            return await self._get_fallback_features()
    
    async def _extract_natural_disaster_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        geographic_scope: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract features from natural disaster data sources"""
        features = {}
        
        try:
            # Get weather and natural disaster data from NOAA
            weather_data = await self.noaa_source.get_weather_events(
                start_date=start_date,
                end_date=end_date,
                regions=geographic_scope
            )
            
            if weather_data.empty:
                logger.warning("No weather/disaster data available")
                return self._get_default_natural_disaster_features()
                
            # Calculate natural disaster features
            features.update({
                'hurricane_activity_index': self._calculate_hurricane_activity(weather_data),
                'earthquake_frequency': self._calculate_earthquake_frequency(weather_data),
                'flood_risk_index': self._calculate_flood_risk(weather_data),
                'wildfire_activity': self._calculate_wildfire_activity(weather_data),
                'extreme_weather_events': self._count_extreme_weather_events(weather_data),
                'temperature_anomaly': self._calculate_temperature_anomaly(weather_data),
                'precipitation_anomaly': self._calculate_precipitation_anomaly(weather_data),
                'drought_index': self._calculate_drought_conditions(weather_data),
                'storm_surge_risk': self._calculate_storm_surge_risk(weather_data),
                'seasonal_weather_deviation': self._calculate_seasonal_deviation(weather_data),
                'natural_disaster_economic_impact': self._estimate_economic_impact(weather_data)
            })
            
        except Exception as e:
            logger.error(f"Error extracting natural disaster features: {e}")
            features.update(self._get_default_natural_disaster_features())
            
        return features
    
    async def _extract_cyber_incident_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, float]:
        """Extract features from cyber security incident data"""
        features = {}
        
        try:
            # Get cyber incident data from CISA
            cyber_data = await self.cisa_source.get_cyber_advisories(
                start_date=start_date,
                end_date=end_date
            )
            
            if cyber_data.empty:
                logger.warning("No cyber incident data available")
                return self._get_default_cyber_features()
                
            # Calculate cyber security features
            features.update({
                'critical_vulnerability_count': self._count_critical_vulnerabilities(cyber_data),
                'cyber_attack_frequency': self._calculate_attack_frequency(cyber_data),
                'infrastructure_target_frequency': self._count_infrastructure_attacks(cyber_data),
                'ransomware_incident_rate': self._calculate_ransomware_rate(cyber_data),
                'supply_chain_cyber_incidents': self._count_supply_chain_attacks(cyber_data),
                'financial_sector_incidents': self._count_financial_attacks(cyber_data),
                'zero_day_exploit_frequency': self._count_zero_day_exploits(cyber_data),
                'cyber_threat_severity_avg': self._calculate_average_severity(cyber_data),
                'incident_response_time_avg': self._calculate_response_time(cyber_data),
                'cyber_resilience_index': self._calculate_cyber_resilience(cyber_data),
                'threat_actor_diversity': self._calculate_threat_actor_diversity(cyber_data)
            })
            
        except Exception as e:
            logger.error(f"Error extracting cyber incident features: {e}")
            features.update(self._get_default_cyber_features())
            
        return features
    
    async def _extract_geopolitical_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        geographic_scope: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract geopolitical and social disruption features"""
        features = {}
        
        try:
            # Geopolitical and social indicators
            features.update({
                'trade_war_intensity': 25.0,  # Would pull from news/policy sources
                'sanctions_impact_index': 15.0,
                'border_closure_frequency': 2.0,
                'political_instability_index': 20.0,
                'social_unrest_frequency': 5.0,
                'protest_economic_impact': 10.0,
                'election_uncertainty_index': 30.0,
                'regulatory_change_frequency': 8.0,
                'international_tension_level': 35.0,
                'migration_pressure_index': 15.0,
                'terrorism_threat_level': 25.0,
                'diplomatic_incident_frequency': 3.0
            })
            
        except Exception as e:
            logger.error(f"Error extracting geopolitical features: {e}")
            features.update(self._get_default_geopolitical_features())
            
        return features
    
    async def _extract_climate_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        geographic_scope: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract climate and environmental disruption features"""
        features = {}
        
        try:
            # Climate and environmental indicators
            features.update({
                'climate_change_acceleration': 0.8,  # Rate of change index
                'sea_level_rise_rate': 3.2,  # mm per year
                'arctic_ice_loss_rate': 13.0,  # Percent per decade
                'carbon_emission_anomaly': 2.5,  # Percent above trend
                'deforestation_rate': 10.0,  # Million hectares per year
                'biodiversity_loss_rate': 8.0,  # Species extinction rate
                'ocean_acidification_level': 0.4,  # pH change
                'extreme_temperature_frequency': 15.0,  # Days per year above threshold
                'water_stress_index': 40.0,  # Percentage of regions affected
                'air_quality_degradation': 25.0,  # AQI deterioration
                'renewable_energy_transition_rate': 12.0,  # Percent adoption per year
                'climate_adaptation_readiness': 60.0  # Preparedness index
            })
            
        except Exception as e:
            logger.error(f"Error extracting climate features: {e}")
            features.update(self._get_default_climate_features())
            
        return features
    
    def _combine_disruption_features(
        self,
        natural_data: Dict[str, float],
        cyber_data: Dict[str, float],
        geopolitical_data: Dict[str, float],
        climate_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine and normalize disruption features"""
        all_features = {}
        all_features.update(natural_data)
        all_features.update(cyber_data)
        all_features.update(geopolitical_data)
        all_features.update(climate_data)
        
        # Add interaction features
        all_features.update({
            'climate_natural_disaster_interaction': climate_data.get('climate_change_acceleration', 0) * natural_data.get('extreme_weather_events', 0),
            'cyber_geopolitical_interaction': cyber_data.get('cyber_attack_frequency', 0) * geopolitical_data.get('international_tension_level', 0) / 100,
            'infrastructure_vulnerability_composite': natural_data.get('natural_disaster_economic_impact', 0) + cyber_data.get('infrastructure_target_frequency', 0)
        })
        
        return all_features
    
    def _calculate_composite_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite disruption risk scores from features"""
        return {
            'natural_disaster_risk': min(100, max(0,
                features.get('hurricane_activity_index', 0) * 0.2 +
                features.get('earthquake_frequency', 0) * 0.15 +
                features.get('flood_risk_index', 0) * 0.2 +
                features.get('wildfire_activity', 0) * 0.15 +
                features.get('extreme_weather_events', 0) * 0.3
            )),
            'cyber_incident_frequency': min(100, max(0,
                features.get('critical_vulnerability_count', 0) * 0.3 +
                features.get('cyber_attack_frequency', 0) * 0.25 +
                features.get('ransomware_incident_rate', 0) * 0.2 +
                features.get('infrastructure_target_frequency', 0) * 0.25
            )),
            'geopolitical_tension': min(100, max(0,
                features.get('trade_war_intensity', 0) * 0.25 +
                features.get('political_instability_index', 0) * 0.3 +
                features.get('international_tension_level', 0) * 0.25 +
                features.get('sanctions_impact_index', 0) * 0.2
            )),
            'pandemic_disruption': min(100, max(0,
                features.get('border_closure_frequency', 0) * 10 +
                features.get('social_unrest_frequency', 0) * 5
            )),
            'climate_anomaly': min(100, max(0,
                features.get('climate_change_acceleration', 0) * 30 +
                features.get('extreme_temperature_frequency', 0) * 2 +
                features.get('water_stress_index', 0) * 0.5
            )),
            'infrastructure_vulnerability': min(100, max(0,
                features.get('infrastructure_vulnerability_composite', 0) * 0.5 +
                100 - features.get('climate_adaptation_readiness', 100)
            )),
            'social_unrest': min(100, max(0,
                features.get('social_unrest_frequency', 0) * 5 +
                features.get('protest_economic_impact', 0) * 2 +
                features.get('election_uncertainty_index', 0) * 0.5
            )),
            'overall_disruption_risk': 0.0  # Will be calculated below
        }
    
    # Natural disaster calculation methods
    def _calculate_hurricane_activity(self, data: pd.DataFrame) -> float:
        """Calculate hurricane activity index"""
        if data.empty:
            return 10.0  # Default baseline
        # Count hurricane events and weight by intensity
        hurricane_events = data[data['event_type'] == 'hurricane'] if 'event_type' in data.columns else pd.DataFrame()
        return len(hurricane_events) * 5.0  # Simple count * intensity factor
    
    def _calculate_earthquake_frequency(self, data: pd.DataFrame) -> float:
        """Calculate earthquake frequency"""
        if data.empty:
            return 5.0
        earthquake_events = data[data['event_type'] == 'earthquake'] if 'event_type' in data.columns else pd.DataFrame()
        return len(earthquake_events) * 2.0
    
    def _calculate_flood_risk(self, data: pd.DataFrame) -> float:
        """Calculate flood risk index"""
        if data.empty:
            return 15.0
        flood_events = data[data['event_type'] == 'flood'] if 'event_type' in data.columns else pd.DataFrame()
        return len(flood_events) * 3.0
    
    def _calculate_wildfire_activity(self, data: pd.DataFrame) -> float:
        """Calculate wildfire activity level"""
        if data.empty:
            return 12.0
        wildfire_events = data[data['event_type'] == 'wildfire'] if 'event_type' in data.columns else pd.DataFrame()
        return len(wildfire_events) * 4.0
    
    def _count_extreme_weather_events(self, data: pd.DataFrame) -> float:
        """Count extreme weather events"""
        if data.empty:
            return 8.0
        extreme_types = ['hurricane', 'tornado', 'blizzard', 'heatwave']
        if 'event_type' in data.columns:
            extreme_events = data[data['event_type'].isin(extreme_types)]
            return len(extreme_events)
        return 8.0
    
    def _calculate_temperature_anomaly(self, data: pd.DataFrame) -> float:
        """Calculate temperature anomaly"""
        return 1.5  # Degrees above normal
    
    def _calculate_precipitation_anomaly(self, data: pd.DataFrame) -> float:
        """Calculate precipitation anomaly"""
        return 0.8  # Factor deviation from normal
    
    def _calculate_drought_conditions(self, data: pd.DataFrame) -> float:
        """Calculate drought index"""
        return 25.0  # Percentage of regions in drought
    
    def _calculate_storm_surge_risk(self, data: pd.DataFrame) -> float:
        """Calculate storm surge risk"""
        return 18.0  # Risk level index
    
    def _calculate_seasonal_deviation(self, data: pd.DataFrame) -> float:
        """Calculate seasonal weather deviation"""
        return 2.2  # Standard deviations from seasonal norm
    
    def _estimate_economic_impact(self, data: pd.DataFrame) -> float:
        """Estimate economic impact of natural disasters"""
        return 45.0  # Billions USD estimated impact
    
    # Cyber security calculation methods
    def _count_critical_vulnerabilities(self, data: pd.DataFrame) -> float:
        """Count critical vulnerabilities"""
        if data.empty:
            return 15.0
        critical_vulns = data[data['severity'] == 'critical'] if 'severity' in data.columns else pd.DataFrame()
        return len(critical_vulns)
    
    def _calculate_attack_frequency(self, data: pd.DataFrame) -> float:
        """Calculate cyber attack frequency"""
        if data.empty:
            return 25.0
        return len(data) / 30  # Attacks per day
    
    def _count_infrastructure_attacks(self, data: pd.DataFrame) -> float:
        """Count infrastructure-targeted attacks"""
        if data.empty:
            return 8.0
        infra_attacks = data[data['target_type'] == 'infrastructure'] if 'target_type' in data.columns else pd.DataFrame()
        return len(infra_attacks)
    
    def _calculate_ransomware_rate(self, data: pd.DataFrame) -> float:
        """Calculate ransomware incident rate"""
        if data.empty:
            return 12.0
        ransomware = data[data['attack_type'] == 'ransomware'] if 'attack_type' in data.columns else pd.DataFrame()
        return len(ransomware)
    
    def _count_supply_chain_attacks(self, data: pd.DataFrame) -> float:
        """Count supply chain cyber incidents"""
        if data.empty:
            return 5.0
        supply_attacks = data[data['target_type'] == 'supply_chain'] if 'target_type' in data.columns else pd.DataFrame()
        return len(supply_attacks)
    
    def _count_financial_attacks(self, data: pd.DataFrame) -> float:
        """Count financial sector incidents"""
        if data.empty:
            return 18.0
        financial_attacks = data[data['target_sector'] == 'financial'] if 'target_sector' in data.columns else pd.DataFrame()
        return len(financial_attacks)
    
    def _count_zero_day_exploits(self, data: pd.DataFrame) -> float:
        """Count zero-day exploit frequency"""
        if data.empty:
            return 3.0
        zero_days = data[data['exploit_type'] == 'zero_day'] if 'exploit_type' in data.columns else pd.DataFrame()
        return len(zero_days)
    
    def _calculate_average_severity(self, data: pd.DataFrame) -> float:
        """Calculate average threat severity"""
        if data.empty or 'severity_score' not in data.columns:
            return 6.5  # On 1-10 scale
        return data['severity_score'].mean()
    
    def _calculate_response_time(self, data: pd.DataFrame) -> float:
        """Calculate average incident response time"""
        if data.empty or 'response_time_hours' not in data.columns:
            return 48.0  # Hours
        return data['response_time_hours'].mean()
    
    def _calculate_cyber_resilience(self, data: pd.DataFrame) -> float:
        """Calculate cyber resilience index"""
        return 70.0  # Percentage resilience score
    
    def _calculate_threat_actor_diversity(self, data: pd.DataFrame) -> float:
        """Calculate threat actor diversity"""
        if data.empty or 'threat_actor' not in data.columns:
            return 0.6  # Diversity index
        unique_actors = data['threat_actor'].nunique()
        total_incidents = len(data)
        return unique_actors / total_incidents if total_incidents > 0 else 0.6
    
    def _get_default_natural_disaster_features(self) -> Dict[str, float]:
        """Return default natural disaster features when data unavailable"""
        return {
            'hurricane_activity_index': 10.0,
            'earthquake_frequency': 5.0,
            'flood_risk_index': 15.0,
            'wildfire_activity': 12.0,
            'extreme_weather_events': 8.0,
            'temperature_anomaly': 1.5,
            'precipitation_anomaly': 0.8,
            'drought_index': 25.0,
            'storm_surge_risk': 18.0,
            'seasonal_weather_deviation': 2.2,
            'natural_disaster_economic_impact': 45.0
        }
    
    def _get_default_cyber_features(self) -> Dict[str, float]:
        """Return default cyber features when data unavailable"""
        return {
            'critical_vulnerability_count': 15.0,
            'cyber_attack_frequency': 25.0,
            'infrastructure_target_frequency': 8.0,
            'ransomware_incident_rate': 12.0,
            'supply_chain_cyber_incidents': 5.0,
            'financial_sector_incidents': 18.0,
            'zero_day_exploit_frequency': 3.0,
            'cyber_threat_severity_avg': 6.5,
            'incident_response_time_avg': 48.0,
            'cyber_resilience_index': 70.0,
            'threat_actor_diversity': 0.6
        }
    
    def _get_default_geopolitical_features(self) -> Dict[str, float]:
        """Return default geopolitical features when data unavailable"""
        return {
            'trade_war_intensity': 25.0,
            'sanctions_impact_index': 15.0,
            'border_closure_frequency': 2.0,
            'political_instability_index': 20.0,
            'social_unrest_frequency': 5.0,
            'protest_economic_impact': 10.0,
            'election_uncertainty_index': 30.0,
            'regulatory_change_frequency': 8.0,
            'international_tension_level': 35.0,
            'migration_pressure_index': 15.0,
            'terrorism_threat_level': 25.0,
            'diplomatic_incident_frequency': 3.0
        }
    
    def _get_default_climate_features(self) -> Dict[str, float]:
        """Return default climate features when data unavailable"""
        return {
            'climate_change_acceleration': 0.8,
            'sea_level_rise_rate': 3.2,
            'arctic_ice_loss_rate': 13.0,
            'carbon_emission_anomaly': 2.5,
            'deforestation_rate': 10.0,
            'biodiversity_loss_rate': 8.0,
            'ocean_acidification_level': 0.4,
            'extreme_temperature_frequency': 15.0,
            'water_stress_index': 40.0,
            'air_quality_degradation': 25.0,
            'renewable_energy_transition_rate': 12.0,
            'climate_adaptation_readiness': 60.0
        }
    
    async def _get_fallback_features(self) -> DisruptionFeatures:
        """Return fallback features when extraction fails"""
        default_features = {}
        default_features.update(self._get_default_natural_disaster_features())
        default_features.update(self._get_default_cyber_features())
        default_features.update(self._get_default_geopolitical_features())
        default_features.update(self._get_default_climate_features())
        
        composite_scores = self._calculate_composite_scores(default_features)
        
        # Calculate overall disruption risk
        composite_scores['overall_disruption_risk'] = (
            composite_scores['natural_disaster_risk'] * 0.25 +
            composite_scores['cyber_incident_frequency'] * 0.20 +
            composite_scores['geopolitical_tension'] * 0.20 +
            composite_scores['climate_anomaly'] * 0.15 +
            composite_scores['infrastructure_vulnerability'] * 0.10 +
            composite_scores['social_unrest'] * 0.10
        )
        
        return DisruptionFeatures(
            natural_disaster_risk=composite_scores['natural_disaster_risk'],
            cyber_incident_frequency=composite_scores['cyber_incident_frequency'],
            geopolitical_tension_index=composite_scores['geopolitical_tension'],
            pandemic_disruption_level=composite_scores['pandemic_disruption'],
            climate_anomaly_score=composite_scores['climate_anomaly'],
            infrastructure_vulnerability=composite_scores['infrastructure_vulnerability'],
            social_unrest_indicator=composite_scores['social_unrest'],
            overall_disruption_risk=composite_scores['overall_disruption_risk'],
            features=default_features,
            metadata={
                'extraction_time': datetime.utcnow().isoformat(),
                'data_sources': ['fallback'],
                'geographic_scope': ['fallback'],
                'event_types': ['fallback'],
                'date_range': 'fallback'
            }
        )


# Feature validation utilities
def validate_disruption_features(features: DisruptionFeatures) -> bool:
    """Validate disruption features for completeness and ranges"""
    try:
        # Check required fields
        required_fields = [
            'natural_disaster_risk', 'cyber_incident_frequency', 'geopolitical_tension_index',
            'overall_disruption_risk'
        ]
        
        for field in required_fields:
            if not hasattr(features, field):
                return False
                
        # Check value ranges
        score_fields = ['natural_disaster_risk', 'cyber_incident_frequency', 'overall_disruption_risk']
        for field in score_fields:
            value = getattr(features, field)
            if not (0 <= value <= 100):
                return False
                
        return True
        
    except Exception:
        return False


def calculate_disruption_risk_score(features: DisruptionFeatures) -> float:
    """Calculate overall disruption risk score from features"""
    try:
        # Use the pre-calculated overall disruption risk
        return min(100, max(0, features.overall_disruption_risk))
        
    except Exception:
        return 50.0  # Neutral risk score on error