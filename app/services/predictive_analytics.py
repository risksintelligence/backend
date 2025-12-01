"""
Predictive Disruption Modeling Service

Provides predictive analytics for supply chain disruptions using statistical models
and pattern recognition to forecast potential risks and cascade effects.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DisruptionType(Enum):
    GEOPOLITICAL = "geopolitical"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    TRADE_POLICY = "trade_policy"
    ECONOMIC_SHOCK = "economic_shock"
    PANDEMIC = "pandemic"
    MARITIME_INCIDENT = "maritime_incident"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskPrediction:
    disruption_type: DisruptionType
    probability: float  # 0.0 to 1.0
    risk_level: RiskLevel
    estimated_impact_usd: float
    confidence_score: float  # 0.0 to 1.0
    time_horizon_days: int
    affected_regions: List[str]
    affected_commodities: List[str]
    triggers: List[str]
    mitigation_recommendations: List[str]


@dataclass
class CascadeImpactModel:
    node_id: str
    node_name: str
    direct_impact_probability: float
    indirect_impact_probability: float
    cascade_delay_hours: float
    economic_impact_usd: float
    recovery_time_days: int


class PredictiveAnalyticsService:
    """Predictive analytics service for supply chain disruption modeling."""
    
    def __init__(self):
        # Historical disruption patterns (simplified statistical model)
        self.disruption_patterns = {
            DisruptionType.GEOPOLITICAL: {
                "base_probability": 0.15,
                "seasonal_multiplier": {"winter": 1.3, "summer": 0.8, "spring": 1.0, "fall": 1.1},
                "regional_multipliers": {
                    "middle_east": 2.1,
                    "eastern_europe": 1.8,
                    "south_china_sea": 1.6,
                    "horn_of_africa": 1.4
                },
                "economic_impact_base": 500_000_000
            },
            DisruptionType.NATURAL_DISASTER: {
                "base_probability": 0.25,
                "seasonal_multiplier": {"winter": 0.7, "summer": 1.4, "spring": 1.1, "fall": 1.2},
                "regional_multipliers": {
                    "pacific_rim": 2.2,
                    "caribbean": 1.8,
                    "mediterranean": 1.3,
                    "indian_ocean": 1.5
                },
                "economic_impact_base": 750_000_000
            },
            DisruptionType.CYBER_ATTACK: {
                "base_probability": 0.20,
                "seasonal_multiplier": {"winter": 1.0, "summer": 1.0, "spring": 1.0, "fall": 1.0},
                "regional_multipliers": {
                    "north_america": 1.4,
                    "europe": 1.3,
                    "east_asia": 1.5,
                    "middle_east": 1.2
                },
                "economic_impact_base": 300_000_000
            },
            DisruptionType.MARITIME_INCIDENT: {
                "base_probability": 0.18,
                "seasonal_multiplier": {"winter": 1.5, "summer": 0.9, "spring": 1.1, "fall": 1.2},
                "regional_multipliers": {
                    "suez_canal": 3.0,
                    "panama_canal": 2.5,
                    "strait_of_hormuz": 2.8,
                    "strait_of_malacca": 2.2
                },
                "economic_impact_base": 1_200_000_000
            }
        }
        
        # Critical supply chain nodes with vulnerability scores
        self.critical_nodes = {
            "singapore": {
                "vulnerability_score": 0.35,
                "strategic_importance": 0.92,
                "recovery_capability": 0.78
            },
            "suez_canal": {
                "vulnerability_score": 0.65,
                "strategic_importance": 0.95,
                "recovery_capability": 0.60
            },
            "shanghai": {
                "vulnerability_score": 0.42,
                "strategic_importance": 0.88,
                "recovery_capability": 0.85
            },
            "long_beach_port": {
                "vulnerability_score": 0.38,
                "strategic_importance": 0.82,
                "recovery_capability": 0.75
            }
        }

    def predict_disruptions(
        self, 
        time_horizon_days: int = 30,
        include_cascade_effects: bool = True
    ) -> List[RiskPrediction]:
        """Generate predictive risk assessments for supply chain disruptions."""
        
        predictions = []
        current_date = datetime.utcnow()
        
        # Get current season for seasonal adjustments
        month = current_date.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"
        
        for disruption_type, pattern_data in self.disruption_patterns.items():
            # Calculate base probability adjusted for seasonality
            base_prob = pattern_data["base_probability"]
            seasonal_mult = pattern_data["seasonal_multiplier"][season]
            adjusted_prob = base_prob * seasonal_mult
            
            # Apply time horizon adjustments (higher probability over longer periods)
            time_factor = min(1.5, 1.0 + (time_horizon_days / 365))
            final_probability = min(0.95, adjusted_prob * time_factor)
            
            # Determine risk level
            if final_probability >= 0.7:
                risk_level = RiskLevel.CRITICAL
            elif final_probability >= 0.5:
                risk_level = RiskLevel.HIGH
            elif final_probability >= 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Calculate economic impact with uncertainty
            base_impact = pattern_data["economic_impact_base"]
            impact_multiplier = 1.0 + (final_probability * 0.8)  # Higher probability = higher impact
            estimated_impact = base_impact * impact_multiplier
            
            # Generate affected regions and commodities
            affected_regions = self._get_high_risk_regions(disruption_type)
            affected_commodities = self._get_vulnerable_commodities(disruption_type)
            
            # Calculate confidence based on data quality and model accuracy
            confidence = self._calculate_confidence_score(disruption_type, final_probability)
            
            # Generate mitigation recommendations
            mitigation_recs = self._generate_mitigation_strategies(disruption_type, risk_level)
            
            # Generate risk triggers
            triggers = self._generate_risk_triggers(disruption_type, season)
            
            prediction = RiskPrediction(
                disruption_type=disruption_type,
                probability=final_probability,
                risk_level=risk_level,
                estimated_impact_usd=estimated_impact,
                confidence_score=confidence,
                time_horizon_days=time_horizon_days,
                affected_regions=affected_regions,
                affected_commodities=affected_commodities,
                triggers=triggers,
                mitigation_recommendations=mitigation_recs
            )
            
            predictions.append(prediction)
        
        # Sort by probability descending
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        if include_cascade_effects:
            # Add cascade impact modeling
            cascade_models = self._model_cascade_impacts(predictions)
            return predictions, cascade_models
        
        return predictions

    def _get_high_risk_regions(self, disruption_type: DisruptionType) -> List[str]:
        """Get regions at high risk for specific disruption types."""
        
        region_mapping = {
            DisruptionType.GEOPOLITICAL: ["Middle East", "Eastern Europe", "South China Sea", "Horn of Africa"],
            DisruptionType.NATURAL_DISASTER: ["Pacific Ring of Fire", "Caribbean", "Mediterranean", "Indian Ocean"],
            DisruptionType.CYBER_ATTACK: ["North America", "Europe", "East Asia", "Middle East"],
            DisruptionType.MARITIME_INCIDENT: ["Suez Canal", "Panama Canal", "Strait of Hormuz", "Strait of Malacca"],
            DisruptionType.TRADE_POLICY: ["US-China Trade Routes", "EU-UK Routes", "NAFTA Corridor"],
            DisruptionType.ECONOMIC_SHOCK: ["Global Financial Centers", "Manufacturing Hubs", "Energy Corridors"]
        }
        
        return region_mapping.get(disruption_type, ["Global"])

    def _get_vulnerable_commodities(self, disruption_type: DisruptionType) -> List[str]:
        """Get commodities most vulnerable to specific disruption types."""
        
        commodity_mapping = {
            DisruptionType.GEOPOLITICAL: ["crude_oil", "natural_gas", "rare_earth_metals", "grain"],
            DisruptionType.NATURAL_DISASTER: ["agricultural_products", "energy", "minerals", "textiles"],
            DisruptionType.CYBER_ATTACK: ["semiconductors", "electronics", "financial_services", "telecommunications"],
            DisruptionType.MARITIME_INCIDENT: ["containerized_goods", "crude_oil", "lng", "automobiles"],
            DisruptionType.TRADE_POLICY: ["manufactured_goods", "agricultural_products", "technology", "steel"],
            DisruptionType.ECONOMIC_SHOCK: ["commodities", "financial_instruments", "luxury_goods", "real_estate"]
        }
        
        return commodity_mapping.get(disruption_type, ["general_goods"])

    def _calculate_confidence_score(self, disruption_type: DisruptionType, probability: float) -> float:
        """Calculate confidence score for prediction."""
        
        # Base confidence depends on disruption type (some are more predictable)
        base_confidence = {
            DisruptionType.NATURAL_DISASTER: 0.75,  # Weather patterns are somewhat predictable
            DisruptionType.MARITIME_INCIDENT: 0.70,  # Historical patterns available
            DisruptionType.GEOPOLITICAL: 0.55,      # More unpredictable
            DisruptionType.CYBER_ATTACK: 0.60,      # Trend analysis available
            DisruptionType.TRADE_POLICY: 0.65,      # Policy patterns somewhat predictable
            DisruptionType.ECONOMIC_SHOCK: 0.50     # Highly unpredictable
        }.get(disruption_type, 0.60)
        
        # Adjust confidence based on probability (moderate probabilities are more confident)
        prob_adjustment = 1.0 - abs(0.5 - probability) * 0.4
        
        return min(0.95, base_confidence * prob_adjustment)

    def _generate_mitigation_strategies(
        self, 
        disruption_type: DisruptionType, 
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate mitigation strategy recommendations."""
        
        base_strategies = {
            DisruptionType.GEOPOLITICAL: [
                "Diversify supplier base across multiple regions",
                "Establish strategic inventory buffers",
                "Monitor geopolitical risk indicators",
                "Develop alternative trade routes"
            ],
            DisruptionType.NATURAL_DISASTER: [
                "Implement climate-resilient infrastructure",
                "Create disaster response protocols",
                "Establish emergency supplier networks",
                "Invest in early warning systems"
            ],
            DisruptionType.CYBER_ATTACK: [
                "Strengthen cybersecurity infrastructure",
                "Implement multi-factor authentication",
                "Develop incident response plans",
                "Create offline backup systems"
            ],
            DisruptionType.MARITIME_INCIDENT: [
                "Diversify shipping routes and carriers",
                "Monitor vessel tracking systems",
                "Establish port contingency plans",
                "Invest in maritime insurance"
            ]
        }
        
        strategies = base_strategies.get(disruption_type, ["Develop general contingency plans"])
        
        # Add risk-level specific recommendations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            strategies.extend([
                "Activate crisis management team",
                "Accelerate inventory buildup",
                "Engage emergency suppliers immediately"
            ])
        
        return strategies

    def _generate_risk_triggers(
        self, 
        disruption_type: DisruptionType, 
        season: str
    ) -> List[str]:
        """Generate early warning triggers for disruption types."""
        
        trigger_mapping = {
            DisruptionType.GEOPOLITICAL: [
                "Increased military activity in strategic regions",
                "Diplomatic tensions escalation",
                "Trade dispute announcements",
                "Sanctions implementation"
            ],
            DisruptionType.NATURAL_DISASTER: [
                "Severe weather warnings issued",
                "Seismic activity increases",
                "Hurricane season approaching" if season in ["summer", "fall"] else "Winter storm systems",
                "Climate pattern changes (El Niño/La Niña)"
            ],
            DisruptionType.CYBER_ATTACK: [
                "Increased cyber threat intelligence",
                "High-profile security breaches reported",
                "Nation-state hacking warnings",
                "Critical infrastructure vulnerabilities disclosed"
            ],
            DisruptionType.MARITIME_INCIDENT: [
                "Severe weather in critical shipping lanes",
                "Port congestion increasing",
                "Vessel mechanical issues rising",
                "Piracy activity increases"
            ]
        }
        
        return trigger_mapping.get(disruption_type, ["General risk indicators detected"])

    def _model_cascade_impacts(
        self, 
        primary_predictions: List[RiskPrediction]
    ) -> List[CascadeImpactModel]:
        """Model cascade effects across supply chain network."""
        
        cascade_models = []
        
        for node_id, node_data in self.critical_nodes.items():
            # Calculate impact probability based on primary disruptions
            direct_impact_prob = 0.0
            indirect_impact_prob = 0.0
            total_economic_impact = 0.0
            
            for prediction in primary_predictions:
                # Direct impact probability
                if any(region.lower().replace(" ", "_") in node_id for region in prediction.affected_regions):
                    direct_impact_prob = max(direct_impact_prob, 
                                           prediction.probability * node_data["vulnerability_score"])
                
                # Indirect impact through network effects
                network_effect = prediction.probability * 0.3  # 30% of primary disruption probability
                indirect_impact_prob = max(indirect_impact_prob, network_effect)
                
                # Economic impact calculation
                node_strategic_value = node_data["strategic_importance"] * 1_000_000_000  # $1B base
                impact_factor = prediction.probability * node_data["vulnerability_score"]
                total_economic_impact += prediction.estimated_impact_usd * impact_factor * 0.1
            
            # Calculate cascade delay based on node characteristics
            recovery_capability = node_data["recovery_capability"]
            base_delay = 48  # 48 hours base
            cascade_delay = base_delay / recovery_capability
            
            # Recovery time estimation
            recovery_days = int(7 / recovery_capability)  # 7 days base, adjusted by capability
            
            cascade_model = CascadeImpactModel(
                node_id=node_id,
                node_name=node_id.replace("_", " ").title(),
                direct_impact_probability=min(0.95, direct_impact_prob),
                indirect_impact_probability=min(0.95, indirect_impact_prob),
                cascade_delay_hours=cascade_delay,
                economic_impact_usd=total_economic_impact,
                recovery_time_days=recovery_days
            )
            
            cascade_models.append(cascade_model)
        
        # Sort by total impact probability (direct + indirect)
        cascade_models.sort(
            key=lambda x: x.direct_impact_probability + x.indirect_impact_probability, 
            reverse=True
        )
        
        return cascade_models

    def analyze_disruption_correlation(self, historical_events: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations between different types of disruptions."""
        
        # Simplified correlation analysis
        correlations = {
            "geopolitical_natural_disaster": 0.15,
            "cyber_attack_geopolitical": 0.65,
            "maritime_natural_disaster": 0.45,
            "trade_policy_geopolitical": 0.78,
            "economic_shock_multiple_factors": 0.85
        }
        
        # Risk amplification factors
        amplification_matrix = {
            "simultaneous_disruptions": {
                "2_types": 1.4,
                "3_types": 2.1,
                "4_or_more_types": 3.2
            },
            "cascade_multipliers": {
                "critical_node_affected": 1.8,
                "multiple_regions_affected": 1.5,
                "peak_season_timing": 1.3
            }
        }
        
        return {
            "correlations": correlations,
            "amplification_factors": amplification_matrix,
            "high_risk_combinations": [
                "Geopolitical tension + Maritime incident",
                "Cyber attack + Natural disaster",
                "Trade policy change + Economic shock"
            ]
        }

    def generate_early_warning_system(self) -> Dict[str, Any]:
        """Generate early warning system configuration."""
        
        warning_thresholds = {
            "amber_alert": {
                "probability_threshold": 0.4,
                "impact_threshold_usd": 200_000_000,
                "confidence_threshold": 0.6
            },
            "red_alert": {
                "probability_threshold": 0.7,
                "impact_threshold_usd": 500_000_000,
                "confidence_threshold": 0.7
            }
        }
        
        monitoring_indicators = {
            "geopolitical": [
                "Military deployment indicators",
                "Diplomatic communication sentiment",
                "Trade flow anomalies",
                "Energy price volatility"
            ],
            "natural_disaster": [
                "Meteorological warnings",
                "Seismic activity monitoring",
                "Ocean temperature anomalies",
                "Satellite imagery analysis"
            ],
            "cyber_security": [
                "Network intrusion attempts",
                "Malware detection rates",
                "Critical infrastructure probes",
                "Dark web intelligence"
            ]
        }
        
        return {
            "thresholds": warning_thresholds,
            "indicators": monitoring_indicators,
            "update_frequency": "every_4_hours",
            "escalation_protocols": [
                "Automated stakeholder notification",
                "Emergency response team activation",
                "Supply chain partner alerts",
                "Regulatory body notification"
            ]
        }


# Singleton instance
_predictive_service = None


def get_predictive_analytics() -> PredictiveAnalyticsService:
    """Get the predictive analytics service instance."""
    global _predictive_service
    if _predictive_service is None:
        _predictive_service = PredictiveAnalyticsService()
    return _predictive_service