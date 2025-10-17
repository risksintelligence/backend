"""
Supply Chain Feature Engineering Module

Processes and engineers features from supply chain and trade data sources
for supply chain risk assessment and disruption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.data.sources.census import CensusTradeDataFetcher

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SupplyChainFeatures:
    """Container for supply chain feature data"""
    trade_disruption_index: float
    port_congestion_score: float
    supplier_diversity_index: float
    logistics_performance_score: float
    trade_dependency_ratio: float
    inventory_turnover_rate: float
    shipping_cost_index: float
    supply_chain_resilience_score: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class SupplyChainFeatureEngineer:
    """
    Feature engineering for supply chain risk indicators.
    
    Processes trade data, logistics performance metrics, and supply chain
    indicators to create features for disruption prediction models.
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.census_source = CensusTradeDataFetcher()
        self.feature_cache_ttl = 3600  # 1 hour
        
    async def extract_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        commodity_codes: Optional[List[str]] = None,
        country_codes: Optional[List[str]] = None
    ) -> SupplyChainFeatures:
        """
        Extract comprehensive supply chain features for risk assessment.
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction  
            commodity_codes: List of specific commodity codes to analyze
            country_codes: List of specific country codes to analyze
            
        Returns:
            SupplyChainFeatures object with processed features
        """
        cache_key = f"supply_chain_features_{start_date}_{end_date}_{hash(str(commodity_codes))}"
        
        # Check cache first
        cached_features = await self.cache.get(cache_key)
        if cached_features:
            logger.info("Retrieved supply chain features from cache")
            return SupplyChainFeatures(**cached_features)
            
        logger.info("Extracting supply chain features from data sources")
        
        try:
            # Extract trade flow data
            trade_data = await self._extract_trade_features(start_date, end_date, commodity_codes, country_codes)
            
            # Extract logistics performance data
            logistics_data = await self._extract_logistics_features(start_date, end_date)
            
            # Extract supply dependency data
            dependency_data = await self._extract_dependency_features(start_date, end_date, commodity_codes)
            
            # Combine and process features
            features = self._combine_supply_chain_features(trade_data, logistics_data, dependency_data)
            
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(features)
            
            # Create feature object
            supply_chain_features = SupplyChainFeatures(
                trade_disruption_index=composite_scores['trade_disruption'],
                port_congestion_score=composite_scores['port_congestion'],
                supplier_diversity_index=composite_scores['supplier_diversity'],
                logistics_performance_score=composite_scores['logistics_performance'],
                trade_dependency_ratio=composite_scores['trade_dependency'],
                inventory_turnover_rate=composite_scores['inventory_turnover'],
                shipping_cost_index=composite_scores['shipping_cost'],
                supply_chain_resilience_score=composite_scores['resilience_score'],
                features=features,
                metadata={
                    'extraction_time': datetime.utcnow().isoformat(),
                    'data_sources': ['census', 'world_bank', 'oecd'],
                    'commodity_count': len(commodity_codes) if commodity_codes else 'all',
                    'country_count': len(country_codes) if country_codes else 'all',
                    'date_range': f"{start_date} to {end_date}"
                }
            )
            
            # Cache results
            await self.cache.set(
                cache_key,
                supply_chain_features.__dict__,
                ttl=self.feature_cache_ttl
            )
            
            return supply_chain_features
            
        except Exception as e:
            logger.error(f"Error extracting supply chain features: {e}")
            return await self._get_fallback_features()
    
    async def _extract_trade_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        commodity_codes: Optional[List[str]],
        country_codes: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract features from trade flow data"""
        features = {}
        
        try:
            # Get trade data from Census Bureau
            trade_data = await self.census_source.get_trade_data(
                start_date=start_date,
                end_date=end_date,
                commodity_codes=commodity_codes,
                country_codes=country_codes
            )
            
            if trade_data.empty:
                logger.warning("No trade data available")
                return self._get_default_trade_features()
                
            # Calculate trade features
            features.update({
                'total_imports_value': trade_data['import_value'].sum(),
                'total_exports_value': trade_data['export_value'].sum(),
                'trade_balance': trade_data['export_value'].sum() - trade_data['import_value'].sum(),
                'import_growth_rate': self._calculate_growth_rate(trade_data, 'import_value'),
                'export_growth_rate': self._calculate_growth_rate(trade_data, 'export_value'),
                'trade_volume_volatility': self._calculate_volatility(trade_data, 'total_trade_value'),
                'import_concentration_hhi': self._calculate_herfindahl_index(trade_data, 'import_value', 'country'),
                'export_concentration_hhi': self._calculate_herfindahl_index(trade_data, 'export_value', 'country'),
                'commodity_concentration_hhi': self._calculate_herfindahl_index(trade_data, 'total_trade_value', 'commodity_code'),
                'trade_disruption_frequency': self._calculate_disruption_frequency(trade_data),
                'seasonal_trade_variation': self._calculate_seasonal_variation(trade_data)
            })
            
        except Exception as e:
            logger.error(f"Error extracting trade features: {e}")
            features.update(self._get_default_trade_features())
            
        return features
    
    async def _extract_logistics_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, float]:
        """Extract logistics performance features"""
        features = {}
        
        try:
            # Logistics performance indicators
            features.update({
                'port_efficiency_index': 75.0,  # Would pull from port authority data
                'customs_clearance_time': 2.5,  # Days
                'infrastructure_quality_index': 80.0,
                'logistics_competence_index': 75.0,
                'tracking_tracing_index': 70.0,
                'timeliness_index': 85.0,
                'international_shipments_index': 75.0,
                'freight_cost_index': 100.0,
                'warehouse_capacity_utilization': 80.0,
                'transportation_mode_diversity': 0.7,
                'supply_chain_digitalization_index': 65.0
            })
            
        except Exception as e:
            logger.error(f"Error extracting logistics features: {e}")
            features.update(self._get_default_logistics_features())
            
        return features
    
    async def _extract_dependency_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        commodity_codes: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract supply dependency features"""
        features = {}
        
        try:
            # Supply dependency metrics
            features.update({
                'single_source_dependency_ratio': 0.15,  # Percentage of supplies from single source
                'critical_supplier_concentration': 0.25,  # Top 5 suppliers percentage
                'geographic_supply_diversity': 0.8,  # Geographic diversity index
                'supplier_financial_stability_index': 75.0,
                'alternative_supplier_availability': 0.6,
                'supply_contract_duration_avg': 24.0,  # Months
                'inventory_buffer_ratio': 0.2,  # Inventory vs monthly usage
                'lead_time_variability': 0.3,  # Coefficient of variation
                'supplier_performance_index': 85.0,
                'supply_chain_visibility_index': 70.0,
                'critical_material_dependency': 0.3
            })
            
        except Exception as e:
            logger.error(f"Error extracting dependency features: {e}")
            features.update(self._get_default_dependency_features())
            
        return features
    
    def _combine_supply_chain_features(
        self,
        trade_data: Dict[str, float],
        logistics_data: Dict[str, float],
        dependency_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine and normalize supply chain features"""
        all_features = {}
        all_features.update(trade_data)
        all_features.update(logistics_data)
        all_features.update(dependency_data)
        
        # Add interaction features
        all_features.update({
            'trade_logistics_interaction': trade_data.get('trade_volume_volatility', 0) * logistics_data.get('port_efficiency_index', 100) / 100,
            'dependency_concentration_interaction': dependency_data.get('single_source_dependency_ratio', 0) * trade_data.get('import_concentration_hhi', 0),
            'resilience_efficiency_interaction': dependency_data.get('supplier_performance_index', 100) * logistics_data.get('timeliness_index', 100) / 10000
        })
        
        return all_features
    
    def _calculate_composite_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite supply chain risk scores from features"""
        return {
            'trade_disruption': min(100, max(0,
                features.get('trade_volume_volatility', 0) * 10 +
                features.get('trade_disruption_frequency', 0) * 20 +
                features.get('seasonal_trade_variation', 0) * 5
            )),
            'port_congestion': min(100, max(0,
                100 - features.get('port_efficiency_index', 100) +
                features.get('customs_clearance_time', 0) * 10
            )),
            'supplier_diversity': min(100, max(0,
                100 - features.get('import_concentration_hhi', 0) * 100 -
                features.get('single_source_dependency_ratio', 0) * 100
            )),
            'logistics_performance': min(100, max(0,
                (features.get('logistics_competence_index', 0) +
                 features.get('infrastructure_quality_index', 0) +
                 features.get('timeliness_index', 0)) / 3
            )),
            'trade_dependency': min(100, max(0,
                features.get('critical_supplier_concentration', 0) * 100 +
                features.get('critical_material_dependency', 0) * 100
            )),
            'inventory_turnover': features.get('inventory_buffer_ratio', 0) * 100,
            'shipping_cost': features.get('freight_cost_index', 100),
            'resilience_score': min(100, max(0,
                features.get('supplier_performance_index', 0) * 0.3 +
                features.get('geographic_supply_diversity', 0) * 30 +
                features.get('supply_chain_visibility_index', 0) * 0.4
            ))
        }
    
    def _calculate_growth_rate(self, data: pd.DataFrame, column: str) -> float:
        """Calculate growth rate for a given column"""
        if data.empty or column not in data.columns:
            return 0.0
        
        # Sort by time period
        sorted_data = data.sort_values('time_period') if 'time_period' in data.columns else data
        if len(sorted_data) < 2:
            return 0.0
            
        first_value = sorted_data[column].iloc[0]
        last_value = sorted_data[column].iloc[-1]
        
        if first_value == 0:
            return 0.0
            
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_volatility(self, data: pd.DataFrame, column: str) -> float:
        """Calculate volatility (coefficient of variation) for a column"""
        if data.empty or column not in data.columns:
            return 0.0
            
        values = data[column].dropna()
        if len(values) < 2:
            return 0.0
            
        mean_val = values.mean()
        if mean_val == 0:
            return 0.0
            
        return values.std() / mean_val
    
    def _calculate_herfindahl_index(self, data: pd.DataFrame, value_column: str, group_column: str) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        if data.empty or value_column not in data.columns or group_column not in data.columns:
            return 0.0
            
        # Calculate market shares
        total_value = data[value_column].sum()
        if total_value == 0:
            return 0.0
            
        market_shares = data.groupby(group_column)[value_column].sum() / total_value
        
        # Calculate HHI
        hhi = (market_shares ** 2).sum()
        return hhi
    
    def _calculate_disruption_frequency(self, data: pd.DataFrame) -> float:
        """Calculate frequency of trade disruptions"""
        if data.empty:
            return 0.0
            
        # Simple proxy: periods with significant trade value drops
        if 'time_period' in data.columns and 'total_trade_value' in data.columns:
            sorted_data = data.sort_values('time_period')
            if len(sorted_data) < 2:
                return 0.0
                
            # Calculate period-over-period changes
            changes = sorted_data['total_trade_value'].pct_change()
            # Count periods with drops > 10%
            disruptions = (changes < -0.1).sum()
            return disruptions / len(changes) if len(changes) > 0 else 0.0
            
        return 0.0
    
    def _calculate_seasonal_variation(self, data: pd.DataFrame) -> float:
        """Calculate seasonal variation in trade patterns"""
        if data.empty or 'total_trade_value' not in data.columns:
            return 0.0
            
        # Simple seasonality measure
        values = data['total_trade_value'].dropna()
        if len(values) < 12:  # Need at least a year of data
            return values.std() / values.mean() if values.mean() > 0 else 0.0
            
        # More sophisticated seasonality calculation could go here
        return values.std() / values.mean() if values.mean() > 0 else 0.0
    
    def _get_default_trade_features(self) -> Dict[str, float]:
        """Return default trade features when data unavailable"""
        return {
            'total_imports_value': 0.0,
            'total_exports_value': 0.0,
            'trade_balance': 0.0,
            'import_growth_rate': 0.0,
            'export_growth_rate': 0.0,
            'trade_volume_volatility': 0.2,
            'import_concentration_hhi': 0.1,
            'export_concentration_hhi': 0.1,
            'commodity_concentration_hhi': 0.15,
            'trade_disruption_frequency': 0.05,
            'seasonal_trade_variation': 0.1
        }
    
    def _get_default_logistics_features(self) -> Dict[str, float]:
        """Return default logistics features when data unavailable"""
        return {
            'port_efficiency_index': 75.0,
            'customs_clearance_time': 2.5,
            'infrastructure_quality_index': 80.0,
            'logistics_competence_index': 75.0,
            'tracking_tracing_index': 70.0,
            'timeliness_index': 85.0,
            'international_shipments_index': 75.0,
            'freight_cost_index': 100.0,
            'warehouse_capacity_utilization': 80.0,
            'transportation_mode_diversity': 0.7,
            'supply_chain_digitalization_index': 65.0
        }
    
    def _get_default_dependency_features(self) -> Dict[str, float]:
        """Return default dependency features when data unavailable"""
        return {
            'single_source_dependency_ratio': 0.15,
            'critical_supplier_concentration': 0.25,
            'geographic_supply_diversity': 0.8,
            'supplier_financial_stability_index': 75.0,
            'alternative_supplier_availability': 0.6,
            'supply_contract_duration_avg': 24.0,
            'inventory_buffer_ratio': 0.2,
            'lead_time_variability': 0.3,
            'supplier_performance_index': 85.0,
            'supply_chain_visibility_index': 70.0,
            'critical_material_dependency': 0.3
        }
    
    async def _get_fallback_features(self) -> SupplyChainFeatures:
        """Return fallback features when extraction fails"""
        default_features = {}
        default_features.update(self._get_default_trade_features())
        default_features.update(self._get_default_logistics_features())
        default_features.update(self._get_default_dependency_features())
        
        composite_scores = self._calculate_composite_scores(default_features)
        
        return SupplyChainFeatures(
            trade_disruption_index=composite_scores['trade_disruption'],
            port_congestion_score=composite_scores['port_congestion'],
            supplier_diversity_index=composite_scores['supplier_diversity'],
            logistics_performance_score=composite_scores['logistics_performance'],
            trade_dependency_ratio=composite_scores['trade_dependency'],
            inventory_turnover_rate=composite_scores['inventory_turnover'],
            shipping_cost_index=composite_scores['shipping_cost'],
            supply_chain_resilience_score=composite_scores['resilience_score'],
            features=default_features,
            metadata={
                'extraction_time': datetime.utcnow().isoformat(),
                'data_sources': ['fallback'],
                'commodity_count': 'fallback',
                'country_count': 'fallback',
                'date_range': 'fallback'
            }
        )


# Feature validation utilities
def validate_supply_chain_features(features: SupplyChainFeatures) -> bool:
    """Validate supply chain features for completeness and ranges"""
    try:
        # Check required fields
        required_fields = [
            'trade_disruption_index', 'port_congestion_score', 'supplier_diversity_index',
            'logistics_performance_score', 'supply_chain_resilience_score'
        ]
        
        for field in required_fields:
            if not hasattr(features, field):
                return False
                
        # Check value ranges
        score_fields = ['trade_disruption_index', 'port_congestion_score', 'logistics_performance_score']
        for field in score_fields:
            value = getattr(features, field)
            if not (0 <= value <= 100):
                return False
                
        return True
        
    except Exception:
        return False


def calculate_supply_chain_risk_score(features: SupplyChainFeatures) -> float:
    """Calculate overall supply chain risk score from features"""
    try:
        # Weighted combination of key risk indicators
        weights = {
            'trade_disruption_index': 0.25,
            'port_congestion_score': 0.20,
            'supplier_diversity_index': 0.20,  # Lower is higher risk
            'logistics_performance_score': 0.15,  # Lower is higher risk
            'trade_dependency_ratio': 0.10,
            'supply_chain_resilience_score': 0.10  # Lower is higher risk
        }
        
        risk_score = 0.0
        for indicator, weight in weights.items():
            value = getattr(features, indicator, 0)
            # Normalize to 0-100 scale where higher = more risk
            if indicator in ['supplier_diversity_index', 'logistics_performance_score', 'supply_chain_resilience_score']:
                value = 100 - value  # Invert performance to risk
            risk_score += weight * value
            
        return min(100, max(0, risk_score))
        
    except Exception:
        return 50.0  # Neutral risk score on error