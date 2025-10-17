"""
Basic risk scoring engine for economic and financial risk assessment.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.data.sources.fred import FREDConnector
from src.cache.cache_manager import CacheManager
from src.ml.features.economic import EconomicFeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Individual risk factor with weight and value."""
    name: str
    category: str
    value: float
    weight: float
    normalized_value: float
    description: str
    confidence: float = 1.0


@dataclass
class RiskScore:
    """Composite risk score with contributing factors."""
    overall_score: float
    risk_level: str
    confidence: float
    factors: List[RiskFactor]
    timestamp: datetime
    methodology_version: str = "1.0"


class BasicRiskScorer:
    """
    Basic risk scoring engine using economic and financial indicators.
    
    This scorer uses a weighted combination of economic indicators to produce
    risk scores from 0-100, where higher scores indicate higher risk.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize risk scorer.
        
        Args:
            cache_manager: Cache manager for data retrieval
        """
        self.cache_manager = cache_manager or CacheManager()
        self.fred_connector = FREDConnector(self.cache_manager)
        self.economic_engineer = EconomicFeatureEngineer()
        
        # Risk factor definitions with weights
        self.risk_factors_config = {
            "unemployment_rate": {
                "weight": 0.15,
                "threshold_low": 3.5,   # Below this is low risk
                "threshold_high": 7.0,  # Above this is high risk
                "direction": "higher_is_riskier",
                "description": "Unemployment rate indicating labor market stress"
            },
            "inflation_rate": {
                "weight": 0.12,
                "threshold_low": 1.5,
                "threshold_high": 4.0,
                "direction": "deviation_from_target",  # Target ~2%
                "target": 2.0,
                "description": "Consumer price inflation rate"
            },
            "federal_funds_rate": {
                "weight": 0.10,
                "threshold_low": 1.0,
                "threshold_high": 5.0,
                "direction": "higher_is_riskier",
                "description": "Federal funds interest rate"
            },
            "yield_curve_spread": {
                "weight": 0.18,
                "threshold_low": 0.5,
                "threshold_high": -0.5,  # Inverted yield curve
                "direction": "lower_is_riskier",
                "description": "10Y-2Y Treasury yield spread (inverted curve = high risk)"
            },
            "gdp_growth": {
                "weight": 0.15,
                "threshold_low": 2.0,
                "threshold_high": -1.0,  # Negative growth
                "direction": "lower_is_riskier",
                "description": "Real GDP quarterly growth rate"
            },
            "trade_balance": {
                "weight": 0.08,
                "threshold_low": -50.0,  # Billions
                "threshold_high": -100.0,
                "direction": "lower_is_riskier",
                "description": "Trade balance (deficit = negative)"
            },
            "financial_stress_index": {
                "weight": 0.12,
                "threshold_low": 0.0,
                "threshold_high": 1.0,
                "direction": "higher_is_riskier",
                "description": "Financial stress conditions"
            },
            "personal_savings_rate": {
                "weight": 0.10,
                "threshold_low": 5.0,   # Below 5% is concerning
                "threshold_high": 15.0,  # Above 15% might indicate fear
                "direction": "deviation_from_normal",
                "normal_range": (8.0, 12.0),
                "description": "Personal savings rate as % of income"
            }
        }
        
        # Risk level thresholds
        self.risk_levels = {
            (0, 25): "low",
            (25, 50): "medium", 
            (50, 75): "high",
            (75, 100): "critical"
        }
    
    async def calculate_risk_score(
        self,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[RiskScore]:
        """
        Calculate comprehensive risk score.
        
        Args:
            use_cache: Whether to use cached data
            force_refresh: Force refresh of all data
            
        Returns:
            RiskScore object with overall score and factors
        """
        cache_key = "risk_score:comprehensive"
        
        if use_cache and not force_refresh:
            cached_score = self.cache_manager.get(cache_key)
            if cached_score:
                return self._deserialize_risk_score(cached_score)
        
        try:
            # Gather all risk factors using enhanced economic features
            factors = []
            
            # Get comprehensive economic features
            try:
                economic_features = await self.economic_engineer.engineer_features()
                if economic_features and economic_features.features:
                    logger.info(f"Successfully engineered {len(economic_features.features)} economic features")
                    # Convert economic features to risk factors
                    factors.extend(self._convert_economic_features_to_risk_factors(economic_features))
                else:
                    logger.warning("No economic features available, using basic FRED data")
                    raise Exception("Economic features unavailable")
            except Exception as e:
                logger.warning(f"Economic features unavailable, falling back to basic FRED: {e}")
                # Fallback to basic FRED indicators
                try:
                    fred_indicators = self.fred_connector.get_key_indicators(use_cache=use_cache)
                except Exception as fred_error:
                    logger.warning(f"FRED data unavailable, using complete fallback: {fred_error}")
                    fred_indicators = {}
                
                # Ensure indicators have fallback structure
                if not fred_indicators:
                    fred_indicators = {}
            
                # Calculate individual risk factors with error handling (fallback mode)
                try:
                    factors.extend(self._calculate_unemployment_risk(fred_indicators))
                except Exception as e:
                    logger.warning(f"Unemployment risk calculation failed: {e}")
                    
                try:
                    factors.extend(self._calculate_inflation_risk(fred_indicators))
                except Exception as e:
                    logger.warning(f"Inflation risk calculation failed: {e}")
                    
                try:
                    factors.extend(self._calculate_interest_rate_risk(fred_indicators))
                except Exception as e:
                    logger.warning(f"Interest rate risk calculation failed: {e}")
                    
                try:
                    factors.extend(self._calculate_yield_curve_risk(fred_indicators))
                except Exception as e:
                    logger.warning(f"Yield curve risk calculation failed: {e}")
                    
                try:
                    factors.extend(self._calculate_financial_stress_risk(fred_indicators))
                except Exception as e:
                    logger.warning(f"Financial stress risk calculation failed: {e}")
            
            # If no factors were calculated, create dummy factors
            if not factors:
                factors = self._create_fallback_factors()
            
            # Calculate overall score
            overall_score, confidence = self._aggregate_risk_factors(factors)
            risk_level = self._determine_risk_level(overall_score)
            
            risk_score = RiskScore(
                overall_score=overall_score,
                risk_level=risk_level,
                confidence=confidence,
                factors=factors,
                timestamp=datetime.utcnow()
            )
            
            # Cache the result
            self.cache_manager.set(
                cache_key, 
                self._serialize_risk_score(risk_score),
                ttl=1800  # 30 minutes
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_unemployment_risk(self, fred_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate unemployment-related risk factors."""
        factors = []
        
        employment_data = fred_data.get("employment", {})
        unrate_data = employment_data.get("UNRATE", {})
        
        if unrate_data and unrate_data.get("value") is not None:
            unemployment_rate = unrate_data["value"]
            config = self.risk_factors_config["unemployment_rate"]
            
            normalized_value = self._normalize_risk_value(
                unemployment_rate,
                config["threshold_low"],
                config["threshold_high"],
                config["direction"]
            )
            
            factors.append(RiskFactor(
                name="unemployment_rate",
                category="employment",
                value=unemployment_rate,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"Unemployment rate: {unemployment_rate}%",
                confidence=0.95
            ))
        
        return factors
    
    def _calculate_inflation_risk(self, fred_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate inflation-related risk factors."""
        factors = []
        
        inflation_data = fred_data.get("inflation", {})
        cpi_data = inflation_data.get("CPIAUCSL", {})
        
        if cpi_data and cpi_data.get("value") is not None:
            # Calculate inflation rate (would need historical data for proper calculation)
            # For now, use a proxy based on CPI level
            cpi_value = cpi_data["value"]
            estimated_inflation = 3.2  # Placeholder - would calculate from time series
            
            config = self.risk_factors_config["inflation_rate"]
            
            if config["direction"] == "deviation_from_target":
                deviation = abs(estimated_inflation - config["target"])
                normalized_value = min(deviation / 3.0, 1.0)  # Max deviation of 3% = 1.0 risk
            else:
                normalized_value = self._normalize_risk_value(
                    estimated_inflation,
                    config["threshold_low"],
                    config["threshold_high"],
                    config["direction"]
                )
            
            factors.append(RiskFactor(
                name="inflation_rate",
                category="inflation",
                value=estimated_inflation,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"Estimated inflation rate: {estimated_inflation:.1f}%",
                confidence=0.8
            ))
        
        return factors
    
    def _calculate_interest_rate_risk(self, fred_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate interest rate risk factors."""
        factors = []
        
        interest_data = fred_data.get("interest_rates", {})
        fedfunds_data = interest_data.get("FEDFUNDS", {})
        
        if fedfunds_data and fedfunds_data.get("value") is not None:
            fed_funds_rate = fedfunds_data["value"]
            config = self.risk_factors_config["federal_funds_rate"]
            
            normalized_value = self._normalize_risk_value(
                fed_funds_rate,
                config["threshold_low"],
                config["threshold_high"],
                config["direction"]
            )
            
            factors.append(RiskFactor(
                name="federal_funds_rate",
                category="interest_rates",
                value=fed_funds_rate,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"Federal funds rate: {fed_funds_rate}%",
                confidence=0.95
            ))
        
        return factors
    
    def _calculate_yield_curve_risk(self, fred_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate yield curve risk factors."""
        factors = []
        
        interest_data = fred_data.get("interest_rates", {})
        dgs10_data = interest_data.get("DGS10", {})
        dgs2_data = interest_data.get("DGS2", {})
        
        if (dgs10_data and dgs10_data.get("value") is not None and 
            dgs2_data and dgs2_data.get("value") is not None):
            
            yield_spread = dgs10_data["value"] - dgs2_data["value"]
            config = self.risk_factors_config["yield_curve_spread"]
            
            normalized_value = self._normalize_risk_value(
                yield_spread,
                config["threshold_low"],
                config["threshold_high"],
                config["direction"]
            )
            
            factors.append(RiskFactor(
                name="yield_curve_spread",
                category="interest_rates",
                value=yield_spread,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"10Y-2Y yield spread: {yield_spread:.2f}%",
                confidence=0.9
            ))
        
        return factors
    
    def _calculate_gdp_risk(self, bea_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate GDP-related risk factors."""
        factors = []
        
        gdp_data = bea_data.get("gdp", {})
        if gdp_data and gdp_data.get("data"):
            # Placeholder GDP growth calculation
            gdp_growth = 2.1  # Would calculate from time series data
            config = self.risk_factors_config["gdp_growth"]
            
            normalized_value = self._normalize_risk_value(
                gdp_growth,
                config["threshold_low"], 
                config["threshold_high"],
                config["direction"]
            )
            
            factors.append(RiskFactor(
                name="gdp_growth",
                category="economic_growth",
                value=gdp_growth,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"Real GDP growth: {gdp_growth:.1f}%",
                confidence=0.85
            ))
        
        return factors
    
    def _calculate_trade_risk(self, bea_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate trade-related risk factors."""
        factors = []
        
        trade_data = bea_data.get("trade", {})
        if trade_data and trade_data.get("data"):
            balance_data = trade_data["data"].get("balance")
            if balance_data:
                trade_balance = balance_data["value"]
                config = self.risk_factors_config["trade_balance"]
                
                normalized_value = self._normalize_risk_value(
                    trade_balance,
                    config["threshold_low"],
                    config["threshold_high"],
                    config["direction"]
                )
                
                factors.append(RiskFactor(
                    name="trade_balance",
                    category="trade",
                    value=trade_balance,
                    weight=config["weight"],
                    normalized_value=normalized_value,
                    description=f"Trade balance: ${trade_balance:.1f}B",
                    confidence=0.9
                ))
        
        return factors
    
    def _calculate_financial_stress_risk(self, fred_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate financial stress risk factors."""
        factors = []
        
        stress_data = fred_data.get("financial_stress", {})
        nfci_data = stress_data.get("NFCI", {})
        
        if nfci_data and nfci_data.get("value") is not None:
            stress_index = nfci_data["value"]
            config = self.risk_factors_config["financial_stress_index"]
            
            normalized_value = self._normalize_risk_value(
                stress_index,
                config["threshold_low"],
                config["threshold_high"],
                config["direction"]
            )
            
            factors.append(RiskFactor(
                name="financial_stress_index",
                category="financial_stress",
                value=stress_index,
                weight=config["weight"],
                normalized_value=normalized_value,
                description=f"Financial stress index: {stress_index:.2f}",
                confidence=0.85
            ))
        
        return factors
    
    def _normalize_risk_value(
        self,
        value: float,
        threshold_low: float,
        threshold_high: float,
        direction: str
    ) -> float:
        """
        Normalize a risk value to 0-1 scale.
        
        Args:
            value: Raw value to normalize
            threshold_low: Low threshold value
            threshold_high: High threshold value
            direction: Risk direction (higher_is_riskier, lower_is_riskier, etc.)
            
        Returns:
            Normalized risk value between 0 and 1
        """
        if direction == "higher_is_riskier":
            if value <= threshold_low:
                return 0.0
            elif value >= threshold_high:
                return 1.0
            else:
                return (value - threshold_low) / (threshold_high - threshold_low)
        
        elif direction == "lower_is_riskier":
            if value >= threshold_low:
                return 0.0
            elif value <= threshold_high:
                return 1.0
            else:
                return (threshold_low - value) / (threshold_low - threshold_high)
        
        elif direction == "deviation_from_target":
            # Would use target value for this calculation
            return min(abs(value - 2.0) / 3.0, 1.0)  # Placeholder
        
        return 0.5  # Default neutral risk
    
    def _aggregate_risk_factors(self, factors: List[RiskFactor]) -> Tuple[float, float]:
        """
        Aggregate individual risk factors into overall score.
        
        Args:
            factors: List of risk factors
            
        Returns:
            Tuple of (overall_score, confidence)
        """
        if not factors:
            return 0.0, 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for factor in factors:
            weighted_score = factor.normalized_value * factor.weight
            total_weighted_score += weighted_score
            total_weight += factor.weight
            total_confidence += factor.confidence * factor.weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        # Scale to 0-100
        overall_score = (total_weighted_score / total_weight) * 100
        confidence = total_confidence / total_weight
        
        return round(overall_score, 2), round(confidence, 3)
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        for (low, high), level in self.risk_levels.items():
            if low <= score < high:
                return level
        return "critical"  # For scores >= 75
    
    def _serialize_risk_score(self, risk_score: RiskScore) -> Dict[str, Any]:
        """Serialize risk score for caching."""
        return {
            "overall_score": risk_score.overall_score,
            "risk_level": risk_score.risk_level,
            "confidence": risk_score.confidence,
            "timestamp": risk_score.timestamp.isoformat(),
            "methodology_version": risk_score.methodology_version,
            "factors": [
                {
                    "name": f.name,
                    "category": f.category,
                    "value": f.value,
                    "weight": f.weight,
                    "normalized_value": f.normalized_value,
                    "description": f.description,
                    "confidence": f.confidence
                }
                for f in risk_score.factors
            ]
        }
    
    def _deserialize_risk_score(self, data: Dict[str, Any]) -> RiskScore:
        """Deserialize cached risk score."""
        factors = []
        for f in data.get("factors", []):
            try:
                factors.append(RiskFactor(
                    name=f.get("name", "unknown"),
                    category=f.get("category", "unknown"),
                    value=f.get("value", 0.0),
                    weight=f.get("weight", 0.1),
                    normalized_value=f.get("normalized_value", 0.5),
                    description=f.get("description", "No description"),
                    confidence=f.get("confidence", 0.5)
                ))
            except Exception as e:
                logger.warning(f"Failed to deserialize factor {f}: {e}")
                continue
        
        return RiskScore(
            overall_score=data.get("overall_score", 50.0),
            risk_level=data.get("risk_level", "medium"),
            confidence=data.get("confidence", 0.5),
            factors=factors,
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            methodology_version=data.get("methodology_version", "1.0")
        )
    
    def _create_fallback_factors(self) -> List[RiskFactor]:
        """Create fallback risk factors when data is unavailable."""
        return [
            RiskFactor(
                name="unemployment_rate",
                category="employment",
                value=3.8,
                weight=0.15,
                normalized_value=0.4,
                description="Unemployment rate: 3.8% (fallback data)",
                confidence=0.5
            ),
            RiskFactor(
                name="inflation_rate",
                category="inflation",
                value=3.2,
                weight=0.12,
                normalized_value=0.6,
                description="Estimated inflation rate: 3.2% (fallback data)",
                confidence=0.4
            ),
            RiskFactor(
                name="federal_funds_rate",
                category="interest_rates",
                value=5.25,
                weight=0.10,
                normalized_value=0.7,
                description="Federal funds rate: 5.25% (fallback data)",
                confidence=0.4
            )
        ]
    
    def _convert_economic_features_to_risk_factors(self, economic_features) -> List[RiskFactor]:
        """Convert economic features to risk factors for scoring."""
        factors = []
        
        # Map economic features to risk factors with proper scaling
        feature_to_risk_map = {
            # Employment indicators
            "unemployment_rate_trend_slope": {
                "name": "unemployment_rate",
                "category": "employment", 
                "weight": 0.15,
                "description": "Unemployment rate trend"
            },
            "unemployment_rate_growth_12m": {
                "name": "unemployment_change",
                "category": "employment",
                "weight": 0.08,
                "description": "12-month unemployment rate change"
            },
            
            # Interest rate indicators
            "fed_funds_rate": {
                "name": "federal_funds_rate",
                "category": "interest_rates",
                "weight": 0.10,
                "description": "Federal funds rate level"
            },
            "fed_funds_rate_volatility_6m": {
                "name": "rate_volatility",
                "category": "interest_rates", 
                "weight": 0.08,
                "description": "Interest rate volatility"
            },
            
            # Inflation indicators
            "cpi_core_growth_12m": {
                "name": "inflation_rate",
                "category": "inflation",
                "weight": 0.12,
                "description": "Core inflation rate"
            },
            "cpi_core_volatility_6m": {
                "name": "inflation_volatility",
                "category": "inflation",
                "weight": 0.06,
                "description": "Inflation volatility"
            },
            
            # Growth indicators
            "gdp_growth_growth_12m": {
                "name": "gdp_growth",
                "category": "economic_growth",
                "weight": 0.15,
                "description": "GDP growth rate"
            },
            "gdp_growth_momentum_12m": {
                "name": "gdp_momentum",
                "category": "economic_growth",
                "weight": 0.08,
                "description": "GDP growth momentum"
            },
            
            # Financial stress
            "term_spread": {
                "name": "yield_curve_spread",
                "category": "financial_stress",
                "weight": 0.18,
                "description": "Yield curve spread"
            }
        }
        
        # Convert features to risk factors
        for feature_name, feature_value in economic_features.features.items():
            if feature_name in feature_to_risk_map and feature_value is not None:
                risk_config = feature_to_risk_map[feature_name]
                
                try:
                    # Normalize feature value to 0-1 risk scale
                    normalized_risk = self._normalize_economic_feature(feature_name, feature_value)
                    
                    factors.append(RiskFactor(
                        name=risk_config["name"],
                        category=risk_config["category"],
                        value=float(feature_value),
                        weight=risk_config["weight"],
                        normalized_value=normalized_risk,
                        description=f"{risk_config['description']}: {feature_value:.2f}",
                        confidence=economic_features.data_quality_score
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to convert feature {feature_name}: {e}")
                    continue
        
        # If we have too few factors, add some basic ones
        if len(factors) < 3:
            logger.warning(f"Only {len(factors)} risk factors from economic features, adding basic factors")
            # Add basic factors from raw data if available
            basic_features = {
                "unemployment_rate": economic_features.features.get("unemployment_rate"),
                "fed_funds_rate": economic_features.features.get("fed_funds_rate"), 
                "gdp_growth": economic_features.features.get("gdp_growth")
            }
            
            for name, value in basic_features.items():
                if value is not None and not any(f.name == name for f in factors):
                    config = self.risk_factors_config.get(name, {})
                    factors.append(RiskFactor(
                        name=name,
                        category=config.get("category", "economic"),
                        value=float(value),
                        weight=config.get("weight", 0.1),
                        normalized_value=min(max(abs(value) / 10.0, 0.0), 1.0),  # Simple normalization
                        description=f"{name}: {value:.2f}",
                        confidence=economic_features.data_quality_score * 0.8
                    ))
        
        logger.info(f"Converted {len(factors)} economic features to risk factors")
        return factors
    
    def _normalize_economic_feature(self, feature_name: str, value: float) -> float:
        """Normalize economic feature value to 0-1 risk scale."""
        # Define normalization rules for different feature types
        normalization_rules = {
            # Unemployment - higher is riskier
            "unemployment_rate_trend_slope": {"min": -2.0, "max": 2.0, "invert": False},
            "unemployment_rate_growth_12m": {"min": -5.0, "max": 10.0, "invert": False},
            
            # Interest rates - extreme values are riskier
            "fed_funds_rate": {"min": 0.0, "max": 8.0, "invert": False, "target": 2.5},
            "fed_funds_rate_volatility_6m": {"min": 0.0, "max": 3.0, "invert": False},
            
            # Inflation - deviation from target is riskier
            "cpi_core_growth_12m": {"min": -2.0, "max": 6.0, "invert": False, "target": 2.0},
            "cpi_core_volatility_6m": {"min": 0.0, "max": 2.0, "invert": False},
            
            # GDP growth - lower is riskier
            "gdp_growth_growth_12m": {"min": -5.0, "max": 5.0, "invert": True},
            "gdp_growth_momentum_12m": {"min": -10.0, "max": 10.0, "invert": True},
            
            # Term spread - inverted curve is risky
            "term_spread": {"min": -2.0, "max": 3.0, "invert": True}
        }
        
        rule = normalization_rules.get(feature_name)
        if not rule:
            # Default normalization
            return min(max(abs(value) / 10.0, 0.0), 1.0)
        
        # Clamp value to range
        clamped_value = max(rule["min"], min(rule["max"], value))
        
        # Handle target-based normalization (deviation from target)
        if "target" in rule:
            deviation = abs(clamped_value - rule["target"])
            max_deviation = max(abs(rule["max"] - rule["target"]), abs(rule["min"] - rule["target"]))
            normalized = deviation / max_deviation if max_deviation > 0 else 0.0
        else:
            # Linear normalization
            normalized = (clamped_value - rule["min"]) / (rule["max"] - rule["min"])
            if rule.get("invert", False):
                normalized = 1.0 - normalized
        
        return min(max(normalized, 0.0), 1.0)
    
    async def update_with_latest_data(self) -> bool:
        """
        Update risk scoring model with latest data.
        
        Returns:
            True if update was successful
        """
        try:
            # Recalculate risk score with fresh data
            risk_score = await self.calculate_risk_score(force_refresh=True)
            
            if risk_score:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error updating risk scorer: {e}")
            return False


# Alias for backward compatibility
RiskScorer = BasicRiskScorer