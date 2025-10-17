"""
Financial Feature Engineering Module

Processes and engineers features from financial data sources including
banking, credit, and financial market indicators for risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager
from src.data.sources.fdic import FdicDataFetcher

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class FinancialFeatures:
    """Container for financial feature data"""
    bank_stability_score: float
    credit_stress_index: float
    liquidity_ratio: float
    capital_adequacy_ratio: float
    non_performing_loans_ratio: float
    interest_rate_risk: float
    systemic_risk_score: float
    market_volatility: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class FinancialFeatureEngineer:
    """
    Feature engineering for financial risk indicators.
    
    Processes banking data, credit metrics, and financial market indicators
    to create features for risk prediction models.
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.fdic_source = FdicDataFetcher()
        self.feature_cache_ttl = 3600  # 1 hour
        
    async def extract_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        institution_ids: Optional[List[str]] = None
    ) -> FinancialFeatures:
        """
        Extract comprehensive financial features for risk assessment.
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction  
            institution_ids: List of specific institution IDs to analyze
            
        Returns:
            FinancialFeatures object with processed features
        """
        cache_key = f"financial_features_{start_date}_{end_date}_{hash(str(institution_ids))}"
        
        # Check cache first
        cached_features = await self.cache.get(cache_key)
        if cached_features:
            logger.info("Retrieved financial features from cache")
            return FinancialFeatures(**cached_features)
            
        logger.info("Extracting financial features from data sources")
        
        try:
            # Extract banking data
            banking_data = await self._extract_banking_features(start_date, end_date, institution_ids)
            
            # Extract credit metrics
            credit_data = await self._extract_credit_features(start_date, end_date)
            
            # Extract market indicators
            market_data = await self._extract_market_features(start_date, end_date)
            
            # Combine and process features
            features = self._combine_financial_features(banking_data, credit_data, market_data)
            
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(features)
            
            # Create feature object
            financial_features = FinancialFeatures(
                bank_stability_score=composite_scores['bank_stability'],
                credit_stress_index=composite_scores['credit_stress'],
                liquidity_ratio=composite_scores['liquidity'],
                capital_adequacy_ratio=composite_scores['capital_adequacy'],
                non_performing_loans_ratio=composite_scores['npl_ratio'],
                interest_rate_risk=composite_scores['interest_rate_risk'],
                systemic_risk_score=composite_scores['systemic_risk'],
                market_volatility=composite_scores['market_volatility'],
                features=features,
                metadata={
                    'extraction_time': datetime.utcnow().isoformat(),
                    'data_sources': ['fdic', 'fred'],
                    'institution_count': len(institution_ids) if institution_ids else 'all',
                    'date_range': f"{start_date} to {end_date}"
                }
            )
            
            # Cache results
            await self.cache.set(
                cache_key,
                financial_features.__dict__,
                ttl=self.feature_cache_ttl
            )
            
            return financial_features
            
        except Exception as e:
            logger.error(f"Error extracting financial features: {e}")
            # Return fallback features
            return await self._get_fallback_features()
    
    async def _extract_banking_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        institution_ids: Optional[List[str]]
    ) -> Dict[str, float]:
        """Extract features from banking data sources"""
        features = {}
        
        try:
            # Get FDIC banking data
            banking_data = await self.fdic_source.get_call_reports(
                start_date=start_date,
                end_date=end_date,
                institution_ids=institution_ids
            )
            
            if banking_data.empty:
                logger.warning("No banking data available")
                return self._get_default_banking_features()
                
            # Calculate banking features
            features.update({
                'total_assets_growth': self._calculate_growth_rate(banking_data, 'total_assets'),
                'deposits_growth': self._calculate_growth_rate(banking_data, 'total_deposits'),
                'loans_growth': self._calculate_growth_rate(banking_data, 'total_loans'),
                'tier1_capital_ratio': banking_data['tier1_capital_ratio'].mean(),
                'leverage_ratio': banking_data['leverage_ratio'].mean(),
                'roa': banking_data['return_on_assets'].mean(),
                'roe': banking_data['return_on_equity'].mean(),
                'net_interest_margin': banking_data['net_interest_margin'].mean(),
                'efficiency_ratio': banking_data['efficiency_ratio'].mean(),
                'loan_loss_provision_ratio': banking_data['loan_loss_provisions'].sum() / banking_data['total_loans'].sum(),
                'asset_quality_index': self._calculate_asset_quality_index(banking_data)
            })
            
        except Exception as e:
            logger.error(f"Error extracting banking features: {e}")
            features.update(self._get_default_banking_features())
            
        return features
    
    async def _extract_credit_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, float]:
        """Extract credit market features"""
        features = {}
        
        try:
            # Credit spreads and yield curve data would come from FRED
            features.update({
                'credit_spread_investment_grade': 0.0,  # Placeholder - would pull from FRED
                'credit_spread_high_yield': 0.0,
                'term_spread': 0.0,
                'treasury_10y_yield': 0.0,
                'mortgage_rates': 0.0,
                'corporate_bond_issuance': 0.0,
                'credit_default_swap_index': 0.0,
                'loan_officer_survey_standards': 0.0
            })
            
        except Exception as e:
            logger.error(f"Error extracting credit features: {e}")
            features.update(self._get_default_credit_features())
            
        return features
    
    async def _extract_market_features(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, float]:
        """Extract financial market features"""
        features = {}
        
        try:
            # Market indicators
            features.update({
                'vix_level': 0.0,  # Volatility index
                'dollar_index': 0.0,
                'commodity_prices_index': 0.0,
                'equity_market_cap_gdp': 0.0,
                'bond_market_volatility': 0.0,
                'fx_volatility': 0.0,
                'gold_prices': 0.0,
                'oil_prices': 0.0
            })
            
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            features.update(self._get_default_market_features())
            
        return features
    
    def _combine_financial_features(
        self,
        banking_data: Dict[str, float],
        credit_data: Dict[str, float],
        market_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine and normalize financial features"""
        all_features = {}
        all_features.update(banking_data)
        all_features.update(credit_data)
        all_features.update(market_data)
        
        # Add interaction features
        all_features.update({
            'credit_asset_interaction': banking_data.get('total_assets_growth', 0) * credit_data.get('credit_spread_investment_grade', 0),
            'leverage_volatility_interaction': banking_data.get('leverage_ratio', 0) * market_data.get('vix_level', 0),
            'margin_spread_interaction': banking_data.get('net_interest_margin', 0) * credit_data.get('term_spread', 0)
        })
        
        return all_features
    
    def _calculate_composite_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite risk scores from features"""
        return {
            'bank_stability': min(100, max(0, 
                features.get('tier1_capital_ratio', 0) * 10 + 
                features.get('roa', 0) * 50 - 
                features.get('loan_loss_provision_ratio', 0) * 100
            )),
            'credit_stress': min(100, max(0,
                features.get('credit_spread_high_yield', 0) * 10 +
                features.get('credit_spread_investment_grade', 0) * 20
            )),
            'liquidity': min(100, max(0,
                100 - features.get('efficiency_ratio', 100)
            )),
            'capital_adequacy': features.get('tier1_capital_ratio', 0),
            'npl_ratio': features.get('loan_loss_provision_ratio', 0) * 100,
            'interest_rate_risk': abs(features.get('term_spread', 0)) * 10,
            'systemic_risk': min(100, max(0,
                features.get('vix_level', 0) +
                features.get('credit_spread_high_yield', 0) * 5
            )),
            'market_volatility': features.get('vix_level', 0)
        }
    
    def _calculate_growth_rate(self, data: pd.DataFrame, column: str) -> float:
        """Calculate growth rate for a given column"""
        if data.empty or column not in data.columns:
            return 0.0
        
        sorted_data = data.sort_values('reporting_date')
        if len(sorted_data) < 2:
            return 0.0
            
        first_value = sorted_data[column].iloc[0]
        last_value = sorted_data[column].iloc[-1]
        
        if first_value == 0:
            return 0.0
            
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_asset_quality_index(self, data: pd.DataFrame) -> float:
        """Calculate asset quality composite index"""
        if data.empty:
            return 50.0  # Neutral score
            
        # Weighted combination of asset quality metrics
        weights = {
            'loan_loss_provisions': -0.3,
            'past_due_loans': -0.3,
            'nonaccrual_loans': -0.4
        }
        
        score = 100.0
        for metric, weight in weights.items():
            if metric in data.columns:
                ratio = data[metric].sum() / data['total_loans'].sum() if data['total_loans'].sum() > 0 else 0
                score += weight * ratio * 100
                
        return max(0, min(100, score))
    
    def _get_default_banking_features(self) -> Dict[str, float]:
        """Return default banking features when data unavailable"""
        return {
            'total_assets_growth': 0.0,
            'deposits_growth': 0.0,
            'loans_growth': 0.0,
            'tier1_capital_ratio': 12.0,  # Typical regulatory minimum
            'leverage_ratio': 5.0,
            'roa': 1.0,
            'roe': 10.0,
            'net_interest_margin': 3.0,
            'efficiency_ratio': 60.0,
            'loan_loss_provision_ratio': 0.5,
            'asset_quality_index': 75.0
        }
    
    def _get_default_credit_features(self) -> Dict[str, float]:
        """Return default credit features when data unavailable"""
        return {
            'credit_spread_investment_grade': 1.0,
            'credit_spread_high_yield': 4.0,
            'term_spread': 1.5,
            'treasury_10y_yield': 4.0,
            'mortgage_rates': 6.5,
            'corporate_bond_issuance': 0.0,
            'credit_default_swap_index': 100.0,
            'loan_officer_survey_standards': 0.0
        }
    
    def _get_default_market_features(self) -> Dict[str, float]:
        """Return default market features when data unavailable"""
        return {
            'vix_level': 20.0,
            'dollar_index': 100.0,
            'commodity_prices_index': 100.0,
            'equity_market_cap_gdp': 150.0,
            'bond_market_volatility': 5.0,
            'fx_volatility': 10.0,
            'gold_prices': 2000.0,
            'oil_prices': 75.0
        }
    
    async def _get_fallback_features(self) -> FinancialFeatures:
        """Return fallback features when extraction fails"""
        default_features = {}
        default_features.update(self._get_default_banking_features())
        default_features.update(self._get_default_credit_features())
        default_features.update(self._get_default_market_features())
        
        composite_scores = self._calculate_composite_scores(default_features)
        
        return FinancialFeatures(
            bank_stability_score=composite_scores['bank_stability'],
            credit_stress_index=composite_scores['credit_stress'],
            liquidity_ratio=composite_scores['liquidity'],
            capital_adequacy_ratio=composite_scores['capital_adequacy'],
            non_performing_loans_ratio=composite_scores['npl_ratio'],
            interest_rate_risk=composite_scores['interest_rate_risk'],
            systemic_risk_score=composite_scores['systemic_risk'],
            market_volatility=composite_scores['market_volatility'],
            features=default_features,
            metadata={
                'extraction_time': datetime.utcnow().isoformat(),
                'data_sources': ['fallback'],
                'institution_count': 'fallback',
                'date_range': 'fallback'
            }
        )


# Feature validation utilities
def validate_financial_features(features: FinancialFeatures) -> bool:
    """Validate financial features for completeness and ranges"""
    try:
        # Check required fields
        required_fields = [
            'bank_stability_score', 'credit_stress_index', 'liquidity_ratio',
            'capital_adequacy_ratio', 'non_performing_loans_ratio'
        ]
        
        for field in required_fields:
            if not hasattr(features, field):
                return False
                
        # Check value ranges
        if not (0 <= features.bank_stability_score <= 100):
            return False
            
        if features.capital_adequacy_ratio < 0:
            return False
            
        return True
        
    except Exception:
        return False


def calculate_financial_risk_score(features: FinancialFeatures) -> float:
    """Calculate overall financial risk score from features"""
    try:
        # Weighted combination of key risk indicators
        weights = {
            'bank_stability_score': 0.25,
            'credit_stress_index': 0.20,
            'systemic_risk_score': 0.20,
            'market_volatility': 0.15,
            'interest_rate_risk': 0.10,
            'non_performing_loans_ratio': 0.10
        }
        
        risk_score = 0.0
        for indicator, weight in weights.items():
            value = getattr(features, indicator, 0)
            # Normalize to 0-100 scale where higher = more risk
            if indicator == 'bank_stability_score':
                value = 100 - value  # Invert stability to risk
            risk_score += weight * value
            
        return min(100, max(0, risk_score))
        
    except Exception:
        return 50.0  # Neutral risk score on error