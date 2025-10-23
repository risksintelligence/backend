#!/usr/bin/env python3
"""
Populate sample data for RiskX Platform.
Creates sample risk scores, factors, and alerts for testing.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import AsyncSessionLocal, engine
from src.data.models.risk_models import RiskScore, RiskFactor, Alert
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_risk_scores():
    """Create sample risk scores for the last 30 days."""
    async with AsyncSessionLocal() as session:
        logger.info("Creating sample risk scores...")
        
        # Create risk scores for last 30 days
        scores = []
        base_date = datetime.utcnow() - timedelta(days=30)
        
        for i in range(30):
            score_date = base_date + timedelta(days=i)
            
            # Simulate varying risk scores
            base_score = 75.0 + (i % 10) - 5  # Varies between 70-80
            
            risk_score = RiskScore(
                overall_score=base_score,
                confidence=0.85 + (i % 3) * 0.05,  # 0.85-0.95
                trend='rising' if i % 3 == 0 else 'falling' if i % 3 == 1 else 'stable',
                economic_score=base_score - 5 + (i % 5),
                market_score=base_score + 3 - (i % 4),
                geopolitical_score=base_score - 2 + (i % 3),
                technical_score=base_score + 1 - (i % 2),
                timestamp=score_date,
                calculation_method="multi_factor_weighted",
                data_sources=["FRED", "BEA", "BLS", "Market_Data"]
            )
            scores.append(risk_score)
        
        session.add_all(scores)
        await session.commit()
        logger.info(f"Created {len(scores)} risk scores")


async def create_sample_risk_factors():
    """Create sample risk factors."""
    async with AsyncSessionLocal() as session:
        logger.info("Creating sample risk factors...")
        
        factors = [
            RiskFactor(
                name="GDP Growth Rate",
                category="economic",
                description="Quarterly GDP growth rate indicating economic health",
                current_value=2.1,
                current_score=72.0,
                impact_level="high",
                weight=0.25,
                data_source="BEA",
                series_id="GDP",
                last_updated=datetime.utcnow(),
                threshold_low=1.0,
                threshold_high=4.0,
                is_active=True
            ),
            RiskFactor(
                name="Unemployment Rate", 
                category="economic",
                description="National unemployment rate from Bureau of Labor Statistics",
                current_value=3.7,
                current_score=78.0,
                impact_level="high",
                weight=0.20,
                data_source="BLS",
                series_id="UNRATE",
                last_updated=datetime.utcnow(),
                threshold_low=2.0,
                threshold_high=6.0,
                is_active=True
            ),
            RiskFactor(
                name="Core Inflation Rate",
                category="economic", 
                description="Core CPI excluding food and energy",
                current_value=3.2,
                current_score=68.0,
                impact_level="medium",
                weight=0.15,
                data_source="FRED",
                series_id="CPILFESL",
                last_updated=datetime.utcnow(),
                threshold_low=1.5,
                threshold_high=4.0,
                is_active=True
            ),
            RiskFactor(
                name="Market Volatility Index",
                category="market",
                description="VIX volatility index measuring market fear",
                current_value=18.5,
                current_score=75.0,
                impact_level="high",
                weight=0.18,
                data_source="Market_Data",
                series_id="VIX",
                last_updated=datetime.utcnow(),
                threshold_low=10.0,
                threshold_high=30.0,
                is_active=True
            ),
            RiskFactor(
                name="Federal Funds Rate",
                category="market",
                description="Federal Reserve interest rate policy indicator",
                current_value=5.25,
                current_score=73.0,
                impact_level="high",
                weight=0.22,
                data_source="FRED",
                series_id="FEDFUNDS",
                last_updated=datetime.utcnow(),
                threshold_low=0.0,
                threshold_high=7.0,
                is_active=True
            ),
            RiskFactor(
                name="Geopolitical Tension Index",
                category="geopolitical",
                description="Composite measure of global geopolitical risks",
                current_value=6.2,
                current_score=70.0,
                impact_level="medium",
                weight=0.12,
                data_source="Analytics",
                series_id="GEOPOLITICAL",
                last_updated=datetime.utcnow(),
                threshold_low=3.0,
                threshold_high=8.0,
                is_active=True
            ),
            RiskFactor(
                name="Supply Chain Disruption Index",
                category="technical",
                description="Measure of global supply chain stress",
                current_value=4.8,
                current_score=76.0,
                impact_level="medium",
                weight=0.10,
                data_source="Analytics",
                series_id="SUPPLY_CHAIN",
                last_updated=datetime.utcnow(),
                threshold_low=2.0,
                threshold_high=7.0,
                is_active=True
            ),
            RiskFactor(
                name="Credit Spread",
                category="market",
                description="Corporate bond credit spread indicating market stress",
                current_value=1.85,
                current_score=74.0,
                impact_level="medium",
                weight=0.08,
                data_source="Market_Data",
                series_id="CREDIT_SPREAD",
                last_updated=datetime.utcnow(),
                threshold_low=0.5,
                threshold_high=3.0,
                is_active=True
            )
        ]
        
        session.add_all(factors)
        await session.commit()
        logger.info(f"Created {len(factors)} risk factors")


async def create_sample_alerts():
    """Create sample alerts."""
    async with AsyncSessionLocal() as session:
        logger.info("Creating sample alerts...")
        
        alerts = [
            Alert(
                alert_type="threshold_breach",
                severity="medium",
                title="Inflation Rate Above Target",
                message="Core inflation rate has exceeded the 3.0% target threshold",
                triggered_by="Core Inflation Rate",
                threshold_value=3.0,
                current_value=3.2,
                triggered_at=datetime.utcnow() - timedelta(hours=2),
                status="active",
                alert_metadata={
                    "factor_id": 3,
                    "trend": "rising",
                    "duration_hours": 2
                }
            ),
            Alert(
                alert_type="trend_change",
                severity="low",
                title="GDP Growth Trend Shift",
                message="GDP growth rate showing declining trend over last quarter",
                triggered_by="GDP Growth Rate",
                threshold_value=2.5,
                current_value=2.1,
                triggered_at=datetime.utcnow() - timedelta(hours=6),
                status="active",
                alert_metadata={
                    "factor_id": 1,
                    "previous_trend": "stable",
                    "new_trend": "declining"
                }
            ),
            Alert(
                alert_type="volatility_spike",
                severity="high", 
                title="Market Volatility Elevated",
                message="VIX index showing elevated volatility above 20",
                triggered_by="Market Volatility Index",
                threshold_value=20.0,
                current_value=18.5,
                triggered_at=datetime.utcnow() - timedelta(minutes=30),
                status="resolved",
                alert_metadata={
                    "factor_id": 4,
                    "spike_duration": "45 minutes",
                    "max_value": 22.3
                }
            ),
            Alert(
                alert_type="correlation_anomaly",
                severity="medium",
                title="Unusual Factor Correlation",
                message="Geopolitical factors showing unusual correlation with market indicators",
                triggered_by="Correlation Analysis",
                threshold_value=0.7,
                current_value=0.85,
                triggered_at=datetime.utcnow() - timedelta(hours=1),
                status="active",
                alert_metadata={
                    "factors": ["geopolitical", "market"],
                    "correlation_window": "7 days",
                    "significance": "high"
                }
            )
        ]
        
        session.add_all(alerts)
        await session.commit()
        logger.info(f"Created {len(alerts)} alerts")


async def populate_all_sample_data():
    """Populate all sample data."""
    try:
        logger.info("Starting sample data population...")
        
        await create_sample_risk_scores()
        await create_sample_risk_factors()
        await create_sample_alerts()
        
        logger.info("✅ Sample data population completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error populating sample data: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(populate_all_sample_data())