from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from src.core.database import get_db
from src.data.models.risk_models import RiskScore, RiskFactor, Alert
from src.cache.cache_manager import IntelligentCacheManager
from src.core.dependencies import get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["risk_analytics"])


@router.get("/risk/current")
async def get_current_risk_score(
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """Get the most current risk score."""
    
    # Try cache first
    cache_key = "analytics:current_risk_score"
    cached_data = await cache.get(cache_key, max_age_seconds=300)
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get from database
        result = await db.execute(
            select(RiskScore)
            .order_by(desc(RiskScore.timestamp))
            .limit(1)
        )
        latest_score = result.scalar_one_or_none()
        
        if latest_score:
            data = {
                "overall_score": latest_score.overall_score,
                "confidence": latest_score.confidence,
                "trend": latest_score.trend,
                "components": {
                    "economic": latest_score.economic_score,
                    "market": latest_score.market_score,
                    "geopolitical": latest_score.geopolitical_score,
                    "technical": latest_score.technical_score
                },
                "timestamp": latest_score.timestamp.isoformat(),
                "calculation_method": latest_score.calculation_method,
                "data_sources": latest_score.data_sources
            }
            
            # Cache the result
            await cache.set(cache_key, data, ttl_seconds=300)
            
            return {
                "status": "success",
                "data": data,
                "source": "database",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "status": "not_found",
            "message": "No risk scores available",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching current risk score: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk score")


@router.get("/risk/history")
async def get_risk_score_history(
    days: int = 30,
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """Get historical risk scores."""
    
    cache_key = f"analytics:risk_history:{days}days"
    cached_data = await cache.get(cache_key, max_age_seconds=1800)  # 30 min cache
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(RiskScore)
            .where(RiskScore.timestamp >= start_date)
            .order_by(desc(RiskScore.timestamp))
            .limit(100)
        )
        scores = result.scalars().all()
        
        history = []
        for score in scores:
            history.append({
                "timestamp": score.timestamp.isoformat(),
                "overall_score": score.overall_score,
                "confidence": score.confidence,
                "trend": score.trend,
                "components": {
                    "economic": score.economic_score,
                    "market": score.market_score,
                    "geopolitical": score.geopolitical_score,
                    "technical": score.technical_score
                }
            })
        
        data = {
            "period_days": days,
            "count": len(history),
            "scores": history
        }
        
        # Cache the result
        await cache.set(cache_key, data, ttl_seconds=1800)
        
        return {
            "status": "success",
            "data": data,
            "source": "database",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching risk history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk history")


@router.get("/factors")
async def get_risk_factors(
    category: Optional[str] = None,
    active_only: bool = True,
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """Get risk factors with optional category filtering."""
    
    cache_key = f"analytics:factors:{category}:{active_only}"
    cached_data = await cache.get(cache_key, max_age_seconds=600)  # 10 min cache
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        query = select(RiskFactor)
        
        if active_only:
            query = query.where(RiskFactor.is_active == True)
        
        if category:
            query = query.where(RiskFactor.category == category)
        
        query = query.order_by(desc(RiskFactor.current_score))
        
        result = await db.execute(query)
        factors = result.scalars().all()
        
        factors_data = []
        for factor in factors:
            factors_data.append({
                "id": factor.id,
                "name": factor.name,
                "category": factor.category,
                "description": factor.description,
                "current_value": factor.current_value,
                "current_score": factor.current_score,
                "impact_level": factor.impact_level,
                "weight": factor.weight,
                "data_source": factor.data_source,
                "series_id": factor.series_id,
                "last_updated": factor.last_updated.isoformat(),
                "thresholds": {
                    "low": factor.threshold_low,
                    "high": factor.threshold_high
                }
            })
        
        data = {
            "category_filter": category,
            "active_only": active_only,
            "count": len(factors_data),
            "factors": factors_data
        }
        
        # Cache the result
        await cache.set(cache_key, data, ttl_seconds=600)
        
        return {
            "status": "success",
            "data": data,
            "source": "database",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching risk factors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk factors")


@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """Get active alerts with optional severity filtering."""
    
    cache_key = f"analytics:alerts:{severity}:{limit}"
    cached_data = await cache.get(cache_key, max_age_seconds=120)  # 2 min cache
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        query = select(Alert).where(Alert.status == 'active')
        
        if severity:
            query = query.where(Alert.severity == severity)
        
        query = query.order_by(desc(Alert.triggered_at)).limit(limit)
        
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                "id": alert.id,
                "type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "triggered_by": alert.triggered_by,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.alert_metadata
            })
        
        data = {
            "severity_filter": severity,
            "count": len(alerts_data),
            "alerts": alerts_data
        }
        
        # Cache the result
        await cache.set(cache_key, data, ttl_seconds=120)
        
        return {
            "status": "success",
            "data": data,
            "source": "database",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")


@router.get("/dashboard")
async def get_dashboard_data(
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive dashboard data."""
    
    cache_key = "analytics:dashboard"
    cached_data = await cache.get(cache_key, max_age_seconds=300)  # 5 min cache
    
    if cached_data:
        return {
            "status": "success",
            "data": cached_data,
            "source": "cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get current risk score
        risk_result = await db.execute(
            select(RiskScore)
            .order_by(desc(RiskScore.timestamp))
            .limit(1)
        )
        current_risk = risk_result.scalar_one_or_none()
        
        # Get active alerts count by severity
        alert_result = await db.execute(
            select(Alert.severity, Alert.id)
            .where(Alert.status == 'active')
        )
        alerts = alert_result.all()
        
        alert_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for alert in alerts:
            severity = alert[0]
            if severity in alert_counts:
                alert_counts[severity] += 1
        
        # Get top risk factors
        factors_result = await db.execute(
            select(RiskFactor)
            .where(RiskFactor.is_active == True)
            .order_by(desc(RiskFactor.current_score))
            .limit(5)
        )
        top_factors = factors_result.scalars().all()
        
        factor_summary = []
        for factor in top_factors:
            factor_summary.append({
                "name": factor.name,
                "category": factor.category,
                "score": factor.current_score,
                "impact": factor.impact_level
            })
        
        dashboard_data = {
            "current_risk": {
                "overall_score": current_risk.overall_score if current_risk else None,
                "confidence": current_risk.confidence if current_risk else None,
                "trend": current_risk.trend if current_risk else None,
                "timestamp": current_risk.timestamp.isoformat() if current_risk else None
            } if current_risk else None,
            "alert_summary": {
                "total_active": sum(alert_counts.values()),
                "by_severity": alert_counts
            },
            "top_risk_factors": factor_summary,
            "system_status": "operational"
        }
        
        # Cache the result
        await cache.set(cache_key, dashboard_data, ttl_seconds=300)
        
        return {
            "status": "success",
            "data": dashboard_data,
            "source": "database",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")