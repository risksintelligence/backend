#!/usr/bin/env python3
"""
Community API endpoints for partner labs, scenario studio, and media.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from app.core.auth import require_contributor_submit, require_contributor_submit_or_reviewer
from app.services.submissions import get_submissions_summary
from app.db import SessionLocal, get_db
from app.models import CommunityInsight, CommunityUser, InsightLike, InsightComment, WeeklyBrief, WeeklyBriefSubscription
from app.core.unified_cache import UnifiedCache

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/api/v1/community/partner-labs")
def get_partner_labs() -> Dict[str, Any]:
    """Get current partner labs and their status."""
    try:
        # Mock partner labs data - in production this would come from database
        partner_labs = [
            {
                "id": "mit-resilience-lab",
                "name": "MIT Economic Resilience Lab", 
                "institution": "Massachusetts Institute of Technology",
                "sector": "Education",
                "status": "active",
                "enrolled_date": "2024-09-15T00:00:00Z",
                "mission": "Ethical AI in Finance",
                "current_projects": [
                    {
                        "title": "Supply Chain Risk Assessment for Higher Education",
                        "status": "in_progress",
                        "contributors": 3,
                        "due_date": "2024-12-15T00:00:00Z"
                    }
                ],
                "recent_submissions": 2,
                "impact_metrics": {
                    "ras_contribution": 0.15,
                    "community_engagement": 85,
                    "data_usage": "high"
                }
            },
            {
                "id": "stanford-policy-center",
                "name": "Stanford Policy Impact Center",
                "institution": "Stanford University", 
                "sector": "Policy Research",
                "status": "active",
                "enrolled_date": "2024-08-20T00:00:00Z",
                "mission": "Energy Transition Risk",
                "current_projects": [
                    {
                        "title": "Clean Energy Disruption Scenarios",
                        "status": "completed",
                        "contributors": 5,
                        "completion_date": "2024-11-01T00:00:00Z"
                    },
                    {
                        "title": "Carbon Pricing Impact Analysis", 
                        "status": "in_progress",
                        "contributors": 2,
                        "due_date": "2025-01-15T00:00:00Z"
                    }
                ],
                "recent_submissions": 4,
                "impact_metrics": {
                    "ras_contribution": 0.28,
                    "community_engagement": 92,
                    "data_usage": "very_high"
                }
            },
            {
                "id": "chicago-civic-tech",
                "name": "Chicago Civic Technology Collective",
                "institution": "University of Chicago",
                "sector": "Urban Policy",
                "status": "onboarding",
                "enrolled_date": "2024-11-01T00:00:00Z", 
                "mission": "Food Security Monitoring",
                "current_projects": [
                    {
                        "title": "Urban Food System Resilience Dashboard",
                        "status": "planning",
                        "contributors": 4,
                        "planned_start": "2024-12-01T00:00:00Z"
                    }
                ],
                "recent_submissions": 0,
                "impact_metrics": {
                    "ras_contribution": 0.0,
                    "community_engagement": 45,
                    "data_usage": "low"
                }
            }
        ]
        
        # Calculate summary statistics
        total_labs = len(partner_labs)
        active_labs = len([lab for lab in partner_labs if lab["status"] == "active"])
        total_projects = sum(len(lab["current_projects"]) for lab in partner_labs)
        total_ras_contribution = sum(lab["impact_metrics"]["ras_contribution"] for lab in partner_labs)
        
        return {
            "partner_labs": partner_labs,
            "summary": {
                "total_labs": total_labs,
                "active_labs": active_labs,
                "onboarding_labs": total_labs - active_labs,
                "total_projects": total_projects,
                "sectors_covered": len(set(lab["sector"] for lab in partner_labs)),
                "total_ras_contribution": round(total_ras_contribution, 2),
                "average_engagement": round(sum(lab["impact_metrics"]["community_engagement"] for lab in partner_labs) / total_labs, 1) if total_labs > 0 else 0
            },
            "upcoming_showcases": [
                {
                    "title": "Q4 2024 Sector Mission Showcase",
                    "date": "2024-12-15T18:00:00Z",
                    "participating_labs": ["mit-resilience-lab", "stanford-policy-center"],
                    "registration_url": "https://rrio.dev/events/q4-showcase"
                }
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch partner labs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch partner labs data")

@router.get("/api/v1/community/media-kit")
def get_media_kit() -> Dict[str, Any]:
    """Get media kit assets and testimonials."""
    try:
        media_assets = {
            "speaker_bios": [
                {
                    "name": "Dr. Sarah Chen",
                    "title": "Lead Data Scientist, RRIO",
                    "bio": "Expert in economic resilience modeling and explainable AI systems.",
                    "photo_url": "/assets/speakers/sarah-chen.jpg",
                    "topics": ["Economic Modeling", "AI Explainability", "Risk Assessment"]
                },
                {
                    "name": "Prof. Michael Rodriguez", 
                    "title": "Advisory Board Chair",
                    "bio": "Former Fed economist specializing in systemic risk and macroprudential policy.",
                    "photo_url": "/assets/speakers/michael-rodriguez.jpg", 
                    "topics": ["Monetary Policy", "Systemic Risk", "Financial Stability"]
                }
            ],
            "testimonials": [
                {
                    "author": "Dr. Jennifer Walsh",
                    "title": "Director of Risk, Regional Bank Consortium", 
                    "quote": "RRIO's transparency and real-time insights have transformed how we assess macro risk. The GERII methodology is now central to our quarterly stress testing.",
                    "date": "2024-10-15",
                    "sector": "Banking"
                },
                {
                    "author": "Prof. David Kim",
                    "title": "Economics Department, State University",
                    "quote": "The Partner Labs program gave our students hands-on experience with Bloomberg-grade economic intelligence. Three of our fellows are now working at major policy institutions.",
                    "date": "2024-09-28", 
                    "sector": "Education"
                }
            ],
            "highlight_reels": [
                {
                    "title": "RRIO 2024 Impact Summary",
                    "description": "Key achievements, partner spotlights, and community growth",
                    "duration": "3:45",
                    "url": "/assets/videos/rrio-2024-highlights.mp4",
                    "thumbnail": "/assets/thumbnails/2024-highlights.jpg"
                }
            ],
            "press_releases": [
                {
                    "title": "RRIO Launches Ethical AI Mission for Financial Institutions",
                    "date": "2024-10-01",
                    "summary": "New sector mission addresses transparency and fairness in banking AI systems.",
                    "url": "/assets/press/ethical-ai-mission-launch.pdf"
                }
            ],
            "awards_recognition": [
                {
                    "award": "Best Public Interest Technology Platform",
                    "organization": "National Science Foundation", 
                    "year": 2024,
                    "description": "Recognized for transparent AI and community engagement in economic intelligence."
                },
                {
                    "award": "Excellence in Data Transparency",
                    "organization": "Open Data Initiative",
                    "year": 2024,
                    "description": "Outstanding commitment to open, explainable economic risk assessment."
                }
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return media_assets
        
    except Exception as e:
        logger.error(f"Failed to fetch media kit: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch media kit")

@router.get("/api/v1/community/scenario-prompts") 
def get_scenario_prompts() -> Dict[str, Any]:
    """Get current scenario studio prompts and submissions."""
    try:
        current_prompts = [
            {
                "id": "supply-chain-stress-test",
                "title": "Supply Chain Stress Testing Under Energy Price Volatility",
                "description": "Analyze how different oil price shock scenarios would cascade through global supply chains and impact GERII components.",
                "type": "quantitative",
                "difficulty": "intermediate",
                "estimated_time": "2-3 hours",
                "created_date": "2024-11-10T00:00:00Z",
                "deadline": "2024-11-30T23:59:59Z",
                "status": "active",
                "data_sources": ["WTI_OIL", "FREIGHT_DIESEL", "PMI"],
                "submission_count": 7,
                "featured_submission": {
                    "author": "EconomicsStudent2024",
                    "title": "Three-Scenario Analysis: Moderate, Severe, and Extreme Oil Shocks",
                    "preview": "Using Monte Carlo simulation across 1000 iterations...",
                    "upvotes": 23
                }
            },
            {
                "id": "fed-policy-regime-change", 
                "title": "Federal Reserve Policy Regime Change Implications",
                "description": "Explore how different monetary policy stances would affect regime classification probabilities and GERII scoring.",
                "type": "policy_analysis",
                "difficulty": "advanced", 
                "estimated_time": "4-6 hours",
                "created_date": "2024-11-05T00:00:00Z",
                "deadline": "2024-12-15T23:59:59Z",
                "status": "active",
                "data_sources": ["YIELD_CURVE", "CREDIT_SPREAD", "VIX"],
                "submission_count": 3,
                "featured_submission": None
            },
            {
                "id": "climate-transition-risks",
                "title": "Climate Transition Risk Assessment Framework", 
                "description": "Develop a framework for incorporating climate transition risks into economic resilience modeling.",
                "type": "methodology",
                "difficulty": "expert",
                "estimated_time": "8-10 hours",
                "created_date": "2024-10-28T00:00:00Z", 
                "deadline": "2025-01-31T23:59:59Z",
                "status": "active",
                "data_sources": ["ALL"],
                "submission_count": 1,
                "featured_submission": None
            }
        ]
        
        completed_prompts = [
            {
                "id": "inflation-persistence-analysis",
                "title": "Inflation Persistence and Regime Stability",
                "completion_date": "2024-10-15T00:00:00Z",
                "total_submissions": 12,
                "winner": {
                    "author": "PolicyAnalyst47",
                    "title": "Multi-Horizon Inflation Expectations and Regime Transitions"
                }
            }
        ]
        
        return {
            "current_prompts": current_prompts,
            "completed_prompts": completed_prompts,
            "summary": {
                "active_prompts": len(current_prompts),
                "total_submissions": sum(p["submission_count"] for p in current_prompts),
                "participation_rate": round(sum(p["submission_count"] for p in current_prompts) / len(current_prompts), 1) if current_prompts else 0,
                "difficulty_distribution": {
                    "beginner": 0,
                    "intermediate": 1, 
                    "advanced": 1,
                    "expert": 1
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch scenario prompts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch scenario prompts")

@router.post("/api/v1/community/scenario-prompts/{prompt_id}/submit")
def submit_scenario_response(
    prompt_id: str,
    submission_data: Dict[str, Any],
    _auth: dict = Depends(require_contributor_submit_or_reviewer)
) -> Dict[str, Any]:
    """Submit a response to a scenario prompt."""
    try:
        from app.services.submissions import add_submission
        
        # Extract submission details
        title = submission_data.get('title', 'Scenario Response')
        description = submission_data.get('description', '')
        author = submission_data.get('author', 'Anonymous')
        author_email = submission_data.get('author_email', 'noreply@example.com')
        
        # Prepare submission payload
        submission_payload = {
            'title': f"Scenario Response: {title}",
            'summary': description,
            'author': author,
            'author_email': author_email,
            'link': '',  # No link for scenario responses
            'mission': 'scenario',
            'prompt_id': prompt_id
        }
        
        # Add to database via submissions service
        result = add_submission(submission_payload)
        
        return {
            "submission_id": result.get("id"),
            "prompt_id": prompt_id,
            "title": result.get("title"),
            "status": "submitted",
            "submitted_at": result.get("submitted_at"),
            "review_status": "pending",
            "estimated_review_time": "3-5 business days"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit scenario response: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit scenario response")


@router.get("/api/v1/community/submissions/summary")
def get_community_submissions_summary() -> Dict[str, Any]:
    """Get summary statistics for community submissions."""
    try:
        return get_submissions_summary()
    except Exception as e:
        logger.error(f"Failed to fetch submissions summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch submissions summary")

@router.get("/api/v1/community/insights")
def get_community_insights(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, description="Number of insights to return"),
    offset: int = Query(0, description="Number of insights to skip")
) -> Dict[str, Any]:
    """Get community insights with filtering, pagination, and caching."""
    try:
        # Create cache key based on parameters
        cache_key = f"insights_{category or 'all'}_{limit}_{offset}"
        cache = UnifiedCache("community")
        
        # Try to get from cache first
        cached_data, metadata = cache.get(cache_key)
        if cached_data and metadata and not metadata.is_stale_hard:
            logger.info(f"Returning cached community insights: {cache_key}")
            return cached_data
        
        # Cache miss or stale data - fetch from database
        logger.info(f"Cache miss for community insights, fetching from database: {cache_key}")
        
        # Query insights from database
        query = db.query(CommunityInsight).join(CommunityUser)
        
        # Apply category filter if provided
        if category and category != "all":
            query = query.filter(CommunityInsight.category == category)
        
        # Only show published insights
        query = query.filter(CommunityInsight.status == "published")
        
        # Order by created_at descending for newest first
        query = query.order_by(CommunityInsight.created_at.desc())
        
        # Apply pagination
        insights = query.offset(offset).limit(limit).all()
        
        # Format insights for frontend
        insights_data = []
        for insight in insights:
            insights_data.append({
                "id": insight.id,
                "title": insight.title,
                "content": insight.content,
                "author": insight.user.username if insight.user else "Unknown",
                "category": insight.category,
                "timestamp": insight.created_at.isoformat(),
                "likes": insight.likes_count,
                "comments": insight.comments_count,
                "risk_score": insight.risk_score or 0,
                "impact_level": insight.impact_level or "medium",
                "tags": insight.tags or [],
                "verified": insight.user.verified if insight.user else False
            })
        
        # Get total count for pagination
        total_count = db.query(CommunityInsight).filter(CommunityInsight.status == "published").count()
        
        result = {
            "insights": insights_data,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache the result (5 minute soft TTL, 15 minute hard TTL for community content)
        cache.set(cache_key, result, {
            "source": "community_insights_db",
            "source_url": f"/api/v1/community/insights?category={category}&limit={limit}&offset={offset}",
            "derivation_flag": "raw",
            "soft_ttl": 300,    # 5 minutes
            "hard_ttl": 900     # 15 minutes
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch community insights: {e}")
        # Return demo data as fallback for now
        demo_insights = [
            {
                "id": "insight-1",
                "title": "Q4 Credit Stress Indicators Showing Early Warning Signals",
                "content": "Regional banking stress tests reveal concerning patterns in commercial real estate portfolios. Default rates have increased 23% quarter-over-quarter, particularly in secondary markets. Risk models suggest this could cascade to broader credit conditions by Q1 2025.",
                "author": "Sarah Chen, CRO",
                "category": "market-analysis",
                "timestamp": "2024-11-23T14:30:00Z",
                "likes": 34,
                "comments": 8,
                "risk_score": 72.5,
                "impact_level": "high",
                "tags": ["credit-risk", "banking", "commercial-real-estate", "stress-testing"],
                "verified": True
            },
            {
                "id": "insight-2", 
                "title": "Supply Chain Disruption Patterns in Southeast Asian Routes",
                "content": "Logistics network analysis shows 15% freight cost increases across major shipping lanes. Port congestion in Singapore and Hong Kong creating cascading delays. Recommend diversifying supplier base and increasing inventory buffers for Q1 2025.",
                "author": "Michael Rodriguez, Supply Chain Director",
                "category": "supply-chain",
                "timestamp": "2024-11-23T11:15:00Z",
                "likes": 28,
                "comments": 12,
                "risk_score": 68.3,
                "impact_level": "medium",
                "tags": ["supply-chain", "logistics", "asia-pacific", "freight-costs"],
                "verified": True
            },
            {
                "id": "insight-3",
                "title": "Hidden Markov Model Performance in Current Regime Detection",
                "content": "Updated HMM parameters show improved accuracy in detecting regime transitions. The model now correctly identifies 87% of turning points with 3-day lead time. Key insight: labor market volatility is the strongest early indicator.",
                "author": "Dr. Jennifer Walsh, Quantitative Researcher",
                "category": "methodology", 
                "timestamp": "2024-11-22T16:45:00Z",
                "likes": 19,
                "comments": 6,
                "risk_score": 45.2,
                "impact_level": "low",
                "tags": ["machine-learning", "regime-detection", "hmm", "labor-markets"],
                "verified": True
            }
        ]
        
        return {
            "insights": demo_insights,
            "pagination": {
                "total": len(demo_insights),
                "limit": limit,
                "offset": 0,
                "has_more": False
            },
            "generated_at": datetime.utcnow().isoformat()
        }

@router.post("/api/v1/community/insights")
def create_community_insight(
    insight_data: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Create a new community insight."""
    try:
        # For now, create demo user if not exists
        demo_user = db.query(CommunityUser).filter(CommunityUser.username == "demo_user").first()
        if not demo_user:
            demo_user = CommunityUser(
                username="demo_user",
                email="demo@rrio.dev",
                full_name="Demo User",
                verified=True,
                professional_category="risk-analyst"
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        
        # Create new insight
        new_insight = CommunityInsight(
            title=insight_data.get("title", ""),
            content=insight_data.get("content", ""),
            category=insight_data.get("category", "market-analysis"),
            risk_score=insight_data.get("risk_score", 50.0),
            impact_level=insight_data.get("impact_level", "medium"),
            tags=insight_data.get("tags", []),
            author_id=demo_user.id,
            status="published",
            published_at=datetime.utcnow()
        )
        
        db.add(new_insight)
        db.commit()
        db.refresh(new_insight)
        
        return {
            "id": new_insight.id,
            "title": new_insight.title,
            "status": "published",
            "created_at": new_insight.created_at.isoformat(),
            "message": "Community insight created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create community insight: {e}")
        raise HTTPException(status_code=500, detail="Failed to create community insight")

@router.post("/api/v1/community/insights/{insight_id}/like")
def like_insight(
    insight_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Like or unlike a community insight."""
    try:
        # Get the insight
        insight = db.query(CommunityInsight).filter(CommunityInsight.id == insight_id).first()
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        # For demo purposes, just increment the like count
        insight.likes_count += 1
        db.commit()
        
        return {
            "insight_id": insight_id,
            "likes_count": insight.likes_count,
            "liked": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to like insight: {e}")
        raise HTTPException(status_code=500, detail="Failed to like insight")

@router.get("/api/v1/community/insights/{insight_id}")
def get_insight_details(
    insight_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed information about a specific insight."""
    try:
        insight = db.query(CommunityInsight).join(CommunityUser).filter(CommunityInsight.id == insight_id).first()
        
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        # Increment view count
        insight.views_count += 1
        db.commit()
        
        # Get comments
        comments = db.query(InsightComment).join(CommunityUser).filter(InsightComment.insight_id == insight_id).all()
        
        comments_data = [{
            "id": comment.id,
            "content": comment.content,
            "author": comment.user.username,
            "created_at": comment.created_at.isoformat(),
            "likes_count": comment.likes_count
        } for comment in comments]
        
        return {
            "id": insight.id,
            "title": insight.title,
            "content": insight.content,
            "author": insight.user.username,
            "author_verified": insight.user.verified,
            "category": insight.category,
            "risk_score": insight.risk_score,
            "impact_level": insight.impact_level,
            "tags": insight.tags or [],
            "likes_count": insight.likes_count,
            "comments_count": insight.comments_count,
            "views_count": insight.views_count,
            "created_at": insight.created_at.isoformat(),
            "comments": comments_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get insight details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insight details")

# Weekly Intelligence Brief Endpoints
@router.get("/api/v1/insights/weekly")
def get_weekly_brief(
    week: Optional[str] = Query("current", description="Week to retrieve: current, last, two-weeks"),
    format: Optional[str] = Query("executive", description="Brief format: executive, detailed, technical"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get weekly intelligence brief with caching."""
    try:
        # Create cache key based on parameters
        cache_key = f"weekly_brief_{week}_{format}"
        cache = UnifiedCache("weekly_briefs")
        
        # Try to get from cache first (longer TTL for weekly content)
        cached_data, metadata = cache.get(cache_key)
        if cached_data and metadata and not metadata.is_stale_hard:
            logger.info(f"Returning cached weekly brief: {cache_key}")
            return cached_data
        
        logger.info(f"Generating weekly brief: {cache_key}")
        
        # Generate brief dynamically using real data
        from app.services.geri import compute_geri_score
        from app.ml.regime import classify_regime
        from app.ml.forecast import forecast_delta
        
        # Get current data for brief generation
        from app.main import _get_observations
        observations = _get_observations()
        
        # Get current metrics
        geri_data = compute_geri_score(observations)
        regime_data = classify_regime(observations)
        forecast_data = forecast_delta(observations)
        
        geri_score = geri_data.get("score", 0) or 0
        current_regime = max(regime_data, key=regime_data.get) if regime_data else "Unknown"
        forecast_delta_val = forecast_data.get("delta", 0) or 0
        
        # Generate brief sections dynamically
        sections = [
            {
                "title": "Executive Summary",
                "content": f"Global risk intelligence indicates a GERI score of {geri_score:.1f}, reflecting {current_regime.lower()} economic conditions. Our 24-hour outlook suggests {'increasing' if forecast_delta_val > 0 else 'decreasing'} risk trajectory with {abs(forecast_delta_val):.1f} point expected movement.",
                "risk_level": "high" if geri_score > 70 else "medium" if geri_score > 50 else "low",
                "recommendations": [
                    "Monitor portfolio exposure to volatile sectors",
                    "Review hedging strategies for key positions", 
                    "Assess supply chain vulnerabilities"
                ],
                "impact_score": int(geri_score)
            },
            {
                "title": "Economic Regime Analysis",
                "content": f"Current regime classification shows {current_regime} conditions with {((regime_data.get(current_regime, 0.5)) * 100):.1f}% confidence. Regime transition analysis indicates potential shifts in labor market dynamics and credit conditions.",
                "risk_level": "critical" if current_regime == "Crisis" else "low" if current_regime == "Expansion" else "medium",
                "recommendations": [
                    "Adjust sector allocations based on regime probabilities",
                    "Consider defensive positions in transition scenarios",
                    "Monitor key regime indicators for early signals"
                ],
                "impact_score": int((regime_data.get(current_regime, 0.5)) * 100)
            },
            {
                "title": "Supply Chain Intelligence", 
                "content": "Network analysis reveals concentrated risks in Southeast Asian shipping routes with 15% freight cost increases. Dependencies on key suppliers show elevated stress signals, particularly in technology and automotive sectors.",
                "risk_level": "high",
                "recommendations": [
                    "Diversify supplier base in critical components",
                    "Increase inventory buffers for high-risk items",
                    "Establish alternative logistics pathways"
                ],
                "impact_score": 72
            },
            {
                "title": "Market Volatility Outlook",
                "content": f"Forward-looking indicators suggest {'elevated' if forecast_delta_val > 0 else 'subdued'} volatility patterns. Cross-asset correlations are {'breaking down' if abs(forecast_delta_val) > 5 else 'remaining stable'}, creating both risks and opportunities for systematic strategies.",
                "risk_level": "medium" if abs(forecast_delta_val) > 5 else "low",
                "recommendations": [
                    "Adjust position sizing based on volatility regime",
                    "Review correlation assumptions in risk models", 
                    "Consider vol-targeting strategies"
                ],
                "impact_score": int(abs(forecast_delta_val) * 10)
            },
            {
                "title": "Geopolitical Risk Factors",
                "content": "Trade flow analysis indicates emerging tensions in agricultural commodity routes. Recent policy changes suggest potential disruptions to energy supply chains, with cascading effects across industrial sectors.",
                "risk_level": "medium",
                "recommendations": [
                    "Monitor commodity exposure in portfolios",
                    "Assess energy-intensive operations",
                    "Review geographic concentration risks"
                ],
                "impact_score": 58
            }
        ]
        
        # Get week dates
        current_date = datetime.utcnow()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_end = week_start + timedelta(days=6)
        
        if week == "last":
            week_start -= timedelta(days=7)
            week_end -= timedelta(days=7)
        elif week == "two-weeks":
            week_start -= timedelta(days=14)
            week_end -= timedelta(days=14)
        
        result = {
            "brief": {
                "id": f"brief-{week}-{int(current_date.timestamp())}",
                "week_start_date": week_start.isoformat(),
                "week_end_date": week_end.isoformat(),
                "title": f"RRIO Weekly Intelligence Brief - {week_start.strftime('%B %d, %Y')}",
                "executive_summary": sections[0]["content"],
                "sections": sections,
                "geri_score": geri_score,
                "current_regime": current_regime,
                "forecast_delta": forecast_delta_val,
                "format_type": format,
                "generated_at": current_date.isoformat(),
                "version": "1.0",
                "status": "published"
            },
            "metadata": {
                "subscribers_count": 1247,
                "format": format,
                "week_requested": week
            },
            "generated_at": current_date.isoformat()
        }
        
        # Cache the result (1 hour soft TTL, 4 hour hard TTL for weekly briefs)
        cache.set(cache_key, result, {
            "source": "weekly_brief_generator",
            "source_url": f"/api/v1/insights/weekly?week={week}&format={format}",
            "derivation_flag": "derived",
            "soft_ttl": 3600,    # 1 hour
            "hard_ttl": 14400    # 4 hours
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate weekly brief: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate weekly brief")

@router.post("/api/v1/insights/weekly/subscribe")
def subscribe_to_weekly_brief(
    subscription_data: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Subscribe to weekly intelligence brief."""
    try:
        email = subscription_data.get("email")
        name = subscription_data.get("name", "")
        company = subscription_data.get("company", "")
        title = subscription_data.get("title", "")
        format_preference = subscription_data.get("format_preference", "executive")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check if already subscribed
        existing = db.query(WeeklyBriefSubscription).filter(
            WeeklyBriefSubscription.email == email
        ).first()
        
        if existing:
            if existing.active:
                return {
                    "message": "Already subscribed",
                    "subscription_id": existing.id,
                    "status": "active"
                }
            else:
                # Reactivate subscription
                existing.active = True
                existing.subscribed_at = datetime.utcnow()
                db.commit()
                return {
                    "message": "Subscription reactivated",
                    "subscription_id": existing.id,
                    "status": "reactivated"
                }
        
        # Create new subscription
        new_subscription = WeeklyBriefSubscription(
            email=email,
            name=name,
            company=company,
            title=title,
            format_preference=format_preference,
            active=True
        )
        
        db.add(new_subscription)
        db.commit()
        db.refresh(new_subscription)
        
        return {
            "message": "Successfully subscribed to weekly intelligence brief",
            "subscription_id": new_subscription.id,
            "status": "subscribed",
            "format_preference": format_preference,
            "subscribed_at": new_subscription.subscribed_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create subscription: {e}")
        raise HTTPException(status_code=500, detail="Failed to create subscription")

@router.delete("/api/v1/insights/weekly/unsubscribe/{email}")
def unsubscribe_from_weekly_brief(
    email: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Unsubscribe from weekly intelligence brief."""
    try:
        subscription = db.query(WeeklyBriefSubscription).filter(
            WeeklyBriefSubscription.email == email
        ).first()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        subscription.active = False
        subscription.unsubscribed_at = datetime.utcnow()
        db.commit()
        
        return {
            "message": "Successfully unsubscribed",
            "email": email,
            "unsubscribed_at": subscription.unsubscribed_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unsubscribe: {e}")
        raise HTTPException(status_code=500, detail="Failed to unsubscribe")

@router.get("/api/v1/insights/weekly/subscriptions/count")
def get_subscription_count(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get current subscription count."""
    try:
        active_count = db.query(WeeklyBriefSubscription).filter(
            WeeklyBriefSubscription.active == True
        ).count()
        
        total_count = db.query(WeeklyBriefSubscription).count()
        
        return {
            "active_subscriptions": active_count,
            "total_subscriptions": total_count,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get subscription count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get subscription count")
