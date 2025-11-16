#!/usr/bin/env python3
"""
Community API endpoints for partner labs, scenario studio, and media.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from app.core.auth import require_contributor_submit, require_contributor_submit_or_reviewer
from app.services.submissions import get_submissions_summary
from app.db import SessionLocal

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
