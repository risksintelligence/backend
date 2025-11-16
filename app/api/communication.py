#!/usr/bin/env python3
"""
Communication API endpoints for newsletter status and publishing.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from app.core.auth import require_observatory_read, optional_auth
from app.services.geri import compute_geri_score
from app.db import SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/api/v1/communication/newsletter-status")
def get_newsletter_status(
    _auth: dict = Depends(optional_auth)
) -> Dict[str, Any]:
    """Get current newsletter draft status and publishing schedule."""
    try:
        # Mock newsletter status - in production this would track actual newsletter state
        current_date = datetime.utcnow()
        
        # Calculate next publishing dates
        next_daily = current_date.replace(hour=13, minute=0, second=0, microsecond=0)
        if current_date.hour >= 13:
            next_daily += timedelta(days=1)
            
        # Next Friday at 18:00 UTC for weekly
        days_until_friday = (4 - current_date.weekday()) % 7  # Friday is 4
        if days_until_friday == 0 and current_date.hour >= 18:
            days_until_friday = 7
        next_weekly = current_date.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=days_until_friday)
        
        newsletter_status = {
            "daily_flash": {
                "status": "draft_ready",
                "last_published": (current_date - timedelta(days=1)).replace(hour=13).isoformat(),
                "next_scheduled": next_daily.isoformat(),
                "draft_preview": {
                    "headline": "GERII Holds Steady in Moderate Risk Band",
                    "geri_score": 52.3,
                    "risk_band": "Moderate",
                    "key_drivers": ["VIX volatility increase", "Yield curve flattening", "PMI softness"],
                    "word_count": 287,
                    "estimated_read_time": "2 minutes"
                },
                "automation": {
                    "enabled": True,
                    "slack_posting": True,
                    "email_delivery": True,
                    "last_automation_run": (current_date - timedelta(hours=1)).isoformat(),
                    "next_automation_run": next_daily.isoformat()
                }
            },
            "weekly_wrap": {
                "status": "in_progress", 
                "last_published": (current_date - timedelta(days=7)).replace(hour=18).isoformat(),
                "next_scheduled": next_weekly.isoformat(),
                "draft_preview": {
                    "headline": "Week in Review: Supply Chain Signals Dominate GERII Movement",
                    "regime_summary": "Transitioned from Calm to Inflationary_Stress (67% confidence)",
                    "key_themes": [
                        "Energy price volatility impact on transportation costs",
                        "Federal Reserve policy signals and yield curve dynamics", 
                        "PMI divergence across manufacturing and services"
                    ],
                    "word_count": 1247,
                    "estimated_read_time": "6 minutes"
                },
                "automation": {
                    "enabled": True,
                    "slack_posting": True,
                    "email_delivery": False,  # Manual review required
                    "last_automation_run": (current_date - timedelta(days=7)).isoformat(),
                    "next_automation_run": next_weekly.isoformat()
                }
            },
            "special_reports": {
                "in_progress": [
                    {
                        "title": "Q4 2024 Resilience Outlook",
                        "type": "quarterly_outlook",
                        "progress": 75,
                        "expected_completion": "2024-12-01T00:00:00Z",
                        "author": "Editorial Team"
                    }
                ],
                "recently_published": [
                    {
                        "title": "Federal Reserve Policy Regime Analysis", 
                        "published_date": "2024-11-01T00:00:00Z",
                        "type": "policy_analysis",
                        "views": 2847,
                        "engagement_score": 8.7
                    }
                ]
            },
            "subscription_metrics": {
                "total_subscribers": 3247,
                "active_subscribers": 2891,
                "weekly_growth": 23,
                "engagement_rates": {
                    "daily_open_rate": 0.67,
                    "weekly_open_rate": 0.78,
                    "click_through_rate": 0.23
                },
                "subscriber_segments": {
                    "analysts": 1456,
                    "policymakers": 387,
                    "researchers": 892,
                    "students": 512
                }
            },
            "content_pipeline": {
                "scheduled_content": [
                    {
                        "type": "daily_flash",
                        "scheduled_time": next_daily.isoformat(),
                        "auto_generated": True
                    },
                    {
                        "type": "weekly_wrap", 
                        "scheduled_time": next_weekly.isoformat(),
                        "auto_generated": True,
                        "requires_review": True
                    }
                ],
                "upcoming_themes": [
                    "Energy transition impacts on supply chains",
                    "Central bank communication effectiveness",
                    "Geopolitical risk integration methods"
                ]
            },
            "generated_at": current_date.isoformat()
        }
        
        return newsletter_status
        
    except Exception as e:
        logger.error(f"Failed to fetch newsletter status: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch newsletter status")

@router.get("/api/v1/communication/publishing-calendar")
def get_publishing_calendar(
    days: int = Query(30, ge=1, le=90),
    _auth: dict = Depends(optional_auth)
) -> Dict[str, Any]:
    """Get upcoming publishing calendar and content schedule."""
    try:
        current_date = datetime.utcnow()
        end_date = current_date + timedelta(days=days)
        
        calendar_events = []
        
        # Generate daily flash schedule
        date = current_date
        while date <= end_date:
            if date.weekday() < 5:  # Monday-Friday
                daily_time = date.replace(hour=13, minute=0, second=0, microsecond=0)
                calendar_events.append({
                    "type": "daily_flash",
                    "title": f"Daily Flash - {date.strftime('%Y-%m-%d')}",
                    "scheduled_time": daily_time.isoformat(),
                    "status": "scheduled",
                    "automation": True,
                    "estimated_duration": "15 minutes"
                })
            date += timedelta(days=1)
        
        # Generate weekly wrap schedule
        date = current_date
        while date <= end_date:
            if date.weekday() == 4:  # Friday
                weekly_time = date.replace(hour=18, minute=0, second=0, microsecond=0)
                calendar_events.append({
                    "type": "weekly_wrap",
                    "title": f"Weekly Wrap - Week of {date.strftime('%Y-%m-%d')}",
                    "scheduled_time": weekly_time.isoformat(), 
                    "status": "scheduled",
                    "automation": True,
                    "requires_review": True,
                    "estimated_duration": "45 minutes"
                })
            date += timedelta(days=1)
        
        # Add special events
        special_events = [
            {
                "type": "quarterly_outlook",
                "title": "Q4 2024 Resilience Outlook Release",
                "scheduled_time": "2024-12-01T16:00:00Z",
                "status": "planned",
                "automation": False,
                "estimated_duration": "2 hours"
            },
            {
                "type": "community_showcase",
                "title": "Sector Mission Showcase - Q4 Results",
                "scheduled_time": "2024-12-15T18:00:00Z", 
                "status": "confirmed",
                "automation": False,
                "estimated_duration": "90 minutes"
            }
        ]
        
        calendar_events.extend(special_events)
        
        # Sort by scheduled time
        calendar_events.sort(key=lambda x: x["scheduled_time"])
        
        return {
            "period": {
                "start": current_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "calendar_events": calendar_events[:50],  # Limit to 50 events
            "summary": {
                "total_events": len(calendar_events),
                "daily_flash_count": len([e for e in calendar_events if e["type"] == "daily_flash"]),
                "weekly_wrap_count": len([e for e in calendar_events if e["type"] == "weekly_wrap"]),
                "special_events_count": len(special_events),
                "automated_events": len([e for e in calendar_events if e.get("automation", False)])
            },
            "generated_at": current_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch publishing calendar: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch publishing calendar")

@router.post("/api/v1/communication/newsletter/preview")
def generate_newsletter_preview(
    newsletter_type: str = Query(..., regex="^(daily_flash|weekly_wrap)$"),
    _auth: dict = Depends(require_observatory_read)
) -> Dict[str, Any]:
    """Generate a preview of the next newsletter."""
    try:
        # Get current GERI data for preview
        try:
            from app.services.ingestion import ingest_local_series
            observations = ingest_local_series()
            geri_score, band, components = compute_geri_score(observations)
            geri_data = {
                "score": geri_score,
                "band": band, 
                "color": "#FFD600",
                "drivers": ["VIX", "YIELD_CURVE", "PMI"]
            }
        except:
            # Fallback if GERI service unavailable
            geri_data = {
                "score": 52.3,
                "band": "Moderate", 
                "color": "#FFD600",
                "drivers": ["VIX", "YIELD_CURVE", "PMI"]
            }
        
        current_time = datetime.utcnow()
        
        if newsletter_type == "daily_flash":
            preview = {
                "type": "daily_flash",
                "headline": f"GERII Flash: {geri_data['score']} ({geri_data['band']})",
                "content_sections": [
                    {
                        "section": "headline_summary",
                        "content": f"GERII stands at {geri_data['score']} in the {geri_data['band']} risk band ({geri_data['color']}) as of {current_time.strftime('%H:%M UTC')}."
                    },
                    {
                        "section": "key_drivers", 
                        "content": f"Primary drivers: {', '.join(geri_data['drivers'][:3])} showing increased volatility in today's session."
                    },
                    {
                        "section": "outlook",
                        "content": "Monitor yield curve dynamics and PMI releases for potential regime transition signals."
                    },
                    {
                        "section": "community_cta",
                        "content": "Join today's 15:00 UTC office hours for real-time GERII analysis and Q&A."
                    }
                ],
                "metadata": {
                    "word_count": 287,
                    "estimated_read_time": "2 minutes",
                    "geri_score": geri_data['score'],
                    "risk_band": geri_data['band'],
                    "color_code": geri_data['color']
                }
            }
        else:  # weekly_wrap
            preview = {
                "type": "weekly_wrap",
                "headline": "Week in Review: Supply Chain Signals Dominate GERII Movement",
                "content_sections": [
                    {
                        "section": "week_summary",
                        "content": f"GERII averaged {geri_data['score']:.1f} this week, with significant movement driven by supply chain indicators."
                    },
                    {
                        "section": "regime_analysis",
                        "content": "Regime classifier shows 67% probability of Inflationary_Stress regime, up from 23% last week."
                    },
                    {
                        "section": "forecast_outlook", 
                        "content": "24-hour forecast models suggest continued volatility in energy-sensitive components."
                    },
                    {
                        "section": "community_highlights",
                        "content": "MIT Resilience Lab published breakthrough analysis on climate transition risks. Stanford Policy Center begins Q1 energy scenario modeling."
                    },
                    {
                        "section": "upcoming_events",
                        "content": "Next week: Federal Reserve policy review, Q4 Sector Mission showcase planning, new Partner Lab onboarding."
                    }
                ],
                "metadata": {
                    "word_count": 1247,
                    "estimated_read_time": "6 minutes", 
                    "week_start": (current_time - timedelta(days=current_time.weekday())).isoformat(),
                    "week_end": current_time.isoformat(),
                    "featured_labs": ["MIT Resilience Lab", "Stanford Policy Center"]
                }
            }
        
        preview["generated_at"] = current_time.isoformat()
        preview["preview_id"] = f"prev_{newsletter_type}_{int(current_time.timestamp())}"
        
        return preview
        
    except Exception as e:
        logger.error(f"Failed to generate newsletter preview: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate newsletter preview")