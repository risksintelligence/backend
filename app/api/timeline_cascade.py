"""
Timeline Cascade Visualization API

Provides endpoints for visualizing and analyzing supply chain disruption cascades
over time with interactive timeline views and historical event analysis.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from app.core.security import require_system_rate_limit
from app.services.timeline_cascade_service import (
    get_timeline_cascade_service, 
    TimelineFilter, 
    SeverityLevel
)

router = APIRouter(prefix="/api/v1/cascade/timeline", tags=["timeline-cascade"])


@router.get("/cascade/{cascade_id}")
async def get_cascade_timeline(
    cascade_id: str = Path(..., description="Cascade ID to get timeline for"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get detailed timeline for a specific cascade event."""
    try:
        timeline_service = get_timeline_cascade_service()
        timeline = await timeline_service.get_cascade_timeline(cascade_id)
        
        if not timeline:
            raise HTTPException(status_code=404, detail=f"Cascade timeline not found for ID: {cascade_id}")
        
        return {
            "timeline": {
                "cascade_id": timeline.cascade_id,
                "title": timeline.title,
                "description": timeline.description,
                "start_date": timeline.start_date.isoformat(),
                "end_date": timeline.end_date.isoformat() if timeline.end_date else None,
                "total_duration_days": timeline.total_duration_days,
                "severity_level": timeline.severity_level.value,
                "affected_regions": timeline.affected_regions,
                "affected_sectors": timeline.affected_sectors,
                "total_events": len(timeline.events),
                "events": [
                    {
                        "event_id": event.event_id,
                        "title": event.title,
                        "description": event.description,
                        "timestamp": event.timestamp.isoformat(),
                        "phase": event.phase,
                        "severity": event.severity.value,
                        "affected_regions": event.affected_regions,
                        "affected_sectors": event.affected_sectors,
                        "key_metrics": event.key_metrics,
                        "cascade_triggers": event.cascade_triggers
                    }
                    for event in timeline.events
                ],
                "key_insights": timeline.key_insights,
                "lessons_learned": timeline.lessons_learned
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "timeline_cascade_analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cascade timeline: {str(e)}")


@router.get("/visualization")
async def get_timeline_visualization(
    start_date: str = Query(..., description="Start date for timeline (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date for timeline (YYYY-MM-DD)"),
    visualization_type: str = Query("timeline", description="Type of visualization (timeline, gantt, flowchart)"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity level"),
    sectors: Optional[List[str]] = Query(None, description="Filter by affected sectors"),
    regions: Optional[List[str]] = Query(None, description="Filter by affected regions"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get timeline visualization data for cascade events within a date range."""
    try:
        # Parse dates
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Validate visualization type
        available_types = ["timeline", "gantt", "flowchart"]
        if visualization_type.lower() not in available_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid visualization type '{visualization_type}'. Available types: {available_types}"
            )
        
        # Validate severity filter if provided
        severity_level = None
        if severity_filter:
            try:
                severity_level = SeverityLevel(severity_filter.lower())
            except ValueError:
                available_severities = [s.value for s in SeverityLevel]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity level '{severity_filter}'. Available levels: {available_severities}"
                )
        
        # Create filter
        time_filter = TimelineFilter(
            start_date=start_dt,
            end_date=end_dt,
            severity_levels=[severity_level] if severity_level else None,
            sectors=sectors,
            regions=regions
        )
        
        timeline_service = get_timeline_cascade_service()
        visualization = await timeline_service.get_timeline_visualization(time_filter, visualization_type.lower())
        
        return {
            "visualization": {
                "filter_criteria": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "visualization_type": visualization_type,
                    "severity_filter": severity_filter,
                    "sectors": sectors,
                    "regions": regions
                },
                "data": visualization.data_points,
                "config": {
                    "visualization_type": visualization.visualization_type,
                    "granularity": visualization.granularity,
                    "time_range": visualization.time_range
                },
                "metadata": {
                    "total_cascades": visualization.metrics.get("total_cascades", 0),
                    "total_events": visualization.metrics.get("total_events", 0),
                    "date_range_days": (end_dt - start_dt).days,
                    "most_affected_sector": "Unknown",
                    "most_affected_region": "Unknown", 
                    "peak_disruption_period": "Unknown"
                }
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "timeline_visualization"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline visualization: {str(e)}")


@router.get("/analytics/patterns")
async def get_cascade_patterns(
    time_period_days: int = Query(365, ge=30, le=1095, description="Analysis period in days"),
    min_severity: str = Query("medium", description="Minimum severity level for analysis"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Analyze cascade patterns and trends over time."""
    try:
        # Validate severity level
        try:
            severity_level = SeverityLevel(min_severity.lower())
        except ValueError:
            available_severities = [s.value for s in SeverityLevel]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity level '{min_severity}'. Available levels: {available_severities}"
            )
        
        # Create filter for analysis period
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_period_days)
        
        time_filter = TimelineFilter(
            start_date=start_date,
            end_date=end_date,
            severity_levels=[severity_level] if severity_level else None
        )
        
        timeline_service = get_timeline_cascade_service()
        patterns = await timeline_service.analyze_cascade_patterns(time_filter)
        
        return {
            "cascade_patterns": patterns,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "cascade_pattern_analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze cascade patterns: {str(e)}")


@router.get("/historical")
async def get_historical_cascades(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of cascades to return"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity level"),
    sector_filter: Optional[str] = Query(None, description="Filter by affected sector"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get list of historical cascade events with summary information."""
    try:
        # Validate severity filter if provided
        severity_level = None
        if severity_filter:
            try:
                severity_level = SeverityLevel(severity_filter.lower())
            except ValueError:
                available_severities = [s.value for s in SeverityLevel]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity level '{severity_filter}'. Available levels: {available_severities}"
                )
        
        timeline_service = get_timeline_cascade_service()
        cascades = await timeline_service.get_cascade_history(
            time_range_days=365,
            limit=limit
        )
        
        # Apply filters after getting data
        if severity_level:
            # Convert severity levels to compare with cascade data
            severity_map = {
                SeverityLevel.LOW: ["low"],
                SeverityLevel.MEDIUM: ["low", "medium"], 
                SeverityLevel.HIGH: ["low", "medium", "high"],
                SeverityLevel.CRITICAL: ["low", "medium", "high", "critical"]
            }
            allowed_severities = severity_map.get(severity_level, [])
            cascades = [c for c in cascades if any(event.severity.value in allowed_severities for event in c.events)]
        
        if sector_filter:
            cascades = [c for c in cascades if sector_filter.lower() in [s.lower() for s in c.affected_sectors]]
        
        cascade_summaries = []
        for cascade in cascades:
            summary = {
                "cascade_id": cascade.cascade_id,
                "title": cascade.title,
                "description": cascade.description,
                "start_date": cascade.start_date.isoformat(),
                "end_date": cascade.end_date.isoformat() if cascade.end_date else None,
                "duration_days": cascade.total_duration_days,
                "severity_level": cascade.severity_level.value,
                "affected_regions": cascade.affected_regions,
                "affected_sectors": cascade.affected_sectors,
                "total_events": len(cascade.events),
                "peak_event": {
                    "title": max(cascade.events, key=lambda x: x.severity.value).title,
                    "date": max(cascade.events, key=lambda x: x.severity.value).timestamp.isoformat()
                } if cascade.events else None
            }
            cascade_summaries.append(summary)
        
        return {
            "historical_cascades": {
                "total_found": len(cascade_summaries),
                "filter_criteria": {
                    "limit": limit,
                    "severity_filter": severity_filter,
                    "sector_filter": sector_filter
                },
                "cascades": cascade_summaries
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "historical_cascade_summary"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical cascades: {str(e)}")


@router.get("/sectors/impact")
async def get_sector_impact_timeline(
    sector: str = Query(..., description="Sector to analyze impact for"),
    time_period_days: int = Query(180, ge=30, le=730, description="Analysis period in days"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Analyze cascade impact timeline for a specific sector."""
    try:
        # Create filter for sector analysis
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_period_days)
        
        time_filter = TimelineFilter(
            start_date=start_date,
            end_date=end_date,
            sectors=[sector.lower()]
        )
        
        timeline_service = get_timeline_cascade_service()
        impact_analysis = await timeline_service.get_sector_impact_timeline(sector, time_filter)
        
        return {
            "sector_impact_timeline": impact_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "sector_impact_timeline"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze sector impact timeline: {str(e)}")


@router.get("/comparison")
async def compare_cascade_timelines(
    cascade_ids: List[str] = Query(..., description="List of cascade IDs to compare"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Compare multiple cascade timelines for pattern analysis."""
    try:
        if len(cascade_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 cascade IDs required for comparison")
        
        if len(cascade_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 cascades can be compared at once")
        
        timeline_service = get_timeline_cascade_service()
        comparison = await timeline_service.compare_cascades(cascade_ids)
        
        return {
            "cascade_comparison": comparison,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "cascade_timeline_comparison"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare cascade timelines: {str(e)}")


@router.get("/summary")
async def get_timeline_summary(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get high-level summary of all timeline cascade data."""
    try:
        timeline_service = get_timeline_cascade_service()
        
        # Get summary statistics
        all_cascades = await timeline_service.get_cascade_history(time_range_days=365, limit=50)
        
        if not all_cascades:
            return {
                "timeline_summary": {
                    "total_cascades": 0,
                    "message": "No cascade data available"
                },
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "source": "timeline_summary"
            }
        
        # Calculate summary statistics
        total_cascades = len(all_cascades)
        total_events = sum(len(cascade.events) for cascade in all_cascades)
        
        severity_counts = {}
        sector_impacts = {}
        region_impacts = {}
        
        for cascade in all_cascades:
            # Count by severity
            severity = cascade.severity_level.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count sector impacts
            for sector in cascade.affected_sectors:
                sector_impacts[sector] = sector_impacts.get(sector, 0) + 1
            
            # Count region impacts
            for region in cascade.affected_regions:
                region_impacts[region] = region_impacts.get(region, 0) + 1
        
        # Find most impacted
        most_impacted_sector = max(sector_impacts, key=sector_impacts.get) if sector_impacts else "None"
        most_impacted_region = max(region_impacts, key=region_impacts.get) if region_impacts else "None"
        
        # Recent activity (last 30 days)
        recent_date = datetime.utcnow() - timedelta(days=30)
        recent_cascades = [c for c in all_cascades if c.start_date >= recent_date]
        
        return {
            "timeline_summary": {
                "total_cascades": total_cascades,
                "total_events": total_events,
                "average_events_per_cascade": round(total_events / total_cascades, 1) if total_cascades > 0 else 0,
                "severity_distribution": severity_counts,
                "most_impacted_sector": most_impacted_sector,
                "most_impacted_region": most_impacted_region,
                "recent_activity": {
                    "cascades_last_30_days": len(recent_cascades),
                    "active_cascades": len([c for c in recent_cascades if c.end_date is None])
                },
                "top_sectors": sorted(sector_impacts.items(), key=lambda x: x[1], reverse=True)[:5],
                "top_regions": sorted(region_impacts.items(), key=lambda x: x[1], reverse=True)[:5],
                "average_duration_days": round(
                    sum(c.total_duration_days for c in all_cascades if c.total_duration_days) / 
                    len([c for c in all_cascades if c.total_duration_days]), 1
                ) if any(c.total_duration_days for c in all_cascades) else 0
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "source": "timeline_cascade_summary"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline summary: {str(e)}")