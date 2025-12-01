"""
Timeline Cascade History Service

Provides comprehensive timeline visualization data for supply chain cascade events,
tracking the progression of disruptions, their impacts, and recovery phases over time.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from ..core.unified_cache import UnifiedCache
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class CascadeEventType(Enum):
    INITIAL_DISRUPTION = "initial_disruption"
    SECONDARY_IMPACT = "secondary_impact"
    TERTIARY_IMPACT = "tertiary_impact"
    RECOVERY_START = "recovery_start"
    PARTIAL_RECOVERY = "partial_recovery"
    FULL_RECOVERY = "full_recovery"
    MITIGATION_ACTION = "mitigation_action"
    INTERVENTION = "intervention"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CascadePhase(Enum):
    PRE_EVENT = "pre_event"
    ONSET = "onset"
    PROPAGATION = "propagation"
    PEAK_IMPACT = "peak_impact"
    STABILIZATION = "stabilization"
    RECOVERY = "recovery"
    POST_EVENT = "post_event"


@dataclass
class TimelineEvent:
    event_id: str
    cascade_id: str
    timestamp: datetime
    event_type: CascadeEventType
    severity: SeverityLevel
    phase: CascadePhase
    title: str
    description: str
    affected_entities: List[str]  # Companies, regions, sectors
    impact_metrics: Dict[str, float]  # Cost, disruption duration, etc.
    location: Optional[Dict[str, float]]  # lat, lng for geographic events
    source: str
    confidence_level: float  # 0-100
    related_events: List[str]  # IDs of related events


@dataclass
class CascadeTimeline:
    cascade_id: str
    cascade_name: str
    trigger_event: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: Optional[int]
    total_events: int
    affected_sectors: List[str]
    affected_regions: List[str]
    peak_impact_date: Optional[datetime]
    recovery_start_date: Optional[datetime]
    total_cost_estimate: Optional[float]
    events: List[TimelineEvent]
    phases: List[Dict[str, Any]]
    summary_metrics: Dict[str, Any]


@dataclass
class TimelineFilter:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sectors: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    severity_levels: Optional[List[SeverityLevel]] = None
    event_types: Optional[List[CascadeEventType]] = None
    min_impact_threshold: Optional[float] = None


@dataclass
class TimelineVisualization:
    timeline_id: str
    title: str
    description: str
    time_range: Dict[str, str]  # start, end dates
    granularity: str  # "day", "week", "month"
    data_points: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    visualization_type: str  # "timeline", "gantt", "flowchart"


class TimelineCascadeService:
    """
    Service for generating timeline visualizations of supply chain cascade events.
    
    Provides historical analysis, event tracking, and interactive timeline data
    for understanding how supply chain disruptions propagate and resolve over time.
    """
    
    def __init__(self):
        self.cache = UnifiedCache("timeline_cascade")
        self.settings = get_settings()
        
        # Initialize historical cascade data
        self.historical_cascades = self._initialize_historical_data()
        
        # Timeline visualization templates
        self.visualization_templates = self._initialize_visualization_templates()


    async def get_cascade_timeline(self, cascade_id: str) -> Optional[CascadeTimeline]:
        """Get detailed timeline for a specific cascade event."""
        cache_key = f"cascade_timeline_{cascade_id}"
        
        cached_data, metadata = self.cache.get(cache_key)
        if cached_data and not (metadata and metadata.is_stale_soft):
            # Reconstruct timeline from cached data
            return self._reconstruct_timeline_from_cache(cached_data)
        
        # Generate or fetch timeline data
        timeline = await self._generate_cascade_timeline(cascade_id)
        
        if timeline:
            # Cache the timeline
            serializable_data = self._serialize_timeline(timeline)
            self.cache.set(
                cache_key,
                serializable_data,
                source="TIMELINE_CASCADE_SERVICE",
                source_url=f"/timeline/cascade/{cascade_id}",
                soft_ttl=21600  # 6 hours
            )
        
        return timeline


    async def get_timeline_visualization(self, 
                                       time_filter: TimelineFilter,
                                       visualization_type: str = "timeline") -> TimelineVisualization:
        """Generate timeline visualization data based on filters."""
        cache_key = f"timeline_viz_{hashlib.md5(str(asdict(time_filter)).encode()).hexdigest()}_{visualization_type}"
        
        cached_data, metadata = self.cache.get(cache_key)
        if cached_data and not (metadata and metadata.is_stale_soft):
            return self._reconstruct_visualization_from_cache(cached_data)
        
        # Generate visualization
        visualization = await self._generate_timeline_visualization(time_filter, visualization_type)
        
        # Cache the visualization
        serializable_data = self._serialize_visualization(visualization)
        self.cache.set(
            cache_key,
            serializable_data,
            source="TIMELINE_VISUALIZATION",
            source_url="/timeline/visualization",
            soft_ttl=7200  # 2 hours
        )
        
        return visualization


    async def get_cascade_history(self, 
                                time_range_days: int = 365,
                                limit: int = 50) -> List[CascadeTimeline]:
        """Get historical cascade timelines within specified time range."""
        cache_key = f"cascade_history_{time_range_days}_{limit}"
        
        cached_data, metadata = self.cache.get(cache_key)
        if cached_data and not (metadata and metadata.is_stale_soft):
            # Reconstruct timelines from cached data
            return [self._reconstruct_timeline_from_cache(item) for item in cached_data]
        
        # Generate historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range_days)
        
        timelines = []
        for cascade_data in self.historical_cascades:
            if start_date <= cascade_data["start_date"] <= end_date:
                timeline = await self._generate_cascade_timeline(cascade_data["cascade_id"])
                if timeline:
                    timelines.append(timeline)
        
        # Sort by start date (most recent first)
        timelines.sort(key=lambda x: x.start_date, reverse=True)
        timelines = timelines[:limit]
        
        # Cache the results
        serializable_data = [self._serialize_timeline(tl) for tl in timelines]
        self.cache.set(
            cache_key,
            serializable_data,
            source="TIMELINE_HISTORY",
            source_url="/timeline/history",
            soft_ttl=14400  # 4 hours
        )
        
        return timelines


    async def get_cascade_analytics(self, time_range_days: int = 365) -> Dict[str, Any]:
        """Get analytics and insights about cascade patterns over time."""
        cache_key = f"cascade_analytics_{time_range_days}"
        
        cached_data, metadata = self.cache.get(cache_key)
        if cached_data and not (metadata and metadata.is_stale_soft):
            return cached_data
        
        # Generate analytics
        analytics = await self._generate_cascade_analytics(time_range_days)
        
        # Cache analytics
        self.cache.set(
            cache_key,
            analytics,
            source="TIMELINE_ANALYTICS",
            source_url="/timeline/analytics",
            soft_ttl=43200  # 12 hours
        )
        
        return analytics


    def _initialize_historical_data(self) -> List[Dict[str, Any]]:
        """Initialize historical cascade event data."""
        return [
            {
                "cascade_id": "COVID19_GLOBAL_2020",
                "cascade_name": "COVID-19 Global Supply Chain Disruption",
                "start_date": datetime(2020, 3, 1),
                "end_date": datetime(2022, 6, 30),
                "trigger_event": "Pandemic lockdowns in China",
                "affected_sectors": ["healthcare", "automotive", "technology", "consumer_goods"],
                "affected_regions": ["China", "Europe", "North America", "Southeast Asia"],
                "peak_impact_date": datetime(2020, 4, 15),
                "recovery_start_date": datetime(2021, 3, 1),
                "estimated_cost": 4000000000000.0  # $4 trillion
            },
            {
                "cascade_id": "SUEZ_BLOCKAGE_2021",
                "cascade_name": "Suez Canal Ever Given Blockage",
                "start_date": datetime(2021, 3, 23),
                "end_date": datetime(2021, 4, 30),
                "trigger_event": "Ever Given container ship grounding",
                "affected_sectors": ["energy", "automotive", "consumer_goods", "materials"],
                "affected_regions": ["Europe", "Asia", "Middle East"],
                "peak_impact_date": datetime(2021, 3, 27),
                "recovery_start_date": datetime(2021, 3, 29),
                "estimated_cost": 60000000000.0  # $60 billion
            },
            {
                "cascade_id": "CHIP_SHORTAGE_2020",
                "cascade_name": "Global Semiconductor Shortage",
                "start_date": datetime(2020, 9, 1),
                "end_date": datetime(2023, 3, 31),
                "trigger_event": "COVID-19 demand shifts and factory shutdowns",
                "affected_sectors": ["automotive", "technology", "consumer_electronics"],
                "affected_regions": ["Global"],
                "peak_impact_date": datetime(2021, 8, 1),
                "recovery_start_date": datetime(2022, 6, 1),
                "estimated_cost": 500000000000.0  # $500 billion
            },
            {
                "cascade_id": "UKRAINE_ENERGY_2022",
                "cascade_name": "Ukraine War Energy and Food Crisis",
                "start_date": datetime(2022, 2, 24),
                "end_date": datetime(2023, 12, 31),
                "trigger_event": "Russian invasion of Ukraine",
                "affected_sectors": ["energy", "agriculture", "materials", "transportation"],
                "affected_regions": ["Europe", "Global"],
                "peak_impact_date": datetime(2022, 9, 1),
                "recovery_start_date": datetime(2023, 3, 1),
                "estimated_cost": 1200000000000.0  # $1.2 trillion
            },
            {
                "cascade_id": "TEXAS_FREEZE_2021",
                "cascade_name": "Texas Winter Storm Supply Chain Freeze",
                "start_date": datetime(2021, 2, 13),
                "end_date": datetime(2021, 3, 15),
                "trigger_event": "Extreme cold weather and power grid failure",
                "affected_sectors": ["energy", "chemicals", "technology", "automotive"],
                "affected_regions": ["Texas", "US Gulf Coast"],
                "peak_impact_date": datetime(2021, 2, 17),
                "recovery_start_date": datetime(2021, 2, 20),
                "estimated_cost": 195000000000.0  # $195 billion
            }
        ]


    def _initialize_visualization_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize visualization templates for different timeline types."""
        return {
            "timeline": {
                "chart_type": "horizontal_timeline",
                "features": ["events", "phases", "severity_indicators", "impact_metrics"],
                "granularity_options": ["day", "week", "month"],
                "interactive_elements": ["zoom", "filter", "tooltip", "drill_down"]
            },
            "gantt": {
                "chart_type": "gantt_chart",
                "features": ["duration_bars", "dependencies", "milestones", "resources"],
                "granularity_options": ["day", "week"],
                "interactive_elements": ["resize", "drag", "dependency_links"]
            },
            "flowchart": {
                "chart_type": "cascade_flow",
                "features": ["cause_effect", "propagation_paths", "impact_magnitude"],
                "granularity_options": ["event", "phase"],
                "interactive_elements": ["expand_collapse", "path_highlight", "impact_radius"]
            }
        }


    async def _generate_cascade_timeline(self, cascade_id: str) -> Optional[CascadeTimeline]:
        """Generate detailed timeline for a cascade event."""
        # Find cascade in historical data
        cascade_data = next((c for c in self.historical_cascades if c["cascade_id"] == cascade_id), None)
        if not cascade_data:
            return None
        
        # Generate timeline events
        events = await self._generate_timeline_events(cascade_data)
        
        # Generate phases
        phases = self._generate_cascade_phases(cascade_data, events)
        
        # Calculate metrics
        duration_days = (cascade_data["end_date"] - cascade_data["start_date"]).days if cascade_data.get("end_date") else None
        
        summary_metrics = {
            "total_impact_cost": cascade_data.get("estimated_cost", 0),
            "affected_entities_count": len(cascade_data.get("affected_sectors", [])) + len(cascade_data.get("affected_regions", [])),
            "average_event_severity": self._calculate_average_severity(events),
            "recovery_time_days": (cascade_data.get("recovery_start_date", datetime.utcnow()) - cascade_data["start_date"]).days,
            "propagation_speed": len(events) / max(duration_days or 1, 1) if events else 0
        }
        
        return CascadeTimeline(
            cascade_id=cascade_id,
            cascade_name=cascade_data["cascade_name"],
            trigger_event=cascade_data["trigger_event"],
            start_date=cascade_data["start_date"],
            end_date=cascade_data.get("end_date"),
            duration_days=duration_days,
            total_events=len(events),
            affected_sectors=cascade_data.get("affected_sectors", []),
            affected_regions=cascade_data.get("affected_regions", []),
            peak_impact_date=cascade_data.get("peak_impact_date"),
            recovery_start_date=cascade_data.get("recovery_start_date"),
            total_cost_estimate=cascade_data.get("estimated_cost"),
            events=events,
            phases=phases,
            summary_metrics=summary_metrics
        )


    async def _generate_timeline_events(self, cascade_data: Dict[str, Any]) -> List[TimelineEvent]:
        """Generate timeline events for a cascade."""
        events = []
        cascade_id = cascade_data["cascade_id"]
        start_date = cascade_data["start_date"]
        
        # Generate initial disruption event
        events.append(TimelineEvent(
            event_id=f"{cascade_id}_initial",
            cascade_id=cascade_id,
            timestamp=start_date,
            event_type=CascadeEventType.INITIAL_DISRUPTION,
            severity=SeverityLevel.CRITICAL,
            phase=CascadePhase.ONSET,
            title="Initial Disruption",
            description=cascade_data["trigger_event"],
            affected_entities=cascade_data.get("affected_regions", [])[:2],
            impact_metrics={"initial_cost": cascade_data.get("estimated_cost", 0) * 0.1},
            location=None,
            source="historical_data",
            confidence_level=95.0,
            related_events=[]
        ))
        
        # Generate secondary impact events
        for i, sector in enumerate(cascade_data.get("affected_sectors", [])[:3]):
            event_date = start_date + timedelta(days=(i + 1) * 7)
            events.append(TimelineEvent(
                event_id=f"{cascade_id}_secondary_{i}",
                cascade_id=cascade_id,
                timestamp=event_date,
                event_type=CascadeEventType.SECONDARY_IMPACT,
                severity=SeverityLevel.HIGH,
                phase=CascadePhase.PROPAGATION,
                title=f"{sector.title()} Sector Impact",
                description=f"Supply chain disruption spreads to {sector} sector",
                affected_entities=[sector],
                impact_metrics={"sector_impact_cost": cascade_data.get("estimated_cost", 0) * 0.15},
                location=None,
                source="historical_data",
                confidence_level=85.0,
                related_events=[f"{cascade_id}_initial"]
            ))
        
        # Generate peak impact event
        if cascade_data.get("peak_impact_date"):
            events.append(TimelineEvent(
                event_id=f"{cascade_id}_peak",
                cascade_id=cascade_id,
                timestamp=cascade_data["peak_impact_date"],
                event_type=CascadeEventType.TERTIARY_IMPACT,
                severity=SeverityLevel.CRITICAL,
                phase=CascadePhase.PEAK_IMPACT,
                title="Peak Impact Reached",
                description="Maximum cascade impact across all affected sectors",
                affected_entities=cascade_data.get("affected_sectors", []),
                impact_metrics={"peak_cost": cascade_data.get("estimated_cost", 0) * 0.4},
                location=None,
                source="historical_data",
                confidence_level=90.0,
                related_events=[f"{cascade_id}_secondary_{i}" for i in range(len(cascade_data.get("affected_sectors", [])[:3]))]
            ))
        
        # Generate recovery events
        if cascade_data.get("recovery_start_date"):
            recovery_date = cascade_data["recovery_start_date"]
            events.append(TimelineEvent(
                event_id=f"{cascade_id}_recovery_start",
                cascade_id=cascade_id,
                timestamp=recovery_date,
                event_type=CascadeEventType.RECOVERY_START,
                severity=SeverityLevel.MEDIUM,
                phase=CascadePhase.RECOVERY,
                title="Recovery Phase Begins",
                description="Supply chain recovery and stabilization efforts commence",
                affected_entities=cascade_data.get("affected_regions", []),
                impact_metrics={"recovery_investment": cascade_data.get("estimated_cost", 0) * 0.1},
                location=None,
                source="historical_data",
                confidence_level=80.0,
                related_events=[f"{cascade_id}_peak"]
            ))
            
            # Partial recovery
            partial_recovery_date = recovery_date + timedelta(days=90)
            events.append(TimelineEvent(
                event_id=f"{cascade_id}_partial_recovery",
                cascade_id=cascade_id,
                timestamp=partial_recovery_date,
                event_type=CascadeEventType.PARTIAL_RECOVERY,
                severity=SeverityLevel.LOW,
                phase=CascadePhase.RECOVERY,
                title="Partial Recovery Achieved",
                description="Significant improvement in supply chain operations",
                affected_entities=cascade_data.get("affected_sectors", []),
                impact_metrics={"recovery_progress": 0.7},
                location=None,
                source="historical_data",
                confidence_level=75.0,
                related_events=[f"{cascade_id}_recovery_start"]
            ))
        
        return events


    def _generate_cascade_phases(self, cascade_data: Dict[str, Any], events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Generate cascade phases from timeline events."""
        phases = []
        
        # Group events by phase
        phase_groups = {}
        for event in events:
            phase = event.phase.value
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(event)
        
        # Generate phase summaries
        for phase_name, phase_events in phase_groups.items():
            start_time = min(event.timestamp for event in phase_events)
            end_time = max(event.timestamp for event in phase_events)
            duration = (end_time - start_time).days
            
            phases.append({
                "phase": phase_name,
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat(),
                "duration_days": duration,
                "event_count": len(phase_events),
                "severity": max(event.severity.value for event in phase_events),
                "description": self._get_phase_description(phase_name)
            })
        
        return sorted(phases, key=lambda x: x["start_date"])


    def _get_phase_description(self, phase: str) -> str:
        """Get description for cascade phase."""
        descriptions = {
            "onset": "Initial disruption occurs and immediate impacts are felt",
            "propagation": "Disruption spreads through supply chain networks",
            "peak_impact": "Maximum impact reached across all affected sectors",
            "stabilization": "Situation stabilizes and new equilibrium is established",
            "recovery": "Active recovery and restoration of normal operations"
        }
        return descriptions.get(phase, f"Phase: {phase}")


    async def _generate_timeline_visualization(self, time_filter: TimelineFilter, 
                                             visualization_type: str) -> TimelineVisualization:
        """Generate timeline visualization data."""
        timeline_id = f"viz_{int(datetime.utcnow().timestamp())}"
        
        # Apply filters to get relevant cascades
        filtered_cascades = await self._filter_cascades(time_filter)
        
        # Generate data points based on visualization type
        data_points = []
        annotations = []
        
        if visualization_type == "timeline":
            data_points, annotations = self._generate_timeline_data_points(filtered_cascades)
        elif visualization_type == "gantt":
            data_points, annotations = self._generate_gantt_data_points(filtered_cascades)
        elif visualization_type == "flowchart":
            data_points, annotations = self._generate_flowchart_data_points(filtered_cascades)
        
        # Calculate visualization metrics
        metrics = {
            "total_cascades": len(filtered_cascades),
            "total_events": sum(len(cascade.events) for cascade in filtered_cascades),
            "time_span_days": (time_filter.end_date - time_filter.start_date).days if time_filter.start_date and time_filter.end_date else 365,
            "average_cascade_duration": sum((cascade.duration_days or 0) for cascade in filtered_cascades) / len(filtered_cascades) if filtered_cascades else 0
        }
        
        return TimelineVisualization(
            timeline_id=timeline_id,
            title=f"Supply Chain Cascade Timeline ({visualization_type})",
            description=f"Interactive {visualization_type} visualization of supply chain cascade events",
            time_range={
                "start": time_filter.start_date.isoformat() if time_filter.start_date else (datetime.utcnow() - timedelta(days=365)).isoformat(),
                "end": time_filter.end_date.isoformat() if time_filter.end_date else datetime.utcnow().isoformat()
            },
            granularity="day",
            data_points=data_points,
            annotations=annotations,
            metrics=metrics,
            visualization_type=visualization_type
        )


    def _generate_timeline_data_points(self, cascades: List[CascadeTimeline]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate data points for timeline visualization."""
        data_points = []
        annotations = []
        
        for cascade in cascades:
            for event in cascade.events:
                data_points.append({
                    "id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "title": event.title,
                    "description": event.description,
                    "severity": event.severity.value,
                    "event_type": event.event_type.value,
                    "phase": event.phase.value,
                    "cascade_id": cascade.cascade_id,
                    "cascade_name": cascade.cascade_name,
                    "impact_cost": event.impact_metrics.get("initial_cost", 0)
                })
            
            # Add cascade-level annotations
            annotations.append({
                "type": "cascade_start",
                "timestamp": cascade.start_date.isoformat(),
                "label": f"Start: {cascade.cascade_name}",
                "cascade_id": cascade.cascade_id
            })
            
            if cascade.peak_impact_date:
                annotations.append({
                    "type": "peak_impact",
                    "timestamp": cascade.peak_impact_date.isoformat(),
                    "label": f"Peak: {cascade.cascade_name}",
                    "cascade_id": cascade.cascade_id
                })
        
        return data_points, annotations


    def _generate_gantt_data_points(self, cascades: List[CascadeTimeline]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate data points for Gantt chart visualization."""
        data_points = []
        annotations = []
        
        for i, cascade in enumerate(cascades):
            # Main cascade bar
            data_points.append({
                "id": cascade.cascade_id,
                "name": cascade.cascade_name,
                "start": cascade.start_date.isoformat(),
                "end": cascade.end_date.isoformat() if cascade.end_date else datetime.utcnow().isoformat(),
                "duration": cascade.duration_days or 0,
                "row": i,
                "type": "cascade",
                "total_cost": cascade.total_cost_estimate or 0
            })
            
            # Phase bars within cascade
            for j, phase in enumerate(cascade.phases):
                data_points.append({
                    "id": f"{cascade.cascade_id}_phase_{j}",
                    "name": phase["phase"].replace("_", " ").title(),
                    "start": phase["start_date"],
                    "end": phase["end_date"],
                    "duration": phase["duration_days"],
                    "row": i,
                    "sub_row": j,
                    "type": "phase",
                    "parent": cascade.cascade_id,
                    "severity": phase["severity"]
                })
        
        return data_points, annotations


    def _generate_flowchart_data_points(self, cascades: List[CascadeTimeline]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate data points for flowchart visualization."""
        data_points = []
        annotations = []
        
        for cascade in cascades:
            # Create nodes for each event
            for event in cascade.events:
                data_points.append({
                    "id": event.event_id,
                    "type": "event_node",
                    "label": event.title,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "timestamp": event.timestamp.isoformat(),
                    "affected_entities": event.affected_entities,
                    "cascade_id": cascade.cascade_id
                })
            
            # Create edges for event relationships
            for event in cascade.events:
                for related_event_id in event.related_events:
                    data_points.append({
                        "id": f"edge_{event.event_id}_{related_event_id}",
                        "type": "event_edge",
                        "source": related_event_id,
                        "target": event.event_id,
                        "cascade_id": cascade.cascade_id
                    })
        
        return data_points, annotations


    async def _filter_cascades(self, time_filter: TimelineFilter) -> List[CascadeTimeline]:
        """Filter cascades based on time filter criteria."""
        filtered_cascades = []
        
        for cascade_data in self.historical_cascades:
            # Apply time filter
            if time_filter.start_date and cascade_data["start_date"] < time_filter.start_date:
                continue
            if time_filter.end_date and cascade_data["start_date"] > time_filter.end_date:
                continue
            
            # Apply sector filter
            if time_filter.sectors:
                if not any(sector in cascade_data.get("affected_sectors", []) for sector in time_filter.sectors):
                    continue
            
            # Apply region filter
            if time_filter.regions:
                if not any(region in cascade_data.get("affected_regions", []) for region in time_filter.regions):
                    continue
            
            # Generate full timeline
            timeline = await self._generate_cascade_timeline(cascade_data["cascade_id"])
            if timeline:
                filtered_cascades.append(timeline)
        
        return filtered_cascades


    async def _generate_cascade_analytics(self, time_range_days: int) -> Dict[str, Any]:
        """Generate analytics about cascade patterns."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range_days)
        
        # Get cascades in time range
        relevant_cascades = [
            c for c in self.historical_cascades
            if start_date <= c["start_date"] <= end_date
        ]
        
        if not relevant_cascades:
            return {"message": "No cascades in specified time range"}
        
        # Calculate analytics
        total_cost = sum(c.get("estimated_cost", 0) for c in relevant_cascades)
        avg_duration = sum((c.get("end_date", datetime.utcnow()) - c["start_date"]).days for c in relevant_cascades) / len(relevant_cascades)
        
        # Sector impact analysis
        sector_impacts = {}
        for cascade in relevant_cascades:
            for sector in cascade.get("affected_sectors", []):
                sector_impacts[sector] = sector_impacts.get(sector, 0) + 1
        
        # Regional impact analysis
        region_impacts = {}
        for cascade in relevant_cascades:
            for region in cascade.get("affected_regions", []):
                region_impacts[region] = region_impacts.get(region, 0) + 1
        
        return {
            "time_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": time_range_days
            },
            "cascade_summary": {
                "total_cascades": len(relevant_cascades),
                "total_estimated_cost": total_cost,
                "average_duration_days": round(avg_duration, 1),
                "most_affected_sector": max(sector_impacts.items(), key=lambda x: x[1])[0] if sector_impacts else None,
                "most_affected_region": max(region_impacts.items(), key=lambda x: x[1])[0] if region_impacts else None
            },
            "sector_impacts": sector_impacts,
            "region_impacts": region_impacts,
            "trends": {
                "cascade_frequency": len(relevant_cascades) / (time_range_days / 365.25),
                "cost_trend": "increasing",  # Would be calculated from data
                "duration_trend": "stable",   # Would be calculated from data
                "severity_trend": "increasing"  # Would be calculated from data
            },
            "patterns": {
                "common_triggers": ["pandemic", "geopolitical_events", "natural_disasters", "technology_failures"],
                "propagation_speed": "7-14 days average",
                "recovery_time": "6-18 months average"
            }
        }


    def _calculate_average_severity(self, events: List[TimelineEvent]) -> float:
        """Calculate average severity score for events."""
        if not events:
            return 0.0
        
        severity_scores = {
            SeverityLevel.LOW: 25,
            SeverityLevel.MEDIUM: 50,
            SeverityLevel.HIGH: 75,
            SeverityLevel.CRITICAL: 100
        }
        
        total_score = sum(severity_scores.get(event.severity, 0) for event in events)
        return total_score / len(events)


    def _serialize_timeline(self, timeline: CascadeTimeline) -> Dict[str, Any]:
        """Serialize timeline for caching."""
        data = asdict(timeline)
        data['start_date'] = timeline.start_date.isoformat()
        data['end_date'] = timeline.end_date.isoformat() if timeline.end_date else None
        data['peak_impact_date'] = timeline.peak_impact_date.isoformat() if timeline.peak_impact_date else None
        data['recovery_start_date'] = timeline.recovery_start_date.isoformat() if timeline.recovery_start_date else None
        
        # Serialize events
        serialized_events = []
        for event in timeline.events:
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            event_dict['event_type'] = event.event_type.value
            event_dict['severity'] = event.severity.value
            event_dict['phase'] = event.phase.value
            serialized_events.append(event_dict)
        data['events'] = serialized_events
        
        return data


    def _serialize_visualization(self, visualization: TimelineVisualization) -> Dict[str, Any]:
        """Serialize visualization for caching."""
        return asdict(visualization)


    def _reconstruct_timeline_from_cache(self, cached_data: Dict[str, Any]) -> CascadeTimeline:
        """Reconstruct timeline object from cached data."""
        # Reconstruct events
        events = []
        for event_data in cached_data['events']:
            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
            event_data['event_type'] = CascadeEventType(event_data['event_type'])
            event_data['severity'] = SeverityLevel(event_data['severity'])
            event_data['phase'] = CascadePhase(event_data['phase'])
            events.append(TimelineEvent(**event_data))
        
        # Reconstruct timeline
        cached_data['events'] = events
        cached_data['start_date'] = datetime.fromisoformat(cached_data['start_date'])
        cached_data['end_date'] = datetime.fromisoformat(cached_data['end_date']) if cached_data['end_date'] else None
        cached_data['peak_impact_date'] = datetime.fromisoformat(cached_data['peak_impact_date']) if cached_data['peak_impact_date'] else None
        cached_data['recovery_start_date'] = datetime.fromisoformat(cached_data['recovery_start_date']) if cached_data['recovery_start_date'] else None
        
        return CascadeTimeline(**cached_data)


    def _reconstruct_visualization_from_cache(self, cached_data: Dict[str, Any]) -> TimelineVisualization:
        """Reconstruct visualization object from cached data."""
        return TimelineVisualization(**cached_data)


# Singleton instance
_timeline_cascade_service = None

def get_timeline_cascade_service() -> TimelineCascadeService:
    """Get singleton timeline cascade service instance."""
    global _timeline_cascade_service
    if _timeline_cascade_service is None:
        _timeline_cascade_service = TimelineCascadeService()
    return _timeline_cascade_service