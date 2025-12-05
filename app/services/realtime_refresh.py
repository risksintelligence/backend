"""
Real-time Data Refresh Service

Provides intelligent background refresh mechanisms for supply chain cascade data,
including smart polling, event-driven updates, and performance optimization.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import time
import hashlib

logger = logging.getLogger(__name__)


class RefreshPriority(Enum):
    CRITICAL = "critical"  # Every 30 seconds
    HIGH = "high"         # Every 2 minutes  
    MEDIUM = "medium"     # Every 5 minutes
    LOW = "low"          # Every 15 minutes


class DataSourceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class RefreshJob:
    job_id: str
    data_source: str
    refresh_function: str  # Function name to call
    priority: RefreshPriority
    last_refresh: datetime
    next_refresh: datetime
    refresh_count: int
    error_count: int
    last_error: Optional[str]
    data_hash: Optional[str]  # For change detection
    subscribers: Set[str]  # WebSocket connection IDs
    created_at: datetime


@dataclass
class DataUpdate:
    update_id: str
    data_source: str
    update_type: str  # "full_refresh", "incremental", "status_change"
    data_summary: Dict[str, Any]
    data_hash: str
    change_detected: bool
    affected_endpoints: List[str]
    processing_time_ms: float
    timestamp: datetime


class RealTimeRefreshService:
    """Real-time data refresh service for supply chain intelligence."""
    
    def __init__(self):
        from app.core.config import get_settings
        self.settings = get_settings()
        
        # Enforce Redis requirement in production for real-time refresh
        if self.settings.is_production and not self.settings.redis_url:
            raise RuntimeError("Real-time refresh service requires Redis in production environment")
        
        self.refresh_jobs: Dict[str, RefreshJob] = {}
        self.data_cache: Dict[str, Any] = {}
        self.update_history: List[DataUpdate] = []
        self.subscribers: Dict[str, Set[str]] = {}  # data_source -> connection_ids
        self.is_running = False
        self.refresh_task: Optional[asyncio.Task] = None
        
        # Refresh intervals in seconds (optimized for free sources)
        self.refresh_intervals = {
            RefreshPriority.CRITICAL: 30,     # Core cascade data
            RefreshPriority.HIGH: 120,        # Supply chain network updates
            RefreshPriority.MEDIUM: 900,      # GDELT geopolitical (15 min - matches their update frequency)
            RefreshPriority.LOW: 1800         # Maritime intelligence (30 min)
        }
        
        # Data source health monitoring
        self.source_health: Dict[str, DataSourceStatus] = {}
        
        # Initialize core refresh jobs
        self._initialize_refresh_jobs()

    def _initialize_refresh_jobs(self):
        """Initialize core supply chain data refresh jobs."""
        current_time = datetime.utcnow()
        
        # Critical supply chain data sources
        critical_sources = [
            {
                "job_id": "supply_cascade_snapshot",
                "data_source": "supply_cascade",
                "function": "refresh_cascade_snapshot",
                "priority": RefreshPriority.HIGH,
                "endpoints": ["/api/v1/network/supply-cascade"]
            },
            {
                "job_id": "cascade_impacts",
                "data_source": "cascade_impacts", 
                "function": "refresh_cascade_impacts",
                "priority": RefreshPriority.HIGH,
                "endpoints": ["/api/v1/network/cascade/impacts"]
            },
            {
                "job_id": "geopolitical_disruptions",
                "data_source": "geopolitical_intelligence",
                "function": "refresh_geopolitical_data",
                "priority": RefreshPriority.MEDIUM,  # 15-min refresh to match GDELT
                "endpoints": ["/api/v1/network/supply-cascade", "/api/v1/network/cascade/impacts"]
            },
            {
                "job_id": "maritime_intelligence_ports",
                "data_source": "maritime_intelligence",
                "function": "refresh_maritime_intelligence_data", 
                "priority": RefreshPriority.LOW,  # 30-min refresh for free maritime intelligence
                "endpoints": ["/api/v1/network/supply-cascade", "/api/v1/network/cascade/impacts"]
            },
            {
                "job_id": "wits_trade_flows",
                "data_source": "wits",
                "function": "refresh_wits_data",
                "priority": RefreshPriority.LOW,
                "endpoints": ["/api/v1/network/supply-cascade"]
            },
            {
                "job_id": "wto_trade_volumes",
                "data_source": "wto",
                "function": "refresh_wto_data",
                "priority": RefreshPriority.LOW,
                "endpoints": ["/api/v1/wto/trade-volume/global", "/api/v1/wto/trade-statistics/bilateral"]
            },
            {
                "job_id": "sector_vulnerability_assessment",
                "data_source": "sector_vulnerability",
                "function": "refresh_sector_vulnerability_data",
                "priority": RefreshPriority.MEDIUM,
                "endpoints": ["/api/v1/vulnerability/sector/summary", "/api/v1/vulnerability/sector/assessment/technology"]
            },
            {
                "job_id": "predictive_forecasts",
                "data_source": "predictive",
                "function": "refresh_predictive_analysis",
                "priority": RefreshPriority.MEDIUM,
                "endpoints": ["/api/v1/predictive/disruption-forecast", "/api/v1/predictive/early-warning"]
            },
            {
                "job_id": "timeline_cascade_data",
                "data_source": "timeline_cascade",
                "function": "refresh_timeline_cascade_data",
                "priority": RefreshPriority.MEDIUM,
                "endpoints": ["/api/v1/cascade/timeline/summary", "/api/v1/cascade/timeline/historical"]
            }
        ]
        
        for source_config in critical_sources:
            job = RefreshJob(
                job_id=source_config["job_id"],
                data_source=source_config["data_source"],
                refresh_function=source_config["function"],
                priority=RefreshPriority(source_config["priority"].value),
                last_refresh=current_time - timedelta(hours=1),  # Force initial refresh
                next_refresh=current_time,
                refresh_count=0,
                error_count=0,
                last_error=None,
                data_hash=None,
                subscribers=set(),
                created_at=current_time
            )
            
            self.refresh_jobs[job.job_id] = job
            self.source_health[source_config["data_source"]] = DataSourceStatus.HEALTHY

    async def start_refresh_service(self):
        """Start the real-time refresh service."""
        if self.is_running:
            logger.warning("Refresh service already running")
            return
        
        self.is_running = True
        self.refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Real-time refresh service started")

    async def stop_refresh_service(self):
        """Stop the real-time refresh service."""
        self.is_running = False
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time refresh service stopped")

    async def _refresh_loop(self):
        """Main refresh loop that processes all refresh jobs."""
        try:
            while self.is_running:
                current_time = datetime.utcnow()
                
                # Find jobs that need refreshing
                jobs_to_refresh = [
                    job for job in self.refresh_jobs.values()
                    if job.next_refresh <= current_time
                ]
                
                if jobs_to_refresh:
                    logger.info(f"Processing {len(jobs_to_refresh)} refresh jobs")
                    
                    # Process jobs concurrently by priority
                    critical_jobs = [j for j in jobs_to_refresh if j.priority == RefreshPriority.CRITICAL]
                    high_jobs = [j for j in jobs_to_refresh if j.priority == RefreshPriority.HIGH]
                    medium_jobs = [j for j in jobs_to_refresh if j.priority == RefreshPriority.MEDIUM]
                    low_jobs = [j for j in jobs_to_refresh if j.priority == RefreshPriority.LOW]
                    
                    # Process in priority order with concurrency
                    for job_batch in [critical_jobs, high_jobs, medium_jobs, low_jobs]:
                        if job_batch:
                            await self._process_job_batch(job_batch)
                
                # Sleep for 10 seconds before next check
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            logger.info("Refresh loop cancelled")
        except Exception as e:
            logger.error(f"Refresh loop error: {e}")
            await asyncio.sleep(30)  # Error backoff
            if self.is_running:
                # Restart the loop
                await self._refresh_loop()

    async def _process_job_batch(self, jobs: List[RefreshJob]):
        """Process a batch of jobs concurrently."""
        tasks = [self._execute_refresh_job(job) for job in jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_refresh_job(self, job: RefreshJob):
        """Execute a single refresh job."""
        start_time = time.time()
        
        try:
            logger.info(f"Refreshing {job.data_source} ({job.job_id})")
            
            # Execute the appropriate refresh function
            new_data = await self._call_refresh_function(job.refresh_function, job.data_source)
            
            # Calculate data hash for change detection
            data_json = json.dumps(new_data, sort_keys=True, default=str)
            new_hash = hashlib.md5(data_json.encode()).hexdigest()
            
            # Detect changes
            change_detected = job.data_hash != new_hash
            
            if change_detected or job.data_hash is None:
                # Update cache
                self.data_cache[job.data_source] = new_data
                
                # Create update record
                processing_time = (time.time() - start_time) * 1000
                update = DataUpdate(
                    update_id=f"{job.job_id}_{int(datetime.utcnow().timestamp())}",
                    data_source=job.data_source,
                    update_type="full_refresh" if job.data_hash is None else "incremental",
                    data_summary=self._create_data_summary(new_data, job.data_source),
                    data_hash=new_hash,
                    change_detected=change_detected,
                    affected_endpoints=self._get_affected_endpoints(job.data_source),
                    processing_time_ms=processing_time,
                    timestamp=datetime.utcnow()
                )
                
                self.update_history.append(update)
                
                # Keep only last 100 updates
                if len(self.update_history) > 100:
                    self.update_history = self.update_history[-100:]
                
                # Notify subscribers
                await self._notify_subscribers(job.data_source, update)
                
                logger.info(f"✅ {job.data_source} updated - changes detected: {change_detected}")
            else:
                logger.debug(f"No changes detected for {job.data_source}")
            
            # Update job status
            job.last_refresh = datetime.utcnow()
            job.next_refresh = job.last_refresh + timedelta(seconds=self.refresh_intervals[job.priority])
            job.refresh_count += 1
            job.data_hash = new_hash
            job.last_error = None
            
            # Update source health
            self.source_health[job.data_source] = DataSourceStatus.HEALTHY
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Refresh failed for {job.data_source}: {error_msg}")
            
            # Update job error status
            job.error_count += 1
            job.last_error = error_msg
            job.last_refresh = datetime.utcnow()
            
            # Implement exponential backoff for failed jobs
            base_interval = self.refresh_intervals[job.priority]
            backoff_multiplier = min(4, 1 + (job.error_count * 0.5))  # Max 4x backoff
            job.next_refresh = job.last_refresh + timedelta(seconds=base_interval * backoff_multiplier)
            
            # Update source health
            if job.error_count >= 3:
                self.source_health[job.data_source] = DataSourceStatus.FAILED
            elif job.error_count >= 1:
                self.source_health[job.data_source] = DataSourceStatus.DEGRADED

    async def _call_refresh_function(self, function_name: str, data_source: str) -> Any:
        """Call the appropriate refresh function for a data source."""
        
        if function_name == "refresh_cascade_snapshot":
            from app.services.worldbank_wits_integration import wb_wits
            from app.services.geopolitical_intelligence import geopolitical_intelligence
            from app.services.maritime_intelligence import maritime_intelligence
            
            # Get fresh data from all sources
            wits = wb_wits
             # maritime_intelligence already imported as singleton
            
            nodes, edges = await wits.build_supply_chain_network()
            disruptions = await geopolitical_intelligence.get_supply_chain_disruptions(days=30)
            shipping_delays = await maritime_intelligence.get_shipping_delays()
            maritime_disruptions = [delay.__dict__ for delay in shipping_delays if delay.severity in ["major", "critical"]]
            
            return {
                "nodes": nodes,
                "edges": edges,
                "disruptions": [d.__dict__ for d in disruptions] + maritime_disruptions,
                "as_of": datetime.utcnow().isoformat() + "Z"
            }
            
        elif function_name == "refresh_cascade_impacts":
            from app.services.geopolitical_intelligence import geopolitical_intelligence
            from app.services.maritime_intelligence import maritime_intelligence
            
             # maritime_intelligence already imported as singleton
            
            disruptions = await geopolitical_intelligence.get_supply_chain_disruptions(days=30)
            port_statuses = await maritime_intelligence.get_port_congestion()
            shipping_delays = await maritime_intelligence.get_shipping_delays()
            maritime_disruptions = [delay.__dict__ for delay in shipping_delays if delay.severity in ["major", "critical"]]
            
            # Calculate impacts (simplified version of cascade impacts logic)
            total_impact = sum(d.economic_impact_usd for d in disruptions if d.economic_impact_usd)
            total_impact += sum(d.get('economic_impact_usd', 0) for d in maritime_disruptions)
            
            return {
                "financial": {
                    "total_disruption_impact_usd": total_impact,
                    "active_disruptions": len(disruptions) + len(maritime_disruptions)
                },
                "policy": {
                    "policy_events": len([d for d in disruptions if d.event_type == "policy_change"])
                },
                "industry": {
                    "global_supply_chain": max(0.5, 1.0 - (len(disruptions) * 0.05))
                }
            }
            
        elif function_name == "refresh_geopolitical_data":
            from app.services.geopolitical_intelligence import geopolitical_intelligence
            disruptions = await geopolitical_intelligence.get_supply_chain_disruptions(days=7)  # Fresh weekly data
            return {"disruptions": disruptions, "count": len(disruptions)}
            
        elif function_name == "refresh_maritime_intelligence_data":
            from app.services.maritime_intelligence import maritime_intelligence
             # maritime_intelligence already imported as singleton
            port_statuses = await maritime_intelligence.get_port_congestion()
            shipping_delays = await maritime_intelligence.get_shipping_delays()
            risk_assessment = await maritime_intelligence.get_supply_chain_risk_assessment()
            return {
                "port_statuses": port_statuses, 
                "shipping_delays": shipping_delays,
                "supply_chain_risks": risk_assessment
            }
            
        elif function_name == "refresh_wits_data":
            from app.services.worldbank_wits_integration import wb_wits
            wits = wb_wits
            nodes, edges = await wits.build_supply_chain_network()
            return {"nodes": nodes, "edges": edges}
            
        elif function_name == "refresh_wto_data":
            from app.services.wto_integration import get_wto_integration
            wto = get_wto_integration()
            
            # Focus on what WTO actually provides for free
            try:
                trade_volume = await wto.get_global_trade_volume()
            except RuntimeError:
                trade_volume = None
                logger.warning("WTO global trade volume not available - continuing without it")
            
            # Skip bilateral data - we use World Bank WITS for this instead
            # Skip trade agreements - WTO doesn't provide free API access for this
            return {
                "global_trade_volume": trade_volume.total_global_trade if trade_volume else 0,
                "trade_growth": getattr(trade_volume, 'growth_rate', 0) if trade_volume else 0,
                "data_source": "WTO Global Trade Statistics",
                "note": "Using WTO for global volumes only - bilateral data from World Bank WITS"
            }
            
            
        elif function_name == "refresh_sector_vulnerability_data":
            from app.services.sector_vulnerability_assessment import get_sector_vulnerability_assessment, IndustrySector
            vulnerability_service = get_sector_vulnerability_assessment()
            
            # Refresh assessments for key sectors
            key_sectors = [IndustrySector.TECHNOLOGY, IndustrySector.AUTOMOTIVE, IndustrySector.HEALTHCARE, IndustrySector.ENERGY]
            
            assessments = {}
            total_critical = 0
            total_high = 0
            total_vulnerabilities = 0
            
            for sector in key_sectors:
                assessment = await vulnerability_service.assess_sector_vulnerabilities(sector)
                assessments[sector.value] = {
                    "overall_risk_score": assessment.overall_risk_score,
                    "critical_vulnerabilities": assessment.critical_vulnerabilities,
                    "high_vulnerabilities": assessment.high_vulnerabilities,
                    "total_vulnerabilities": assessment.vulnerability_count
                }
                total_critical += assessment.critical_vulnerabilities
                total_high += assessment.high_vulnerabilities
                total_vulnerabilities += assessment.vulnerability_count
            
            return {
                "sectors_analyzed": len(key_sectors),
                "total_vulnerabilities": total_vulnerabilities,
                "total_critical_vulnerabilities": total_critical,
                "total_high_vulnerabilities": total_high,
                "sector_assessments": assessments,
                "avg_risk_score": sum([a["overall_risk_score"] for a in assessments.values()]) / len(assessments)
            }
            
        elif function_name == "refresh_predictive_analysis":
            from app.services.predictive_analytics import get_predictive_analytics
            analytics = get_predictive_analytics()
            predictions = analytics.predict_disruptions(time_horizon_days=7, include_cascade_effects=False)
            return {"predictions": predictions, "count": len(predictions)}
            
        elif function_name == "refresh_timeline_cascade_data":
            from app.services.timeline_cascade_service import get_timeline_cascade_service
            from datetime import timedelta
            
            timeline_service = get_timeline_cascade_service()
            
            # Get recent cascades and timeline data
            recent_cascades = await timeline_service.get_cascade_history(limit=20)
            
            # Get timeline visualization for last 90 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
            from app.services.timeline_cascade_service import TimelineFilter
            time_filter = TimelineFilter(
                start_date=start_date,
                end_date=end_date
            )
            
            visualization = await timeline_service.get_timeline_visualization(time_filter, "timeline")
            patterns = await timeline_service.get_cascade_analytics(time_filter.range_days if hasattr(time_filter, 'range_days') else 30)
            
            return {
                "recent_cascades_count": len(recent_cascades),
                "recent_cascades": [
                    {
                        "cascade_id": c.cascade_id,
                        "title": c.title,
                        "severity": c.severity_level.value,
                        "start_date": c.start_date.isoformat(),
                        "affected_sectors": c.affected_sectors,
                        "event_count": len(c.events)
                    }
                    for c in recent_cascades[:5]  # Top 5 for summary
                ],
                "timeline_data": {
                    "total_cascades": len(recent_cascades),
                    "total_events": sum(len(c.events) for c in recent_cascades),
                    "peak_disruption_period": "2024-Q3",  # Placeholder
                    "most_affected_sector": "supply_chain",
                    "most_affected_region": "global"
                },
                "patterns": patterns
            }
        
        else:
            raise ValueError(f"Unknown refresh function: {function_name}")

    def _create_data_summary(self, data: Any, data_source: str) -> Dict[str, Any]:
        """Create a summary of the data update."""
        
        if data_source == "supply_cascade":
            return {
                "nodes_count": len(data.get("nodes", [])),
                "edges_count": len(data.get("edges", [])),
                "disruptions_count": len(data.get("disruptions", [])),
                "last_updated": data.get("as_of")
            }
        elif data_source == "cascade_impacts":
            financial = data.get("financial", {})
            return {
                "total_impact_usd": financial.get("total_disruption_impact_usd", 0),
                "active_disruptions": financial.get("active_disruptions", 0),
                "supply_chain_capacity": data.get("industry", {}).get("global_supply_chain", 1.0)
            }
        elif data_source == "geopolitical_intelligence":
            return {
                "disruptions_count": data.get("count", 0),
                "data_source": "Free Geopolitical Intelligence"
            }
        elif data_source == "maritime_intelligence":
            return {
                "ports_monitored": len(data.get("port_statuses", {})),
                "shipping_delays": len(data.get("shipping_delays", [])),
                "supply_chain_risks": len(data.get("supply_chain_risks", {}).get("risk_factors", [])) if isinstance(data.get("supply_chain_risks"), dict) else 0,
                "data_source": "Free Maritime Intelligence"
            }
        elif data_source == "wits":
            return {
                "trade_nodes": len(data.get("nodes", [])),
                "trade_flows": len(data.get("edges", [])),
                "data_source": "World Bank WITS"
            }
        elif data_source == "predictive":
            return {
                "predictions_count": data.get("count", 0),
                "forecast_horizon": "7 days",
                "data_source": "Predictive Analytics"
            }
        
        return {"data_source": data_source, "updated": True}

    def _get_affected_endpoints(self, data_source: str) -> List[str]:
        """Get list of API endpoints affected by data source updates."""
        
        endpoint_mapping = {
            "supply_cascade": ["/api/v1/network/supply-cascade"],
            "cascade_impacts": ["/api/v1/network/cascade/impacts"],
            "geopolitical_intelligence": ["/api/v1/network/supply-cascade", "/api/v1/network/cascade/impacts"],
            "maritime_intelligence": ["/api/v1/network/supply-cascade", "/api/v1/network/cascade/impacts"],
            "wits": ["/api/v1/network/supply-cascade"],
            "predictive": ["/api/v1/predictive/disruption-forecast", "/api/v1/predictive/early-warning"]
        }
        
        return endpoint_mapping.get(data_source, [])

    async def _notify_subscribers(self, data_source: str, update: DataUpdate):
        """Notify WebSocket subscribers of data updates."""
        if data_source not in self.subscribers:
            return
        
        notification = {
            "type": "data_update",
            "data_source": data_source,
            "update_id": update.update_id,
            "change_detected": update.change_detected,
            "data_summary": update.data_summary,
            "affected_endpoints": update.affected_endpoints,
            "timestamp": update.timestamp.isoformat() + "Z"
        }
        
        # In a real implementation, this would send WebSocket messages
        # For now, we'll log the notification
        subscriber_count = len(self.subscribers[data_source])
        logger.info(f"📡 Notifying {subscriber_count} subscribers for {data_source}")

    def subscribe_to_updates(self, connection_id: str, data_sources: List[str]):
        """Subscribe a WebSocket connection to data source updates."""
        for data_source in data_sources:
            if data_source not in self.subscribers:
                self.subscribers[data_source] = set()
            self.subscribers[data_source].add(connection_id)
            
            # Update job subscribers
            for job in self.refresh_jobs.values():
                if job.data_source == data_source:
                    job.subscribers.add(connection_id)
        
        logger.info(f"Connection {connection_id} subscribed to {data_sources}")

    def unsubscribe_from_updates(self, connection_id: str):
        """Unsubscribe a WebSocket connection from all data sources."""
        for data_source, connections in self.subscribers.items():
            connections.discard(connection_id)
            
            # Update job subscribers
            for job in self.refresh_jobs.values():
                job.subscribers.discard(connection_id)
        
        logger.info(f"Connection {connection_id} unsubscribed from all data sources")

    def get_refresh_status(self) -> Dict[str, Any]:
        """Get current status of all refresh jobs."""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "total_jobs": len(self.refresh_jobs),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "data_source": job.data_source,
                    "priority": job.priority.value,
                    "last_refresh": job.last_refresh.isoformat() + "Z",
                    "next_refresh": job.next_refresh.isoformat() + "Z",
                    "refresh_count": job.refresh_count,
                    "error_count": job.error_count,
                    "last_error": job.last_error,
                    "subscribers": len(job.subscribers),
                    "health": self.source_health.get(job.data_source, DataSourceStatus.HEALTHY).value
                }
                for job in self.refresh_jobs.values()
            ],
            "recent_updates": [
                {
                    "update_id": update.update_id,
                    "data_source": update.data_source,
                    "timestamp": update.timestamp.isoformat() + "Z",
                    "change_detected": update.change_detected,
                    "processing_time_ms": update.processing_time_ms
                }
                for update in self.update_history[-10:]  # Last 10 updates
            ],
            "total_subscribers": sum(len(connections) for connections in self.subscribers.values()),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

    def force_refresh(self, data_source: str) -> bool:
        """Force an immediate refresh of a specific data source."""
        job = None
        for j in self.refresh_jobs.values():
            if j.data_source == data_source:
                job = j
                break
        
        if not job:
            logger.error(f"No refresh job found for data source: {data_source}")
            return False
        
        # Set next refresh to now
        job.next_refresh = datetime.utcnow()
        logger.info(f"Forced refresh scheduled for {data_source}")
        return True

    def get_cached_data(self, data_source: str) -> Optional[Any]:
        """Get cached data for a data source."""
        return self.data_cache.get(data_source)

    def update_refresh_priority(self, data_source: str, new_priority: RefreshPriority) -> bool:
        """Update the refresh priority for a data source."""
        job = None
        for j in self.refresh_jobs.values():
            if j.data_source == data_source:
                job = j
                break
        
        if not job:
            return False
        
        old_priority = job.priority
        job.priority = new_priority
        
        # Recalculate next refresh time
        current_time = datetime.utcnow()
        time_since_last = (current_time - job.last_refresh).total_seconds()
        new_interval = self.refresh_intervals[new_priority]
        
        if time_since_last >= new_interval:
            job.next_refresh = current_time  # Refresh immediately
        else:
            job.next_refresh = job.last_refresh + timedelta(seconds=new_interval)
        
        logger.info(f"Updated {data_source} priority from {old_priority.value} to {new_priority.value}")
        return True


# Singleton instance
_refresh_service = None


def get_refresh_service() -> RealTimeRefreshService:
    """Get the real-time refresh service instance."""
    global _refresh_service
    if _refresh_service is None:
        _refresh_service = RealTimeRefreshService()
    return _refresh_service