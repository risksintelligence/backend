from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import hashlib
import logging

from app.data.registry import SERIES_REGISTRY
from app.core.provider_failover import failover_manager
from app.core.unified_cache import UnifiedCache
from app.db import SessionLocal
from app.models import ObservationModel
import json

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    series_id: str
    observed_at: datetime
    value: float


def ingest_local_series() -> Dict[str, List[Observation]]:
    """
    Ingest time series data using unified cache and real provider failover.
    No fake fallbacks - only real data from providers or cached real data.
    """
    observations: Dict[str, List[Observation]] = {}
    cache = UnifiedCache("ingestion")
    
    for series_id, metadata in SERIES_REGISTRY.items():
        try:
            logger.info(f"📊 Ingesting {series_id} from {metadata.provider}")
            
            # Use failover manager to get real data with provider redundancy
            raw_points = failover_manager.fetch_with_failover(series_id, limit=1000)  # 5-year window needs more data
            
            if not raw_points:
                logger.warning(f"No data returned for {series_id}")
                continue
            
            obs_list = []
            now = datetime.utcnow()
            
            for point in raw_points:
                # Skip points with cache metadata (these are stale indicators)
                if "_cache_metadata" in point:
                    logger.warning(f"Using stale cached data for {series_id}")
                
                # Handle different timestamp formats
                timestamp_str = point["timestamp"].replace("Z", "")
                if "T" not in timestamp_str:
                    timestamp_str += "T00:00:00"
                
                obs_list.append(
                    Observation(
                        series_id=series_id,
                        observed_at=datetime.fromisoformat(timestamp_str),
                        value=float(point["value"]),
                    )
                )
            
            if obs_list:
                # Sort by time
                obs_list.sort(key=lambda o: o.observed_at)
                observations[series_id] = obs_list
                
                # Persist with full data lineage
                _persist_observations_with_lineage(series_id, obs_list, metadata, raw_points)
                
                # Update unified cache
                latest_obs = obs_list[-1]
                cache_data = {
                    "timestamp": latest_obs.observed_at.isoformat(),
                    "value": str(latest_obs.value)
                }
                
                # Get appropriate TTLs
                soft_ttl, hard_ttl = _get_ttl_for_frequency(metadata.frequency)
                
                cache.set(
                    key=series_id,
                    value=cache_data,
                    source=metadata.provider,
                    source_url=f"https://api.{metadata.provider}.com/{metadata.id}",
                    derivation_flag="raw",
                    soft_ttl=soft_ttl,
                    hard_ttl=hard_ttl
                )
                
                logger.info(f"✅ Ingested {len(obs_list)} observations for {series_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest {series_id}: {e}")
            # Note: No fallbacks to fake data - real provider failover handles redundancy
    
    # Run comprehensive data quality validation on ingested data
    if observations:
        logger.info("🔍 Running data quality validation on ingested observations")
        try:
            from app.services.data_quality import validate_ingestion_output
            
            # Flatten observations for validation
            all_observations = []
            for series_id, obs_list in observations.items():
                for obs in obs_list:
                    all_observations.append({
                        "series_id": obs.series_id,
                        "value": obs.value,
                        "observed_at": obs.observed_at.isoformat(),
                        "fetched_at": datetime.utcnow().isoformat()
                    })
            
            # Validate data quality
            quality_report = validate_ingestion_output(all_observations)
            
            # Log quality results
            if quality_report.institutional_grade:
                logger.info(f"✅ Data quality validation passed - Institutional grade: {quality_report.overall_score:.3f}")
            else:
                logger.warning(f"⚠️ Data quality issues detected - Score: {quality_report.overall_score:.3f}, Critical: {quality_report.critical_issues}")
            
            # Store quality report for transparency endpoint
            _store_quality_report(quality_report)
            
        except Exception as e:
            logger.error(f"❌ Data quality validation failed: {e}")
    
    return observations


def _persist_observations(series_id: str, obs_list: List[Observation]) -> None:
    """Legacy function - use _persist_observations_with_lineage for new code."""
    db = SessionLocal()
    for obs in obs_list:
        db.merge(
            ObservationModel(
                series_id=series_id, 
                observed_at=obs.observed_at, 
                value=obs.value,
                # Basic lineage for legacy calls
                source="unknown",
                fetched_at=datetime.utcnow(),
                derivation_flag="raw"
            )
        )
    db.commit()
    db.close()


def _persist_observations_with_lineage(series_id: str, obs_list: List[Observation], 
                                     metadata, raw_points: List[Dict]) -> None:
    """Persist observations with full data lineage tracking."""
    db = SessionLocal()
    now = datetime.utcnow()
    
    # Calculate checksum for the entire dataset
    data_for_checksum = str([(obs.observed_at, obs.value) for obs in obs_list])
    checksum = hashlib.sha256(data_for_checksum.encode()).hexdigest()[:16]
    
    for obs in obs_list:
        # Check if observation already exists to avoid duplicates
        existing = db.query(ObservationModel).filter(
            ObservationModel.series_id == series_id,
            ObservationModel.observed_at == obs.observed_at
        ).first()
        
        if existing:
            # Update existing with new lineage data
            existing.source = metadata.provider
            existing.source_url = f"https://api.{metadata.provider}.com/{metadata.id}"
            existing.fetched_at = now
            existing.checksum = checksum
            existing.derivation_flag = "raw"
            existing.soft_ttl, existing.hard_ttl = _get_ttl_for_frequency(metadata.frequency)
        else:
            # Create new observation with full lineage
            db.add(ObservationModel(
                series_id=series_id,
                observed_at=obs.observed_at,
                value=obs.value,
                source=metadata.provider,
                source_url=f"https://api.{metadata.provider}.com/{metadata.id}",
                fetched_at=now,
                checksum=checksum,
                derivation_flag="raw",
                soft_ttl=_get_ttl_for_frequency(metadata.frequency)[0],
                hard_ttl=_get_ttl_for_frequency(metadata.frequency)[1]
            ))
    
    db.commit()
    db.close()
    logger.debug(f"Persisted {len(obs_list)} observations for {series_id} with full lineage")


def _get_ttl_for_frequency(frequency: str) -> tuple[int, int]:
    """Get appropriate soft and hard TTL values based on data frequency."""
    ttl_mapping = {
        "daily": (3600, 14400),      # 1hr soft, 4hr hard
        "weekly": (21600, 86400),    # 6hr soft, 24hr hard  
        "monthly": (86400, 2592000), # 24hr soft, 30 days hard
        "quarterly": (259200, 7776000), # 3 days soft, 90 days hard
        "intraday": (900, 3600),     # 15min soft, 1hr hard
    }
    
    return ttl_mapping.get(frequency.lower(), (3600, 14400))  # Default: 1hr soft, 4hr hard

def _store_quality_report(quality_report) -> None:
    """Store data quality report for transparency and audit purposes."""
    try:
        from app.services.data_quality import export_validation_summary
        
        # Use unified cache to store quality reports with 24-hour retention
        cache = UnifiedCache("data_quality")
        
        # Export summary for API consumption
        summary = export_validation_summary(quality_report)
        
        # Store detailed report
        cache.set(
            key="latest_quality_report",
            value=summary,
            source="rrio_data_quality_validator",
            source_url="internal://data_quality_validation",
            derivation_flag="validation",
            soft_ttl=86400,  # 24 hours
            hard_ttl=604800  # 7 days
        )
        
        # Also store historical quality metrics for trending
        timestamp_key = f"quality_report_{int(datetime.utcnow().timestamp())}"
        cache.set(
            key=timestamp_key,
            value=summary,
            source="rrio_data_quality_validator",
            source_url="internal://data_quality_validation", 
            derivation_flag="validation",
            soft_ttl=2592000,  # 30 days
            hard_ttl=7776000   # 90 days
        )
        
        logger.info(f"📊 Data quality report stored - Score: {summary['overall_score']}, Grade: {'INSTITUTIONAL' if summary['institutional_grade'] else 'STANDARD'}")
        
    except Exception as e:
        logger.error(f"Failed to store quality report: {e}")
