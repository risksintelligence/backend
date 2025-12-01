import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

from app.core.unified_cache import UnifiedCache
from app.core.config import get_settings
from app.services.ingestion import ingest_local_series
from app.db import SessionLocal
from app.models import TransparencyLogModel, ObservationModel
from sqlalchemy import desc

settings = get_settings()
cache = UnifiedCache("freshness")
TRANSPARENCY_FILE = settings.data_dir / "transparency.json"
logger = logging.getLogger(__name__)

DEFAULT_DATA = {
    "update_log": [
        {"date": "2024-11-12", "description": "Added RRIO automation scripts"},
        {"date": "2024-11-10", "description": "Updated GRII weights"},
    ]
}


def _load_data() -> Dict[str, List[Dict[str, str]]]:
    if not TRANSPARENCY_FILE.exists():
        TRANSPARENCY_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRANSPARENCY_FILE.write_text(json.dumps(DEFAULT_DATA, indent=2))
    return json.loads(TRANSPARENCY_FILE.read_text())


def _write_data(data: Dict[str, List[Dict[str, str]]]) -> None:
    TRANSPARENCY_FILE.write_text(json.dumps(data, indent=2))


def get_data_freshness() -> List[Dict[str, str]]:
    """Return freshness rows using stored observations + TTL semantics."""
    cached_data, metadata = cache.get("freshness")
    if cached_data:
        return cached_data
    
    freshness: List[Dict[str, str]] = []
    now = datetime.utcnow()
    
    try:
        db = SessionLocal()
        try:
            observations = db.query(ObservationModel).order_by(
                ObservationModel.series_id,
                desc(ObservationModel.observed_at)
            ).all()
        finally:
            db.close()
        
        latest_by_series: Dict[str, ObservationModel] = {}
        for obs in observations:
            if obs.series_id not in latest_by_series:
                latest_by_series[obs.series_id] = obs
        
        for series_id, obs in latest_by_series.items():
            observed_at = obs.observed_at or now
            fetched_at = obs.fetched_at or observed_at
            age_seconds = (now - fetched_at).total_seconds()
            soft_ttl = obs.soft_ttl or 3600
            hard_ttl = obs.hard_ttl or (soft_ttl * 4)
            
            if age_seconds <= soft_ttl:
                status = "fresh"
            elif age_seconds <= hard_ttl:
                status = "warning"
            else:
                status = "stale"
            
            freshness.append({
                "component": series_id,
                "status": status,
                "last_updated": observed_at.isoformat(),
                "age_hours": round(age_seconds / 3600, 2),
                "soft_ttl": soft_ttl,
                "hard_ttl": hard_ttl,
                "source": obs.source,
                "source_url": obs.source_url
            })
    except Exception as exc:
        logger.warning(f"Failed to load database freshness, falling back to ingestion: {exc}")
    
    if not freshness:
        observations = ingest_local_series()
        for component, obs_list in observations.items():
            if not obs_list:
                continue
            last = obs_list[-1]
            age_hours = (now - last.observed_at).total_seconds() / 3600
            status = "fresh"
            if age_hours > 24 * 45:
                status = "stale"
            elif age_hours > 24 * 7:
                status = "warning"
            freshness.append({
                "component": component,
                "status": status,
                "last_updated": last.observed_at.isoformat(),
                "age_hours": round(age_hours, 2),
                "soft_ttl": 24 * 3600,
                "hard_ttl": 24 * 3600 * 4,
                "source": "ingestion_pipeline",
                "source_url": None
            })
    
    cache.set("freshness", freshness, source="transparency_pipeline", soft_ttl=300, hard_ttl=1200)
    return freshness


def get_update_log() -> List[Dict[str, str]]:
    """Get recent transparency log entries from database."""
    try:
        db = SessionLocal()
        logs = db.query(TransparencyLogModel).order_by(
            TransparencyLogModel.timestamp.desc()
        ).limit(50).all()
        
        result = []
        for log in logs:
            result.append({
                "date": log.timestamp.date().isoformat(),
                "description": log.description,
                "event_type": log.event_type
            })
        
        db.close()
        
        # Fallback to file-based system if no database entries
        if not result:
            return _load_data().get("update_log", [])
            
        return result
        
    except Exception as e:
        # Fallback to file-based system on database error
        return _load_data().get("update_log", [])


def add_transparency_log(event_type: str, description: str, metadata: Optional[Dict] = None) -> None:
    """Add a new transparency log entry to database."""
    try:
        db = SessionLocal()
        log_entry = TransparencyLogModel(
            event_type=event_type,
            description=description,
            timestamp=datetime.utcnow(),
            meta_data=metadata or {}
        )
        db.add(log_entry)
        db.commit()
        db.close()
        
    except Exception as e:
        # Fallback to file-based logging
        record_update(description)


def migrate_json_to_database() -> int:
    """Migrate existing JSON transparency log to database."""
    try:
        db = SessionLocal()
        
        # Check if we already have data in database
        existing_count = db.query(TransparencyLogModel).count()
        if existing_count > 0:
            db.close()
            return existing_count
        
        # Load JSON data
        data = _load_data()
        update_log = data.get("update_log", [])
        
        migrated_count = 0
        for entry in update_log:
            # Parse date
            try:
                entry_date = datetime.fromisoformat(entry["date"])
            except:
                entry_date = datetime.utcnow()
            
            # Create database entry
            log_entry = TransparencyLogModel(
                event_type="system_update",
                description=entry["description"],
                timestamp=entry_date,
                meta_data={"migrated_from_json": True}
            )
            db.add(log_entry)
            migrated_count += 1
        
        db.commit()
        db.close()
        
        return migrated_count
        
    except Exception as e:
        if 'db' in locals():
            db.rollback()
            db.close()
        raise e


def record_update(description: str) -> None:
    """Record update in database (with JSON fallback)."""
    try:
        add_transparency_log("system_update", description)
    except Exception:
        # Fallback to file-based logging
        data = _load_data()
        log = data.setdefault("update_log", [])
        log.insert(0, {"date": datetime.utcnow().date().isoformat(), "description": description})
        _write_data(data)
