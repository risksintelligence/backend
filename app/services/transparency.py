import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.core.cache import FileCache
from app.core.config import get_settings
from app.services.ingestion import ingest_local_series
from app.db import SessionLocal
from app.models import TransparencyLogModel, ObservationModel

settings = get_settings()
cache = FileCache("freshness")
TRANSPARENCY_FILE = settings.data_dir / "transparency.json"

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
    cached = cache.get("freshness")
    if cached:
        return cached
    observations = ingest_local_series()
    freshness = []
    for component, obs_list in observations.items():
        if not obs_list:
            continue
        last = obs_list[-1]
        status = 'fresh'
        age_days = (datetime.utcnow() - last.observed_at).days
        if age_days > 45:
            status = 'stale'
        elif age_days > 7:
            status = 'warning'
        freshness.append({
            "component": component,
            "status": status,
            "last_updated": last.observed_at.date().isoformat(),
        })
    cache.set("freshness", freshness)
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
