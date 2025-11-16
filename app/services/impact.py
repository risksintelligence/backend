import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SNAPSHOT_FILE = DATA_DIR / "impact_snapshot.json"
HISTORY_FILE = DATA_DIR / "impact_history.json"
MAX_HISTORY_POINTS = 365

DEFAULT_COMPONENTS = {
    "policy": 0.2,
    "analyses": 0.15,
    "labs": 0.12,
    "media": 0.11,
    "community": 0.1,
}
WEIGHTS = {
    "policy": 0.25,
    "analyses": 0.2,
    "labs": 0.2,
    "media": 0.2,
    "community": 0.15,
}

@dataclass
class RASSnapshot:
    composite: float
    components: Dict[str, float]
    calculated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite": round(self.composite, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "calculated_at": self.calculated_at.isoformat() + "Z",
        }

def _compute_composite(components: Dict[str, float]) -> float:
    return sum(components[c] * WEIGHTS.get(c, 0) for c in components)

def load_snapshot() -> RASSnapshot:
    DATA_DIR.mkdir(exist_ok=True)
    if SNAPSHOT_FILE.exists():
        payload = json.loads(SNAPSHOT_FILE.read_text())
        snapshot = RASSnapshot(
            composite=payload["composite"],
            components=payload["components"],
            calculated_at=datetime.fromisoformat(payload["calculated_at"].replace("Z", "")),
        )
        _ensure_history_seed(snapshot)
        return snapshot
    snapshot = RASSnapshot(
        composite=_compute_composite(DEFAULT_COMPONENTS),
        components=DEFAULT_COMPONENTS,
        calculated_at=datetime.utcnow(),
    )
    save_snapshot(snapshot)
    return snapshot

def save_snapshot(snapshot: RASSnapshot) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(snapshot.to_dict(), indent=2))
    _append_history(snapshot)

def update_snapshot(metric_updates: Dict[str, float]) -> RASSnapshot:
    snapshot = load_snapshot()
    updated_components = {**snapshot.components, **metric_updates}
    new_snapshot = RASSnapshot(
        composite=_compute_composite(updated_components),
        components=updated_components,
        calculated_at=datetime.utcnow(),
    )
    save_snapshot(new_snapshot)
    return new_snapshot

def get_snapshot_history(limit: int = 90) -> List[Dict[str, Any]]:
    """Return chronological list of RAS snapshots for trend visualizations."""
    DATA_DIR.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        snapshot = load_snapshot()
        _append_history(snapshot)
    
    try:
        history = json.loads(HISTORY_FILE.read_text())
    except json.JSONDecodeError:
        history = []
    
    history.sort(key=lambda entry: entry["calculated_at"])
    return history[-max(1, min(limit, MAX_HISTORY_POINTS)):]


def _append_history(snapshot: RASSnapshot) -> None:
    """Persist snapshot into rolling history for charts."""
    DATA_DIR.mkdir(exist_ok=True)
    entry = snapshot.to_dict()
    try:
        history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    except json.JSONDecodeError:
        history = []
    
    # Drop duplicate timestamps before appending
    history = [point for point in history if point.get("calculated_at") != entry["calculated_at"]]
    history.append(entry)
    history.sort(key=lambda item: item["calculated_at"])
    trimmed_history = history[-MAX_HISTORY_POINTS:]
    HISTORY_FILE.write_text(json.dumps(trimmed_history, indent=2))


def _ensure_history_seed(snapshot: RASSnapshot) -> None:
    """Ensure a baseline history file exists for legacy installs."""
    if not HISTORY_FILE.exists():
        _append_history(snapshot)
