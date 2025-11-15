import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SNAPSHOT_FILE = DATA_DIR / "impact_snapshot.json"

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
        return RASSnapshot(
            composite=payload["composite"],
            components=payload["components"],
            calculated_at=datetime.fromisoformat(payload["calculated_at"].replace("Z", "")),
        )
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
