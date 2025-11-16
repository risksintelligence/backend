from datetime import datetime
from typing import Dict, List

from app.db import SessionLocal
from app.models import ObservationModel  # placeholder for actual judging table

_judging_log: List[Dict[str, str]] = []


def log_judging_activity(submission_id: str, judge: str, decision: str) -> None:
    _judging_log.append({
        'submission_id': submission_id,
        'judge': judge,
        'decision': decision,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })


def get_judging_log() -> List[Dict[str, str]]:
    return _judging_log
