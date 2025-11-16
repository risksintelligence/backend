import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.core.config import get_settings
from app.services.transparency import record_update
from app.services.impact import update_snapshot

settings = get_settings()
SUBMISSIONS_FILE = settings.data_dir / "submissions.json"
DEFAULT_DATA: List[Dict[str, str]] = []


def _load_submissions() -> List[Dict[str, str]]:
    if not SUBMISSIONS_FILE.exists():
        SUBMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUBMISSIONS_FILE.write_text(json.dumps(DEFAULT_DATA, indent=2))
    return json.loads(SUBMISSIONS_FILE.read_text())


def _save_submissions(entries: List[Dict[str, str]]) -> None:
    SUBMISSIONS_FILE.write_text(json.dumps(entries, indent=2))


def add_submission(payload: Dict[str, str]) -> Dict[str, str]:
    entries = _load_submissions()
    submission = {
        "id": str(len(entries) + 1),
        "submitted_at": datetime.utcnow().isoformat() + 'Z',
        "status": "pending",
        **payload,
    }
    entries.insert(0, submission)
    _save_submissions(entries)
    record_update(f"New submission: {payload.get('title', 'untitled')}")
    update_snapshot({"analyses": 0.01})
    return submission


def list_submissions() -> List[Dict[str, str]]:
    return _load_submissions()


def update_submission_status(submission_id: str, status: str) -> Dict[str, str]:
    entries = _load_submissions()
    for entry in entries:
        if entry['id'] == submission_id:
            entry['status'] = status
            entry['reviewed_at'] = datetime.utcnow().isoformat() + 'Z'
            break
    else:
        raise ValueError('Submission not found')
    _save_submissions(entries)
    record_update(f"Submission {submission_id} marked {status}")
    return entry
