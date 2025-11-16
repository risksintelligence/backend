import json
from pathlib import Path
from typing import List, Dict

from app.core.config import get_settings


def fetch_local_series(series_id: str) -> List[Dict[str, str]]:
    settings = get_settings()
    file_path = settings.data_dir / "series" / f"{series_id.lower()}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing local series file for {series_id}: {file_path}")
    return json.loads(file_path.read_text())
