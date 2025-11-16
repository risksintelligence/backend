import json
from pathlib import Path
from typing import List, Dict

from app.core.config import get_settings


# Mapping from series names to file names
LOCAL_SERIES_MAPPING = {
    "VIX": "vixcls",
    "VIXCLS": "vixcls", 
    "PMI": "pmi",
    "CREDIT_SPREAD": "credit_spread",
    "YIELD_CURVE": "yield_curve",
    "BALTIC_DRY": "baltic_dry", 
    "BDIY": "baltic_dry",
    "WTI_OIL": "wti_oil",
    "UNEMPLOYMENT": "unemployment",
    "FREIGHT_DIESEL": "wti_oil",  # Using WTI as proxy for diesel
}


def fetch_local_series(series_id: str) -> List[Dict[str, str]]:
    settings = get_settings()
    
    # Map series ID to file name
    file_name = LOCAL_SERIES_MAPPING.get(series_id, series_id.lower())
    file_path = settings.data_dir / "series" / f"{file_name}.json"
    
    if not file_path.exists():
        # Try direct mapping as fallback
        file_path = settings.data_dir / "series" / f"{series_id.lower()}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing local series file for {series_id}: {file_path}")
    
    return json.loads(file_path.read_text())


# Alias for backward compatibility  
fetch_series = fetch_local_series
