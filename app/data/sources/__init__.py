from typing import Callable, List, Dict

from app.data.fetchers.local import fetch_local_series
from app.data.fetchers.fred import fetch_series

SourceFunc = Callable[[str], List[Dict[str, str]]]

SOURCE_MAP = {
    "local": fetch_local_series,
    "fred": fetch_series,
}


def get_source(provider: str) -> SourceFunc:
    if provider not in SOURCE_MAP:
        raise ValueError(f"Unsupported provider {provider}")
    return SOURCE_MAP[provider]
