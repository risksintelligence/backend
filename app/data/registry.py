from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SeriesMetadata:
    id: str
    provider: str
    frequency: str
    direction: str  # high_is_risk or low_is_risk


SERIES_REGISTRY: Dict[str, SeriesMetadata] = {
    "VIX": SeriesMetadata(id="VIXCLS", provider="fred", frequency="daily", direction="high_is_risk"),
    "PMI": SeriesMetadata(id="NAPM", provider="fred", frequency="monthly", direction="low_is_risk"),
    "CREDIT_SPREAD": SeriesMetadata(id="BAA10YM", provider="fred", frequency="daily", direction="high_is_risk"),
}
