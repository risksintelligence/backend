from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SeriesMetadata:
    id: str
    provider: str
    frequency: str
    direction: str  # high_is_risk or low_is_risk


SERIES_REGISTRY: Dict[str, SeriesMetadata] = {
    # Finance pillar
    "VIX": SeriesMetadata(id="VIXCLS", provider="fred", frequency="daily", direction="high_is_risk"),
    "YIELD_CURVE": SeriesMetadata(id="T10Y2Y", provider="fred", frequency="daily", direction="low_is_risk"),
    "CREDIT_SPREAD": SeriesMetadata(id="BAA10YM", provider="fred", frequency="daily", direction="high_is_risk"),
    
    # Supply chain pillar
    "FREIGHT_DIESEL": SeriesMetadata(id="EPD2DXL0", provider="eia", frequency="weekly", direction="high_is_risk"),
    "BALTIC_DRY": SeriesMetadata(id="BDIY", provider="local", frequency="daily", direction="high_is_risk"),
    
    # Macro pillar  
    "PMI": SeriesMetadata(id="NAPM", provider="fred", frequency="monthly", direction="low_is_risk"),
    "WTI_OIL": SeriesMetadata(id="DCOILWTICO", provider="fred", frequency="daily", direction="high_is_risk"),
    "UNEMPLOYMENT": SeriesMetadata(id="UNEMPLOYMENT", provider="bls", frequency="monthly", direction="high_is_risk"),
}
