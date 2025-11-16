from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InsightBundle:
    headline: str
    geri_score: float
    band: str
    color: str
    drivers: Any
    regime: str
    regime_probabilities: Dict[str, float]
    forecast: Dict[str, float]
    anomaly: Dict[str, float]
    ras: Dict[str, Any]

    def to_markdown(self) -> str:
        drivers_text = "\n".join(
            [f"- {d['component']}: {d['contribution']}" for d in self.drivers]
        )
        regime_probs = ", ".join(
            [f"{k}: {v*100:.1f}%" for k, v in self.regime_probabilities.items()]
        )
        return f"""# RRIO Daily Flash\n\n**Headline:** {self.headline}\n\n- GRII Score: **{self.geri_score}** ({self.band}, {self.color})\n- Regime: **{self.regime}** ({regime_probs})\n- Forecast ΔGERI: {self.forecast.get('delta')} (p>5 = {self.forecast.get('p_gt_5')})\n- Anomaly Score: {self.anomaly.get('score')} ({self.anomaly.get('classification')})\n- RAS Composite: {self.ras.get('composite')}\n\n## Drivers\n{drivers_text}\n"""
