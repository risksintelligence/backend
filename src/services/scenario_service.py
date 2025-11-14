"""Scenario Studio simulation and alert stubs."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from src.analytics.geri import IndicatorResult
from src.services.geri_service import GERISnapshotService, seed_demo_data
from src.ml.inference.service import MLInferenceService
from src.services.alerts_delivery import AlertDeliveryService
from src.services.alert_repository import AlertRepository
from src.monitoring.metrics import SCENARIO_RUNS_TOTAL, ALERT_DELIVERIES_TOTAL


@dataclass
class ScenarioResult:
    baseline: float
    scenario: float
    band: str
    delta: float
    explanation: str


class ScenarioService:
    def __init__(self, geri_service: GERISnapshotService) -> None:
        self._geri_service = geri_service
        seed_demo_data(self._geri_service)
        self._runs: List[Dict[str, object]] = []
        self._ml_service = None

    async def simulate(self, shocks: List[Dict[str, float]], horizon_hours: int) -> ScenarioResult:
        payload, _, _ = self._geri_service.build_payload()
        baseline = float(payload["value"])
        band = str(payload["band"])
        delta = sum(shock.get("delta_percent", 0) * 0.1 + shock.get("delta_points", 0) * 0.2 for shock in shocks)
        scenario = baseline + delta
        explanation = ", ".join(f"{shock['series']} adjusted" for shock in shocks) or "No shocks provided"
        entry = {
            "id": len(self._runs) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "baseline": baseline,
            "scenario": scenario,
            "band": band,
            "horizon_hours": horizon_hours,
            "shocks": shocks,
        }
        self._runs.append(entry)
        
        # Initialize ML service lazily
        if self._ml_service is None:
            try:
                self._ml_service = MLInferenceService()
            except Exception as e:
                import logging
                logging.warning(f"Failed to initialize MLInferenceService: {e}")
        
        # Try to get ML predictions with fallback
        try:
            if self._ml_service:
                ml_regime = await self._ml_service.predict_current_regime()
                ml_forecast = await self._ml_service.forecast_change(horizon_hours)
                entry["ml_context"] = {
                    "regime": ml_regime["regime"],
                    "confidence": ml_regime["confidence"],
                    "forecast_delta": ml_forecast["delta_geri_prediction"],
                }
                explanation = f"{explanation}. Regime: {ml_regime['regime']} (confidence {ml_regime['confidence']:.2f}), Δ forecast {ml_forecast['delta_geri_prediction']:.2f}"
            else:
                entry["ml_context"] = {"regime": "unknown", "confidence": 0.0, "forecast_delta": 0.0}
        except Exception as e:
            import logging
            logging.warning(f"ML prediction failed: {e}")
            entry["ml_context"] = {"regime": "unknown", "confidence": 0.0, "forecast_delta": 0.0}
        
        entry["explanation"] = explanation
        SCENARIO_RUNS_TOTAL.labels(channel="api").inc()
        return ScenarioResult(
            baseline=baseline,
            scenario=scenario,
            band=band,
            delta=delta,
            explanation=explanation,
        )

    def list_runs(self, limit: int = 20) -> List[Dict[str, object]]:
        return self._runs[-limit:][::-1]


class AlertService:
    def __init__(self, delivery: AlertDeliveryService, repository: AlertRepository) -> None:
        self._delivery = delivery
        self._repository = repository

    def subscribe(self, channel: str, address: str, conditions: List[Dict[str, object]]) -> Dict[str, object]:
        record = self._repository.create_subscription(channel, address, conditions)
        return {
            "id": record.id,
            "channel": record.channel,
            "address": record.address,
            "conditions": record.conditions,
            "created_at": record.created_at,
            "status": "active",
        }

    def list_subscriptions(self) -> List[Dict[str, object]]:
        return [
            {
                "id": record.id,
                "channel": record.channel,
                "address": record.address,
                "conditions": record.conditions,
                "created_at": record.created_at,
                "status": "active",
            }
            for record in self._repository.list_subscriptions()
        ]

    async def deliver_alerts(self, payload: dict) -> List[dict]:
        events = []
        for sub in self._repository.list_subscriptions():
            delivery = await self._delivery.deliver(sub.id, sub.channel, sub.address, payload)
            self._repository.save_delivery(
                delivery.subscription_id, delivery.channel, delivery.address, delivery.payload
            )
            ALERT_DELIVERIES_TOTAL.labels(channel=sub.channel).inc()
            events.append(delivery.__dict__)
        return events

    def deliveries(self) -> List[dict]:
        return [record.__dict__ for record in self._repository.deliveries()]


_SCENARIO_SERVICE: ScenarioService | None = None
_ALERT_SERVICE: AlertService | None = None
_DELIVERY_SERVICE: AlertDeliveryService | None = None
_ALERT_REPOSITORY: AlertRepository | None = None


def get_scenario_service() -> ScenarioService:
    global _SCENARIO_SERVICE
    if _SCENARIO_SERVICE is None:
        _SCENARIO_SERVICE = ScenarioService(GERISnapshotService())
    return _SCENARIO_SERVICE


def get_alert_service() -> AlertService:
    global _ALERT_SERVICE, _DELIVERY_SERVICE, _ALERT_REPOSITORY
    try:
        if _DELIVERY_SERVICE is None:
            _DELIVERY_SERVICE = AlertDeliveryService()
        if _ALERT_REPOSITORY is None:
            _ALERT_REPOSITORY = AlertRepository()
        if _ALERT_SERVICE is None:
            _ALERT_SERVICE = AlertService(_DELIVERY_SERVICE, _ALERT_REPOSITORY)
    except Exception as e:
        # Create minimal service that works without database
        import logging
        logging.warning(f"Failed to initialize full AlertService, using minimal fallback: {e}")
        if _ALERT_SERVICE is None:
            _ALERT_SERVICE = _create_fallback_alert_service()
    return _ALERT_SERVICE

def _create_fallback_alert_service() -> AlertService:
    """Create a minimal alert service that works without database connections."""
    class MockAlertRepository:
        def create_subscription(self, channel: str, address: str, conditions: list):
            from dataclasses import dataclass
            @dataclass
            class MockRecord:
                id: int = 1
                channel: str = channel
                address: str = address 
                conditions: list = conditions
                created_at: str = datetime.utcnow().isoformat()
            return MockRecord()
        
        def list_subscriptions(self): return []
        def save_delivery(self, *args): pass
        def deliveries(self): return []
    
    class MockDeliveryService:
        async def deliver(self, *args): 
            from src.services.alerts_delivery import DeliveredAlert
            return DeliveredAlert(1, "email", "test@example.com", {}, datetime.utcnow().isoformat(), "success")
    
    return AlertService(MockDeliveryService(), MockAlertRepository())
