import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_anomalies_contract():
    resp = client.get("/api/v1/anomalies/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert "anomalies" in data and isinstance(data["anomalies"], list)
    assert "summary" in data
    if data["anomalies"]:
        alert = data["anomalies"][0]
        for key in ["id", "severity", "message", "driver", "timestamp"]:
            assert key in alert


def test_provider_health_contract():
    resp = client.get("/api/v1/monitoring/provider-health")
    assert resp.status_code == 200
    data = resp.json()
    for key in ["nodes", "criticalPaths", "summary", "updatedAt", "providerHealth"]:
        assert key in data
    if data["nodes"]:
        node = data["nodes"][0]
        for key in ["id", "name", "sector", "risk"]:
            assert key in node


def test_transparency_freshness_contract():
    resp = client.get("/api/v1/transparency/data-freshness")
    assert resp.status_code == 200
    data = resp.json()
    for key in ["cache_layers", "overall_status", "series_freshness", "compliance"]:
        assert key in data


def test_model_inventory_contract():
    resp = client.get("/api/v1/ai/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data and isinstance(data["models"], list)


def test_explainability_contract():
    resp = client.get("/api/v1/ai/explainability")
    assert resp.status_code == 200
    data = resp.json()
    assert "regime" in data and "forecast" in data
