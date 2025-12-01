import datetime

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_governance_models_contract():
    resp = client.get("/api/v1/ai/governance/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "models" in data
    if data["models"]:
        model = data["models"][0]
        assert "model_id" in model
        assert "version" in model
        assert "model_type" in model


def test_compliance_report_contract():
    # Use first model if available; otherwise expect graceful handling
    models_resp = client.get("/api/v1/ai/governance/models")
    model_name = ""
    if models_resp.status_code == 200 and models_resp.json().get("models"):
        model_name = models_resp.json()["models"][0]["model_id"]
    else:
        model_name = "regime_classifier"

    resp = client.get(f"/api/v1/ai/governance/compliance-report/{model_name}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    report = data["compliance_report"]
    assert "overall_compliance_score" in report
    assert "nist_rmf_functions" in report


def test_explainability_audit_contract():
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=7)
    params = {
        "start_date": start.isoformat() + "Z",
        "end_date": end.isoformat() + "Z",
    }
    resp = client.get("/api/v1/ai/explainability/audit-log", params=params)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "audit_logs" in data
