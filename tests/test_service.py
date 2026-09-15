"""Comprehensive unit and integration test suite for the FastAPI Crypto Fraud service."""

import sys
from pathlib import Path

# Ensure root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from service.main import app, predictor


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    """Ensure artifacts are loaded before tests run."""
    predictor.load()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["model_loaded"] is True
    assert "dashboard_ui" in data


def test_dashboard_endpoint(client: TestClient):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "CryptoFraud" in response.text


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] >= 165
    assert data["explainer_ready"] is True
    assert 0.0 < data["current_threshold"] < 1.0


def test_predict_single_transaction_licit(client: TestClient):
    payload = {
        "txId": "tx_licit_001",
        "features": {
            "feat_1": 0.05,
            "feat_2": -0.12,
            "graph_in_degree": 2.0,
            "graph_out_degree": 1.0,
            "graph_pagerank": 0.0001,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["txId"] == "tx_licit_001"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["is_illicit"], bool)
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["action"] in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_FREEZE"]
    assert data["latency_ms"] >= 0


def test_predict_single_transaction_high_risk(client: TestClient):
    payload = {
        "txId": "tx_illicit_999",
        "features": {
            "feat_1": 3.5,
            "feat_2": 4.1,
            "feat_3": 2.9,
            "graph_in_degree": 1.0,
            "graph_out_degree": 18.0,
            "graph_pagerank": 0.005,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["txId"] == "tx_illicit_999"
    assert data["fraud_probability"] > 0.30
    assert data["risk_tier"] in ["MEDIUM", "HIGH"]


def test_predict_batch(client: TestClient):
    payload = {
        "transactions": [
            {
                "txId": f"batch_tx_{i}",
                "features": {
                    "feat_1": float(i) * 0.5,
                    "graph_out_degree": float(i) * 3,
                },
            }
            for i in range(5)
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 5
    assert len(data["predictions"]) == 5
    assert "summary_by_tier" in data


def test_explain_endpoint(client: TestClient):
    payload = {
        "txId": "tx_explain_demo",
        "features": {
            "feat_1": 2.8,
            "feat_2": 3.2,
            "graph_out_degree": 15.0,
            "graph_pagerank": 0.003,
        },
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["txId"] == "tx_explain_demo"
    assert "baseline_expected_value" in data
    assert "top_risk_drivers" in data
    assert "top_mitigating_factors" in data
    assert isinstance(data["top_risk_drivers"], list)
    assert len(data["compliance_reason_summary"]) > 0


def test_threshold_lifecycle(client: TestClient):
    # 1. Get current threshold
    get_res = client.get("/threshold")
    assert get_res.status_code == 200
    orig_thresh = get_res.json()["current_threshold"]

    # 2. Update threshold
    new_thresh = 0.42
    update_payload = {
        "threshold": new_thresh,
        "low_risk_threshold": 0.18,
        "high_risk_threshold": 0.72,
    }
    post_res = client.post("/threshold", json=update_payload)
    assert post_res.status_code == 200
    assert post_res.json()["current_threshold"] == new_thresh
    assert post_res.json()["low_risk_threshold"] == 0.18
    assert post_res.json()["high_risk_threshold"] == 0.72

    # 3. Verify validation error if high <= low
    invalid_payload = {
        "low_risk_threshold": 0.75,
        "high_risk_threshold": 0.20,
    }
    err_res = client.post("/threshold", json=invalid_payload)
    assert err_res.status_code == 422  # Pydantic validation error


def test_invalid_transaction_payload(client: TestClient):
    # Missing required 'features' dict
    bad_payload = {"txId": "corrupt_tx"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
