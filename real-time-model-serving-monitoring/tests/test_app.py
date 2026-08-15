from fastapi.testclient import TestClient

from src.ml_monitoring.app import app, model_service


client = TestClient(app)


def sample_features():
    return [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    ]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_count"] == 30


def test_predict_endpoint():
    response = client.post("/predict", json={"features": sample_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"malignant", "benign"}
    assert 0 <= body["probability_malignant"] <= 1
    assert body["latency_ms"] >= 0


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_batch_predict_endpoint():
    response = client.post(
        "/batch_predict",
        json={"items": [{"features": sample_features()}, {"features": sample_features()}]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_drift_endpoint():
    rows = model_service.baseline_rows[:10]
    response = client.post("/monitor/drift", json={"rows": rows})
    assert response.status_code == 200
    body = response.json()
    assert body["rows_checked"] == 10
    assert len(body["features"]) == 30


def test_retraining_decision_endpoint():
    rows = model_service.baseline_rows[:10]
    response = client.post("/monitor/retraining_decision", json={"rows": rows})
    assert response.status_code == 200
    assert response.json()["action"] in {"monitor", "collect_more_traffic", "retrain_candidate"}


def test_shadow_agreement_endpoint():
    response = client.post(
        "/monitor/shadow_agreement",
        json={
            "champion_probabilities": [0.1, 0.2, 0.8],
            "challenger_probabilities": [0.12, 0.18, 0.77],
            "tolerance": 0.1,
        },
    )
    assert response.status_code == 200
    assert response.json()["verdict"] == "safe_shadow"
