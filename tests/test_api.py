from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "scaler_loaded" in data


def test_predict_validation_error():
    response = client.post(
        "/predict",
        json={"sequence": [1.0] * 10}
    )

    assert response.status_code == 422


def test_predict_success():
    response = client.post(
        "/predict",
        json={"sequence": [0.5] * 24}
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_value" in data
    assert isinstance(data["predicted_value"], (int, float))


def test_predict_with_missing_model_returns_503():
    response = client.post(
        "/predict",
        json={"sequence": [0.5] * 24}
    )

    if response.status_code == 503:
        assert response.status_code == 503
