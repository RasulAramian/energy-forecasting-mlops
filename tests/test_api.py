from fastapi.testclient import TestClient
import api.main
from api.main import app


def test_health_check_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "scaler_loaded" in data


def test_predict_validation_error():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"sequence": [1.0] * 10}
        )

        assert response.status_code == 422


def test_predict_success():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"sequence": [15000.0] * 24}
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "success"
        assert isinstance(data["predicted_value"], (int, float))


def test_predict_with_missing_model_returns_503(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setattr(api.main, "is_model_loaded", False)
        monkeypatch.setattr(api.main, "is_scaler_loaded", False)

        response = client.post(
            "/predict",
            json={"sequence": [15000.0] * 24}
        )

        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()
