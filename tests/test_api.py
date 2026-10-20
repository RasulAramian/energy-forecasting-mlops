from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "scaler_loaded" in data


def test_predict_endpoint_validation():
    response = client.post(
        "/predict",
        json={"sequence": [1.0] * 10}
    )

    assert response.status_code == 422

    data = response.json()
    assert "detail" in data
