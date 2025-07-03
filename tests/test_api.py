from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_sentiment_positive():
    response = client.post(
        "/predict",
        json={"text": "I love this product! It's amazing."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert "model" in data
    assert data["sentiment"] in ["POSITIVE", "NEGATIVE"]

def test_predict_sentiment_negative():
    response = client.post(
        "/predict",
        json={"text": "I hate this product! It's terrible."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["POSITIVE", "NEGATIVE"]

def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Text cannot be empty"}