from fastapi.testclient import TestClient
from app.main import app
import pytest
import os

client = TestClient(app)

# Test Model Loading
def test_model_initialization():
    """Test if the model loads successfully"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# Test Prediction Endpoints
@pytest.mark.parametrize("input_text,expected_sentiment", [
    ("I love this product! It's amazing.", "POSITIVE"),
    ("I hate this product! It's terrible.", "NEGATIVE"),
    ("This is neither good nor bad.", "NEGATIVE"),  # Neutral often defaults to negative
])
def test_predict_endpoint(input_text, expected_sentiment):
    """Test sentiment prediction with various inputs"""
    response = client.post(
        "/predict",
        json={"text": input_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == expected_sentiment
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1

# Test Error Handling
def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Text cannot be empty"}

def test_missing_text_field():
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422  # FastAPI validation error

# Test Retraining Endpoint
def test_retrain_endpoint():
    """Test model retraining endpoint"""
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert "status" in response.json()

# Test Metrics Endpoint
def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    # First make a prediction to generate metrics
    client.post("/predict", json={"text": "Test for metrics"})
    
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inference_requests_total" in response.text