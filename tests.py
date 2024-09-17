from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Sample test data
sample_text = {"text": "Sample text for classification"}

# Test for successful prediction
def test_predict_success():
    response = client.post("/predict", json=sample_text)
    assert response.status_code == 200
    
    response_json = response.json()
    assert "text" in response_json
    assert "predicted_class_name" in response_json

    # Ensure the text matches the input
    assert response_json["text"] == sample_text["text"]

# Test for missing input
def test_predict_missing_input():
    # Sending an empty JSON should trigger a 422 error (Unprocessable Entity)
    response = client.post("/predict", json={})
    
    assert response.status_code == 422  

# Test for incorrect input format
def test_predict_incorrect_input():
    # Sending an invalid input format
    invalid_input = {"invalid_field": "This should fail"}
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422  
