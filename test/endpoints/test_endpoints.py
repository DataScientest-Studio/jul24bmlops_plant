import pytest
from fastapi.testclient import TestClient
from API.app.main import app 

import os

# the_main = os.path.join(os.path.dirname(__file__), '../../API/app/main')
# from the_main import app

client = TestClient(app)

# Example test for the user creation endpoint
def test_create_user():
    response = client.post(
        "/users/",
        json={"username": "testuser", "password": "testpassword", "email": "test@example.com", "role_id": 1},
    )
    assert response.status_code == 201
    assert response.json()["username"] == "testuser"

# Example test for the user retrieval endpoint
def test_get_user():
    user_id = 1  # Example user ID, adjust based on your test setup
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["user_id"] == user_id

# Example test for the prediction endpoint
def test_create_prediction():
    response = client.post(
        "/predictions/",
        json={"user_id": 1, "model_id": 1, "image_path": "test_image.jpg", "prediction": {"class": "plant"}, "confidence": 0.95},
    )
    assert response.status_code == 201
    assert response.json()["prediction"]["class"] == "plant"

# Example test for the model metadata endpoint
def test_get_model_metadata():
    model_id = 1  # Example model ID
    response = client.get(f"/models/{model_id}")
    assert response.status_code == 200
    assert response.json()["model_id"] == model_id

# Example test for the error logs endpoint
def test_get_error_logs():
    response = client.get("/error_logs/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Example test for the AB testing results endpoint
def test_get_ab_testing_results():
    response = client.get("/ab_testing/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Example test for the API request logs endpoint
def test_get_api_request_logs():
    response = client.get("/api_request_logs/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)