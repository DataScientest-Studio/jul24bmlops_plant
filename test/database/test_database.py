import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from API.app.database.db import Base
from API.app.database.tables import User, Role, Prediction, ModelMetadata, ErrorLog, APIRequestLog, ABTestingResult
from fastapi.testclient import TestClient
from API.app.main import app  # Import your FastAPI app

# Create a new database engine for testing (SQLite in-memory database is used here for simplicity)
DATABASE_URL = "sqlite:///:memory:"  # For PostgreSQL, you would use a test database URL like "postgresql://user:password@localhost/test_db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the dependency in the app with the test session
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[app.get_db] = override_get_db

@pytest.fixture(scope="module")
def test_db():
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    # Drop tables after tests are done
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_create_user(test_db, client):
    # Create a new user
    new_role = Role(role_name="admin", role_description="Administrator role")
    test_db.add(new_role)
    test_db.commit()
    test_db.refresh(new_role)

    response = client.post(
        "/users/",
        json={"username": "testuser", "password": "testpassword", "email": "test@example.com", "role_id": new_role.role_id},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"

def test_get_user(test_db, client):
    # Get a user by ID
    response = client.get("/users/1")  # Assuming the first user has ID 1
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["username"] == "testuser"

def test_create_prediction(test_db, client):
    # Assuming a user and a model have already been created
    user = test_db.query(User).filter(User.username == "testuser").first()
    model = ModelMetadata(model_name="test_model", accuracy=0.9, training_loss=0.1, validation_loss=0.2, training_params={})
    test_db.add(model)
    test_db.commit()
    test_db.refresh(model)

    response = client.post(
        "/predictions/",
        json={"user_id": user.user_id, "model_id": model.model_id, "image_path": "test_image.jpg", "prediction": {"class": "plant"}, "confidence": 0.95},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["prediction"]["class"] == "plant"
    assert data["confidence"] == 0.95

def test_get_prediction(test_db, client):
    # Get the prediction by ID
    response = client.get("/predictions/1")  # Assuming the first prediction has ID 1
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == 1
    assert data["confidence"] == 0.95

def test_create_error_log(test_db, client):
    # Create an error log entry
    user = test_db.query(User).filter(User.username == "testuser").first()
    model = test_db.query(ModelMetadata).first()

    response = client.post(
        "/error_logs/",
        json={"error_type": "Inference Error", "error_message": "Failed to predict", "user_id": user.user_id, "model_id": model.model_id},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["error_type"] == "Inference Error"
    assert data["error_message"] == "Failed to predict"

def test_get_error_logs(test_db, client):
    # Get all error logs
    response = client.get("/error_logs/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_create_model_metadata(test_db, client):
    # Create a new model metadata entry
    response = client.post(
        "/models/",
        json={
            "model_name": "plant_detection_model",
            "version": "1.0",
            "training_data": "Plant dataset version 1.0",
            "training_start_time": "2023-08-01T00:00:00Z",
            "training_end_time": "2023-08-02T00:00:00Z",
            "accuracy": 0.92,
            "training_loss": 0.15,
            "validation_loss": 0.18,
            "training_params": {"batch_size": 32, "learning_rate": 0.001}
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["model_name"] == "plant_detection_model"
    assert data["accuracy"] == 0.92

def test_get_model_metadata(test_db, client):
    # Get the model metadata by ID
    response = client.get("/models/1")  # Assuming the first model has ID 1
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == 1
    assert data["model_name"] == "plant_detection_model"

def test_create_ab_testing_result(test_db, client):
    # Assuming two models have already been created
    model_a = test_db.query(ModelMetadata).filter(ModelMetadata.model_name == "plant_detection_model").first()
    model_b = ModelMetadata(model_name="plant_detection_model_v2", accuracy=0.94, training_loss=0.12, validation_loss=0.15, training_params={})
    test_db.add(model_b)
    test_db.commit()
    test_db.refresh(model_b)

    response = client.post(
        "/ab_tests/",
        json={
            "test_name": "Model Comparison",
            "model_a_id": model_a.model_id,
            "model_b_id": model_b.model_id,
            "metric_name": "accuracy",
            "model_a_metric_value": 0.92,
            "model_b_metric_value": 0.94,
            "winning_model_id": model_b.model_id
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["test_name"] == "Model Comparison"
    assert data["winning_model_id"] == model_b.model_id

def test_get_ab_testing_result(test_db, client):
    # Get the A/B testing result by ID
    response = client.get("/ab_tests/1")  # Assuming the first A/B test has ID 1
    assert response.status_code == 200
    data = response.json()
    assert data["test_id"] == 1
    assert data["winning_model_id"] == 2  # Assuming model_b has ID 2

def test_create_api_request_log(test_db, client):
    # Create a new API request log
    user = test_db.query(User).filter(User.username == "testuser").first()

    response = client.post(
        "/api_request_logs/",
        json={
            "endpoint": "/models/",
            "request_method": "POST",
            "request_body": '{"model_name": "plant_detection_model", "accuracy": 0.92}',
            "response_status": 201,
            "response_time_ms": 120.5,
            "user_id": user.user_id,
            "ip_address": "192.168.1.1"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["endpoint"] == "/models/"
    assert data["response_status"] == 201

def test_get_api_request_log(test_db, client):
    # Get the API request log by ID
    response = client.get("/api_request_logs/1")  # Assuming the first API log has ID 1
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == 1
    assert data["endpoint"] == "/models/"


