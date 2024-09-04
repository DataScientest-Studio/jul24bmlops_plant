# tests/conftest.py

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app
from app.database.db import get_db
from app.database.tables import User  # Import User model

# Set the environment variable to indicate that tests are running
os.environ["TESTING"] = "true"

# Mock database session
def mock_get_db():
    # Create a mock session object
    db = MagicMock()

    # Mock user data to simulate a real user in the database
    mock_user = User(
        user_id=1,
        username="testuser",
        hashed_password="$2b$12$KIXGbkUvKzGhs7T5kzx3xOCuWCVVQJrbZf1HmrkCPLvFrp4yy8k2W",  # bcrypt hash for "testpassword"
        email="test@example.com",
        role_id=2,
        disabled=False
    )

    # Mock query behavior for finding a user by username
    db.query.return_value.filter.return_value.first.return_value = mock_user

    yield db

# Override the FastAPI dependency that provides the database session
app.dependency_overrides[get_db] = mock_get_db

# Fixture to provide a test client for the FastAPI app
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
