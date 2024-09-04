# # tests/conftest.py

# import os
# import pytest
# from fastapi.testclient import TestClient
# from unittest.mock import MagicMock, patch
# from app.main import app
# from app.database.db import get_db
# from app.database.tables import User
# from app.utils.authorization_utils import get_current_user  # Import the function to mock

# # Set the environment variable to indicate that tests are running
# os.environ["TESTING"] = "true"

# # Mock database session
# def mock_get_db():
#     db = MagicMock()

#     # Mock user data to simulate a real user in the database
#     mock_user = User(
#         user_id=1,
#         username="testuser",
#         hashed_password="$2b$12$KIXGbkUvKzGhs7T5kzx3xOCuWCVVQJrbZf1HmrkCPLvFrp4yy8k2W",  # bcrypt hash for "testpassword"
#         email="test@example.com",
#         role_id=2,
#         disabled=False
#     )

#     db.query.return_value.filter.return_value.first.return_value = mock_user

#     yield db

# # Override the FastAPI dependency that provides the database session
# app.dependency_overrides[get_db] = mock_get_db

# # Mock the `get_current_user` function to return a mock user
# @pytest.fixture(autouse=True)
# def mock_get_current_user():
#     with patch('app.utils.authorization_utils.get_current_user') as mock:
#         mock.return_value = {
#             "user_id": 1,
#             "username": "testuser",
#             "email": "test@example.com",
#             "role_id": 2,
#             "disabled": False
#         }
#         yield mock

# # Fixture to provide a test client for the FastAPI app
# @pytest.fixture(scope="module")
# def client():
#     with TestClient(app) as c:
#         yield c

# # Fixture to provide a test client that gets an authenticated token
# @pytest.fixture(scope="module")
# def client_authenticated(client):
#     response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = response.json().get("access_token", "")

#     client.headers.update({"Authorization": f"Bearer {token}"})
#     return client
