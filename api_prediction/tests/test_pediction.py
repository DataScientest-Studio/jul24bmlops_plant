# import pytest
# from fastapi.testclient import TestClient
# from unittest.mock import patch, MagicMock

# from app.main import app
# from app.utils.authorization_utils import get_current_user, get_token_from_request
# from app.database.db import get_db
# from app.utils.prediction_utils import model

# client = TestClient(app)

# # Mock data for testing
# MOCK_USER = {"user_id": 1, "is_admin": False}
# MOCK_PREDICTION = {
#     "predicted_class": "Apple___Apple_scab",
#     "confidence": 0.95,
#     "top_5_predictions": [
#         {"class_name": "Apple___Apple_scab", "confidence": 0.95},
#         {"class_name": "Apple___Black_rot", "confidence": 0.03},
#         {"class_name": "Apple___Cedar_apple_rust", "confidence": 0.01},
#         {"class_name": "Apple___healthy", "confidence": 0.005},
#         {"class_name": "Background_without_leaves", "confidence": 0.004},
#     ],
#     "message": "Prediction was successful and metadata has been saved"
# }

# @pytest.fixture
# def mock_user():
#     """Mock the authentication functions"""
#     with patch("app.utils.authorization_utils.get_current_user", return_value=MOCK_USER):
#         with patch("app.utils.authorization_utils.get_token_from_request", return_value="mock-token"):
#             yield

# @pytest.fixture
# def mock_db_session():
#     """Mock the database session dependency"""
#     mock_db = MagicMock()
#     with patch("app.database.db.get_db", return_value=mock_db):
#         yield mock_db

# @pytest.fixture
# def mock_model_predict():
#     """Mock the model prediction"""
#     with patch.object(model, "predict", return_value=[[0.95, 0.03, 0.01, 0.005, 0.004]]):
#         yield

# def test_prediction_endpoint(mock_user, mock_db_session, mock_model_predict):
#     # Create a mock image file
#     file = {'file': ('test_image.jpg', b'test image content', 'image/jpeg')}

#     # Perform the POST request to the /predict endpoint
#     response = client.post("/predict", files=file)

#     # Assert the response
#     # assert response.status_code == 200
#     data = response.json()
#     print('value of data')
#     print(data)
#     assert data["predicted_class"] == MOCK_PREDICTION["predicted_class"]
#     assert data["confidence"] == MOCK_PREDICTION["confidence"]
#     assert data["message"] == MOCK_PREDICTION["message"]
#     assert len(data["top_5_predictions"]) == 5
#     assert data["top_5_predictions"][0]["class_name"] == MOCK_PREDICTION["top_5_predictions"][0]["class_name"]
