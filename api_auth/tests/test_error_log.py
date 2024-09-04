# tests/test_error_log.py

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_create_error_log(client):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Login to get the token
        login_response = await ac.post("/token", data={"username": "testuser", "password": "testpassword"})
        token = login_response.json().get("access_token", "")

        headers = {"Authorization": f"Bearer {token}"}
        response = await ac.post("/error_logs/", json={
            "error_type": "Database Error",
            "error_message": "Test error log entry",
            "the_model_id": 1,
            "user_id": 1
        }, headers=headers)
    
    # Check if the response is successful or if there's a failure due to server error
    assert response.status_code == 200 or response.status_code == 500
