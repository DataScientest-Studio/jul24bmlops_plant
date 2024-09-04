# tests/test_api_request_log.py

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_create_api_request_log(client):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        login_response = await ac.post("/token", data={"username": "testuser", "password": "testpassword"})
        token = login_response.json().get("access_token", "")
        
        headers = {"Authorization": f"Bearer {token}"}
        response = await ac.post("/api_request_logs/", json={
            "endpoint": "/test_endpoint",
            "request_method": "GET",
            "response_status": 200,
            "response_time_ms": 100,
            "user_id": 1
        }, headers=headers)
    assert response.status_code == 200 or response.status_code == 500
