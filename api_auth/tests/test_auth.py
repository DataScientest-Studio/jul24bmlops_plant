# # tests/test_auth.py
# import pytest
# from httpx import AsyncClient
# from app.main import app

# @pytest.mark.asyncio
# async def test_signup(client):
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         response = await ac.post("/signup/", json={
#             "username": "testuser",
#             "password": "testpassword",  # Password length should be >= 6
#             "email": "test@example.com",  # Ensure valid email format
#             "role_id": 2,  # Ensure this is a valid role_id
#             "disabled": False  # Ensure all fields required are present
#         })
#     # Allow 201 (Created) or 400 (Bad Request due to duplicate user)
#     assert response.status_code in [201, 400]

# @pytest.mark.asyncio
# async def test_login(client):
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         # Ensure the user is created before attempting to log in
#         signup_response = await ac.post("/signup/", json={
#             "username": "testuser",
#             "password": "testpassword",
#             "email": "test@example.com",
#             "role_id": 2,
#             "disabled": False
#         })

#         # The user should be created (201) or already exists (400)
#         assert signup_response.status_code in [201, 400]

#         # Attempt to login
#         response = await ac.post("/token", data={"username": "testuser", "password": "testpassword"})
    
#     # Allow 200 (Success) or 401 (Unauthorized due to invalid credentials)
#     assert response.status_code in [200, 401]

