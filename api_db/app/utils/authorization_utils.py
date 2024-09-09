import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "")

# HTTPBearer security scheme
bearer_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Fetches the current user from the auth` service."""
    token = credentials.credentials
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/users/me/", headers={"Authorization": f"{token}"})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Invalid credentials")
    return response.json()

async def get_current_admin_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Fetches the current admin user from the auth service."""
    print('inside get_current_admin_user')
    token = credentials.credentials
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/users/me/admin/", headers={"Authorization": f"{token}"})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Not enough permissions")
    return response.json()


async def create_error_log_in_auth_service(error_log: dict, token: str):
    """Calls the `auth` service to create an error log."""
    print('inside of create_error_log_in_auth_service')
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AUTH_SERVICE_URL}/error_logs/",
            json=error_log,
            headers={"Authorization": f"{token}"}
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to create error log: {response.text}"
        )
    return response.json()