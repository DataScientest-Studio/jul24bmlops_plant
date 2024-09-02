from fastapi import status, Request, HTTPException
import httpx


# Base URL for the authentication service
AUTH_SERVICE_URL = "http://api_auth:8000"

async def get_current_user(token: str):
    """Fetches the current user from the `api_auth` service."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/users/me", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Invalid credentials")
    return response.json()

async def get_current_admin_user(token: str):
    """Fetches the current admin user from the `api_auth` service."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/users/me/admin", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Not enough permissions")
    return response.json()

def get_token_from_request(request: Request):
    """Extract the Bearer token from the request headers."""
    authorization: str = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization code")
    return authorization.split(" ")[1]


