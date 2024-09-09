from fastapi import APIRouter, Depends, HTTPException, Request
from ..utils.db_utils import backup_database, restore_database
from ..schemas.db_schema import BackupResponse, RestoreResponse
from ..database.db import get_db
from sqlalchemy.orm import Session
from ..utils.authorization_utils import get_current_admin_user, get_current_user, create_error_log_in_auth_service
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer

router = APIRouter()

# a lot of operation of Database is handled programmatically across the components without the need of user interaction.

bearer_scheme = HTTPBearer()

# Define your endpoint with the security scheme
@router.get("/test/")
async def test_auth(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
    ):
    token = credentials.credentials
    print('inside test_auth')
    print('value of current_user:', current_user)
    
    # Example error log data
    error_log_data = {
        "error_type": "Example Error Type",
        "error_message": "This is a test error message",
        "the_model_id": None,
        "user_id": current_user['user_id']
    }
    
    print('value of token')
    print(token)
    try:
        created_error_log = await create_error_log_in_auth_service(error_log_data, token)
        print('Created error log:', created_error_log)
    except HTTPException as e:
        print(f"Failed to create error log: {str(e)}")
        raise e
    return {"message": "Welcome to the Auth API TESTTTTTT!"}

# from fastapi import status, Request, HTTPException
# def get_token_from_request(request: Request):
#     """Extract the Bearer token from the request headers."""
#     authorization: str = request.headers.get("Authorization")

@router.post("/backup", response_model=BackupResponse)
async def backup_db(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    try:
        # backup_database()
        return {"message": "Database backup successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to backup the database: {str(e)}")

@router.post("/restore", response_model=RestoreResponse)
async def restore_db(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    try:
        # restore_database()
        return {"message": "Database restored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore the database: {str(e)}")
